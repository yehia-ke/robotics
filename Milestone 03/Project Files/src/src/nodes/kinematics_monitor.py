import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinematics as kin 


# ============================================================================
# ROS-free core (so it can be tested without ROS or a simulator)
# ============================================================================
def _slope(t, Y):
    """Least-squares slope dY/dt over samples (t[k], Y[k]); Y is (n, m)."""
    return np.polyfit(t - t[len(t) // 2], Y, 1)[0]


class MonitorCore:
    def __init__(self, arm, window=0.01, n_points=5):
        self.arm = arm                       # needs .position(q) and .twist(q, qd)
        self.window = window
        self.n_points = n_points
        self.buffer = deque(maxlen=400)      # (t, q, qd)

    def add(self, t, q, qd):
        self.buffer.append((float(t), np.array(q, float), np.array(qd, float)))

    def evaluate(self):
        """Compare equations vs sensed at the centre of the latest window.
        Returns a dict, or None until enough samples have arrived."""
        buf = self.buffer
        if len(buf) < 10:
            return None
        # evenly spaced samples counted back from the newest one -> the fit window is
        # symmetric about the centre sample, so slope and centre value refer to the same time
        dt = (buf[-1][0] - buf[-10][0]) / 9.0
        if dt <= 0:
            return None
        step = max(1, int(round(self.window / (self.n_points - 1) / dt)))
        if len(buf) < step * (self.n_points - 1) + 1:
            return None
        pts = [buf[-1 - step * (self.n_points - 1 - j)] for j in range(self.n_points)]
        t = np.array([p[0] for p in pts])
        if t[-1] - t[0] < 1e-6:
            return None
        Q = np.array([p[1] for p in pts])
        QD = np.array([p[2] for p in pts])
        c = self.n_points // 2
        q, qd = Q[c], QD[c]

        # joint acceleration estimated from the published velocities
        qdd = _slope(t, QD)

        # --- equations -------------------------------------------------------
        p_eq, _ = kin.fk(q)
        tw_eq = kin.fk_velocity(q, qd)
        acc_eq = kin.fk_acceleration(q, qd, qdd)

        # --- sensed (MuJoCo) ----------------------------------------------------
        p_s, _ = self.arm.position(q)
        TW_s = np.array([self.arm.twist(Q[k], QD[k]) for k in range(self.n_points)])
        tw_s = TW_s[c]
        acc_s = _slope(t, TW_s)

        # --- inverse: feed SENSED end-effector values to the inverse equations ------
        # Near a singularity (e.g. the all-zero start pose, q5 = 0) the inverse is not
        # defined, so the inverse check is skipped (nan) instead of crashing / blowing up.
        sigma_min = kin.singularity_measure(q)["sigma_min"]
        if sigma_min > kin.SINGULARITY_EPS:
            inv_vel_err = np.abs(kin.ik_velocity(q, tw_s, damped=False) - qd).max()
            inv_acc_err = np.abs(kin.ik_acceleration(q, qd, acc_s, damped=False) - qdd).max()
        else:
            inv_vel_err = inv_acc_err = float("nan")

        return {
            "t": t[c], "q": q, "qd": qd, "qdd": qdd,
            "eq": np.concatenate([p_eq, tw_eq, acc_eq]),
            "sensed": np.concatenate([p_s, tw_s, acc_s]),
            "inv_vel_err": inv_vel_err,
            "inv_acc_err": inv_acc_err,
            "sigma_min": sigma_min,
        }


# ============================================================================
# ROS2 node
# ============================================================================
def main():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray

    class MonitorNode(Node):
        def __init__(self):
            super().__init__("kinematics_monitor")
            for name, default in [("model_path", ""), ("state_topic", "/joint_state"),
                                  ("rate", 20.0), ("window", 0.01),
                                  ("tol_pos", 1e-3), ("tol_vel", 1e-3), ("tol_acc", 5e-2),
                                  ("csv_path", "")]:
                self.declare_parameter(name, default)
            g = lambda n: self.get_parameter(n).value  # noqa: E731

            model_path = str(g("model_path"))
            if not model_path:
                raise RuntimeError("pass -p model_path:=<scene.xml>")
            self.core = MonitorCore(kin.MujocoArm(model_path), window=float(g("window")))
            self.tol = (float(g("tol_pos")), float(g("tol_vel")), float(g("tol_acc")))

            self.csv = None
            if str(g("csv_path")):
                self.csv = open(str(g("csv_path")), "w")
                cols = ["t"] + [f"{k}_{n}" for k in ("eq", "sensed") for n in
                                ("x y z vx vy vz wx wy wz ax ay az alx aly alz".split())]
                self.csv.write(",".join(cols + ["inv_vel_err", "inv_acc_err"]) + "\n")

            self.pub = {k: self.create_publisher(Float64MultiArray, f"/kin/{k}", 10)
                        for k in ("equations", "sensed", "error")}
            self.create_subscription(JointState, str(g("state_topic")), self.on_state, 50)
            self.create_timer(1.0 / float(g("rate")), self.on_timer)
            self.worst = np.zeros(3)
            self.count = 0
            self.get_logger().info("kinematics_monitor running: waiting for /joint_state")

        def on_state(self, msg):
            if len(msg.position) < 6 or len(msg.velocity) < 6:
                return
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.core.add(t, msg.position[:6], msg.velocity[:6])

        def on_timer(self):
            r = self.core.evaluate()
            if r is None:
                return
            err = r["eq"] - r["sensed"]
            for key, vec in (("equations", r["eq"]), ("sensed", r["sensed"]), ("error", err)):
                m = Float64MultiArray()
                m.data = [float(x) for x in vec]
                self.pub[key].publish(m)

            e_pos = np.abs(err[:3]).max()
            e_vel = np.abs(err[3:9]).max()
            e_acc = np.abs(err[9:]).max()
            self.worst = np.maximum(self.worst, [e_pos, e_vel, e_acc])
            ok = e_pos < self.tol[0] and e_vel < self.tol[1] and e_acc < self.tol[2]
            if self.csv:
                self.csv.write(",".join(f"{x:.9g}" for x in
                                        [r["t"], *r["eq"], *r["sensed"],
                                         r["inv_vel_err"], r["inv_acc_err"]]) + "\n")
            self.count += 1
            if self.count % int(max(1, float(self.get_parameter("rate").value))) == 0:  # ~1 Hz
                self.get_logger().info(
                    f"{'PASS' if ok else 'FAIL'} | pos {e_pos:.1e} m  vel {e_vel:.1e} m/s  "
                    f"acc {e_acc:.1e} m/s^2 | inverse: qd {r['inv_vel_err']:.1e}  "
                    f"qdd {r['inv_acc_err']:.1e} | sigma_min {r['sigma_min']:.3f} | "
                    f"worst so far: {self.worst[0]:.1e}/{self.worst[1]:.1e}/{self.worst[2]:.1e}")

        def destroy_node(self):
            if self.csv:
                self.csv.close()
            super().destroy_node()

    rclpy.init()
    node = MonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()