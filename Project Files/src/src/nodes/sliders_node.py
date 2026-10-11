#!/usr/bin/env python3
"""
Joint sliders + end-effector (Cartesian) control for MuJoCo + ROS2
(woolfrey/mujoco_ros2, UFactory Lite6).

 * Sliders = target joint angles. The node reads /joint_state and publishes
   smooth, speed-limited commands on /joint_commands.
 * Forward kinematics: the live end-effector position is shown in the window.
 * Inverse kinematics: type X Y Z (metres, world frame), press "Go to point";
   the IK solution is applied to the sliders and the arm moves there.

Kinematics use the MuJoCo model itself, so you MUST pass the scene path:

    python3 joint_slider_gui.py --ros-args \
        -p mode:=position -p model_path:=/path/to/scene.xml

Optional: -p ee_name:=<site or body name>   (default: auto-detect)
Needs:    pip install mujoco numpy
"""

import math
import threading
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

N = 6

# FALLBACK limits (UFactory Lite6), only used if the model can't be loaded.
JOINTS = [
    ("Joint 1", -2 * math.pi, 2 * math.pi),
    ("Joint 2", -2.61799, 2.61799),
    ("Joint 3", -0.061087, 5.235988),
    ("Joint 4", -2 * math.pi, 2 * math.pi),
    ("Joint 5", -2.1642, 2.1642),
    ("Joint 6", -2 * math.pi, 2 * math.pi),
]


def clip(x, lo, hi):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Kinematics (forward + inverse) using the MuJoCo model
# ---------------------------------------------------------------------------
class Kinematics:
    EE_SITE_GUESSES = ["attachment_site", "ee_site", "end_effector", "tcp", "tool0"]

    def __init__(self, path, n, ee_name=""):
        import numpy as np
        import mujoco

        self.np, self.mj = np, mujoco
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        m = self.model

        # First n hinge/slide joints = the arm joints (same order as joint_state)
        jids = [j for j in range(m.njnt)
                if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                          int(mujoco.mjtJoint.mjJNT_SLIDE))]
        if len(jids) < n:
            raise RuntimeError(f"Model has only {len(jids)} joints, need {n}")
        jids = jids[:n]
        self.qadr = [int(m.jnt_qposadr[j]) for j in jids]
        self.dadr = [int(m.jnt_dofadr[j]) for j in jids]
        self.limits = []
        for j in jids:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"Joint {len(self.limits) + 1}"
            if m.jnt_limited[j]:
                lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
            else:
                lo, hi = -math.pi, math.pi
            self.limits.append((name, lo, hi))
        self.lo = np.array([l[1] for l in self.limits])
        self.hi = np.array([l[2] for l in self.limits])

        # End effector: named site/body, else a known site, else the last arm link
        self.site_id, self.body_id = -1, -1
        candidates = ([ee_name] if ee_name else []) + ([] if ee_name else self.EE_SITE_GUESSES)
        for nm in candidates:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, nm)
            if sid >= 0:
                self.site_id, self.ee_label = sid, f"site '{nm}'"
                break
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
            if bid >= 0:
                self.body_id, self.ee_label = bid, f"body '{nm}'"
                break
        if self.site_id < 0 and self.body_id < 0:
            self.body_id = int(m.jnt_bodyid[jids[-1]])
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, self.body_id)
            self.ee_label = f"body '{bname}' (last arm link; pass -p ee_name:=<site> for the tool tip)"

    def fk(self, q):
        """Forward kinematics -> (position[3], rotation_matrix[3x3]) in world frame."""
        np, mj = self.np, self.mj
        self.data.qpos[self.qadr] = q
        mj.mj_fwdPosition(self.model, self.data)
        if self.site_id >= 0:
            return (self.data.site_xpos[self.site_id].copy(),
                    self.data.site_xmat[self.site_id].reshape(3, 3).copy())
        return (self.data.xpos[self.body_id].copy(),
                self.data.xmat[self.body_id].reshape(3, 3).copy())

    def _jac(self):
        np, mj = self.np, self.mj
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        if self.site_id >= 0:
            mj.mj_jacSite(self.model, self.data, jp, jr, self.site_id)
        else:
            mj.mj_jacBody(self.model, self.data, jp, jr, self.body_id)
        return jp[:, self.dadr], jr[:, self.dadr]

    def _solve(self, target_pos, target_mat, seed, iters=300, lam=0.05):
        np = self.np
        q = np.clip(np.array(seed, dtype=float), self.lo, self.hi)
        for _ in range(iters):
            pos, mat = self.fk(q)
            e_pos = target_pos - pos
            if target_mat is not None:
                e_rot = 0.5 * sum(np.cross(mat[:, i], target_mat[:, i]) for i in range(3))
                if np.linalg.norm(e_pos) < 5e-4 and np.linalg.norm(e_rot) < 5e-3:
                    break
                e = np.concatenate([e_pos, e_rot])
                jp, jr = self._jac()
                J = np.vstack([jp, jr])
            else:
                if np.linalg.norm(e_pos) < 5e-4:
                    break
                e = e_pos
                J, _jr = self._jac()
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(J.shape[0]), e)
            big = np.max(np.abs(dq))
            if big > 0.3:
                dq *= 0.3 / big
            q = np.clip(q + dq, self.lo, self.hi)
        pos, mat = self.fk(q)
        pos_err = float(np.linalg.norm(target_pos - pos))
        rot_err = 0.0
        if target_mat is not None:
            rot_err = float(np.linalg.norm(
                0.5 * sum(np.cross(mat[:, i], target_mat[:, i]) for i in range(3))))
        return q, pos_err, rot_err

    def ik(self, target_pos, q_now, target_mat=None, restarts=10):
        """Inverse kinematics. Tries the current pose first, then random seeds.
        Returns (q, pos_err_m, rot_err, ok)."""
        np = self.np
        target_pos = np.array(target_pos, dtype=float)
        rng = np.random.default_rng(0)
        seeds = [np.array(q_now, dtype=float)] + [
            rng.uniform(self.lo, self.hi) for _ in range(restarts)]
        best = None
        for seed in seeds:
            q, pe, re = self._solve(target_pos, target_mat, seed)
            ok = pe < 1e-3 and (target_mat is None or re < 1e-2)
            if ok:
                return q, pe, re, True
            if best is None or pe + re < best[1] + best[2]:
                best = (q, pe, re)
        return best[0], best[1], best[2], False


# ---------------------------------------------------------------------------
# ROS2 node: target angles -> smooth commands
# ---------------------------------------------------------------------------
class JointController(Node):
    def __init__(self):
        super().__init__("joint_slider_gui")
        self.declare_parameter("command_topic", "/joint_commands")
        self.declare_parameter("state_topic", "/joint_state")
        self.declare_parameter("mode", "velocity")      # velocity | position
        self.declare_parameter("rate", 100.0)           # Hz
        self.declare_parameter("kp", 6.0)               # velocity-mode gain (1/s)
        self.declare_parameter("max_accel", 20.0)       # rad/s^2 (velocity mode)
        self.declare_parameter("max_speed", 5.0)        # rad/s (also set from GUI)
        self.declare_parameter("model_path", "")        # scene.xml (limits + kinematics)
        self.declare_parameter("ee_name", "")           # end-effector site/body name

        p = self.get_parameter
        self.mode = str(p("mode").value).lower()
        self.rate = float(p("rate").value)
        self.kp = float(p("kp").value)
        self.max_accel = float(p("max_accel").value)
        self.max_speed = float(p("max_speed").value)

        self.state_topic = str(p("state_topic").value)
        self.lock = threading.Lock()
        self.current = None
        self.target = [0.0] * N
        self.setpoint = [0.0] * N
        self.vel_cmd = [0.0] * N
        self.initialised = False

        self.pub = self.create_publisher(Float64MultiArray, p("command_topic").value, 10)
        self.create_subscription(JointState, self.state_topic, self.on_state, 10)
        self.create_timer(1.0 / self.rate, self.step)
        self.get_logger().info(
            f"mode={self.mode}, listening on {self.state_topic}, "
            f"publishing on {p('command_topic').value}")

    def on_state(self, msg):
        if len(msg.position) < N:
            return
        with self.lock:
            self.current = list(msg.position[:N])
            if not self.initialised:
                self.target = list(self.current)
                self.setpoint = list(self.current)
                self.initialised = True

    def resync(self):
        with self.lock:
            if self.current is None:
                return None
            self.target = list(self.current)
            self.setpoint = list(self.current)
            self.vel_cmd = [0.0] * N
            return list(self.current)

    def set_target(self, idx, value):
        with self.lock:
            self.target[idx] = float(value)

    def set_max_speed(self, value):
        with self.lock:
            self.max_speed = float(value)

    def step(self):
        dt = 1.0 / self.rate
        with self.lock:
            if not self.initialised:
                return
            out = [0.0] * N
            if self.mode == "position":
                for i in range(N):
                    d = self.target[i] - self.setpoint[i]
                    lim = self.max_speed * dt
                    self.setpoint[i] += clip(d, -lim, lim)
                    out[i] = self.setpoint[i]
            else:
                for i in range(N):
                    err = self.target[i] - self.current[i]
                    desired = clip(self.kp * err, -self.max_speed, self.max_speed)
                    if abs(err) < 0.002:
                        desired = 0.0
                    a = self.max_accel * dt
                    self.vel_cmd[i] += clip(desired - self.vel_cmd[i], -a, a)
                    out[i] = self.vel_cmd[i]
        msg = Float64MultiArray()
        msg.data = out
        self.pub.publish(msg)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class SliderGUI:
    def __init__(self, root, node, kin):
        self.root, self.node, self.kin = root, node, kin
        self.synced = False
        self.ticks = 0
        self.vars, self.scales, self.target_lbls, self.actual_lbls = [], [], [], []

        root.title("Lite6 Controller")
        root.minsize(640, 0)
        f = ttk.Frame(root, padding=12)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)

        self.status = ttk.Label(f, text="Waiting for joint state...", foreground="#b36b00")
        self.status.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        for i, (name, lo, hi) in enumerate(JOINTS):
            r = i + 1
            ttk.Label(f, text=name, width=8).grid(row=r, column=0, sticky="w", pady=5)
            var = tk.DoubleVar(value=clip(0.0, lo, hi))
            sc = ttk.Scale(f, from_=lo, to=hi, variable=var, orient="horizontal",
                           command=lambda v, k=i: self.on_move(k, v))
            sc.grid(row=r, column=1, sticky="ew", padx=8)
            sc.state(["disabled"])
            self.scales.append(sc)
            t = ttk.Label(f, text="target    0.0°", width=14, anchor="e")
            t.grid(row=r, column=2)
            a = ttk.Label(f, text="actual    0.0°", width=14, anchor="e")
            a.grid(row=r, column=3)
            self.vars.append(var)
            self.target_lbls.append(t)
            self.actual_lbls.append(a)

        r = N + 1
        ttk.Separator(f).grid(row=r, column=0, columnspan=4, sticky="ew", pady=10)
        ttk.Label(f, text="Max speed").grid(row=r + 1, column=0, sticky="w")
        self.speed_var = tk.DoubleVar(value=node.max_speed)
        ttk.Scale(f, from_=0.05, to=10.0, variable=self.speed_var, orient="horizontal",
                  command=self.on_speed).grid(row=r + 1, column=1, sticky="ew", padx=8)
        self.speed_lbl = ttk.Label(f, text=f"{node.max_speed:.2f} rad/s", width=14, anchor="e")
        self.speed_lbl.grid(row=r + 1, column=2)

        btns = ttk.Frame(f)
        btns.grid(row=r + 2, column=0, columnspan=4, pady=(12, 0))
        ttk.Button(btns, text="Sync to arm", command=self.sync).pack(side="left", padx=4)
        ttk.Button(btns, text="Go to zero", command=self.zero).pack(side="left", padx=4)

        # ---- Cartesian / kinematics panel ----
        box = ttk.LabelFrame(f, text="End effector (kinematics)", padding=10)
        box.grid(row=r + 3, column=0, columnspan=4, sticky="ew", pady=(14, 0))

        self.fk_lbl = ttk.Label(box, text="Current position:  --", font=("TkDefaultFont", 10, "bold"))
        self.fk_lbl.grid(row=0, column=0, columnspan=8, sticky="w", pady=(0, 8))

        self.entries = {}
        for c, axis in enumerate("XYZ"):
            ttk.Label(box, text=f"{axis} (m)").grid(row=1, column=c * 2, padx=(0, 4))
            e = ttk.Entry(box, width=9)
            e.insert(0, "0.000")
            e.grid(row=1, column=c * 2 + 1, padx=(0, 10))
            self.entries[axis] = e

        self.keep_orient = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Keep orientation", variable=self.keep_orient
                        ).grid(row=1, column=6, padx=6)
        self.go_btn = ttk.Button(box, text="Go to point", command=self.go_to_point)
        self.go_btn.grid(row=1, column=7, padx=4)
        self.use_btn = ttk.Button(box, text="Use current", command=self.use_current)
        self.use_btn.grid(row=2, column=7, padx=4, pady=(6, 0))

        self.ik_msg = ttk.Label(box, text="", wraplength=560, justify="left")
        self.ik_msg.grid(row=2, column=0, columnspan=7, sticky="w", pady=(6, 0))

        if kin is None:
            self.go_btn.state(["disabled"])
            self.use_btn.state(["disabled"])
            self.ik_msg.config(
                text="Kinematics unavailable: pass -p model_path:=<scene.xml> "
                     "and make sure `pip install mujoco numpy` is done.",
                foreground="#c62828")
        else:
            self.ik_msg.config(text=f"End effector = {kin.ee_label}")

        self.poll()

    # ----- slider callbacks -----
    def on_move(self, i, value):
        v = float(value)
        self.target_lbls[i].config(text=f"target {math.degrees(v):6.1f}°")
        self.node.set_target(i, v)

    def on_speed(self, value):
        v = float(value)
        self.speed_lbl.config(text=f"{v:.2f} rad/s")
        self.node.set_max_speed(v)

    def zero(self):
        for i, (_, lo, hi) in enumerate(JOINTS):
            z = clip(0.0, lo, hi)
            self.vars[i].set(z)
            self.on_move(i, z)

    def sync(self):
        cur = self.node.resync()
        if cur is None:
            return
        for i, (_, lo, hi) in enumerate(JOINTS):
            self.vars[i].set(clip(cur[i], lo, hi))
            self.target_lbls[i].config(text=f"target {math.degrees(cur[i]):6.1f}°")

    # ----- kinematics -----
    def current_q(self):
        with self.node.lock:
            return list(self.node.current) if self.node.current else None

    def use_current(self):
        q = self.current_q()
        if q is None or self.kin is None:
            return
        pos, _ = self.kin.fk(q)
        for axis, val in zip("XYZ", pos):
            self.entries[axis].delete(0, "end")
            self.entries[axis].insert(0, f"{val:.3f}")

    def go_to_point(self):
        q_now = self.current_q()
        if q_now is None:
            self.ik_msg.config(text="No joint state yet.", foreground="#c62828")
            return
        try:
            target = [float(self.entries[a].get()) for a in "XYZ"]
        except ValueError:
            self.ik_msg.config(text="X, Y, Z must be numbers (metres).", foreground="#c62828")
            return

        _, cur_mat = self.kin.fk(q_now)
        note = ""
        if self.keep_orient.get():
            q, pe, re, ok = self.kin.ik(target, q_now, cur_mat)
            if not ok:
                q, pe, re, ok = self.kin.ik(target, q_now, None)
                note = " (orientation could not be held, so it will change)"
        else:
            q, pe, re, ok = self.kin.ik(target, q_now, None)

        if not ok:
            self.ik_msg.config(
                text=f"Unreachable: closest solution is {pe * 1000:.1f} mm away. Not moving.",
                foreground="#c62828")
            return

        for i, (_, lo, hi) in enumerate(JOINTS):
            v = clip(float(q[i]), lo, hi)
            self.vars[i].set(v)
            self.on_move(i, v)
        angles = ", ".join(f"{math.degrees(a):.0f}°" for a in q)
        self.ik_msg.config(
            text=f"IK solved (error {pe * 1000:.2f} mm){note}. Joints: {angles}",
            foreground="#2e7d32")

    # ----- periodic refresh -----
    def poll(self):
        self.ticks += 1
        cur = self.current_q()
        if cur:
            if not self.synced:
                self.sync()
                for sc in self.scales:
                    sc.state(["!disabled"])
                self.synced = True
                self.status.config(
                    text=f"Connected to {self.node.state_topic} ({self.node.mode} mode)",
                    foreground="#2e7d32")
                self.use_current()
            for i in range(N):
                self.actual_lbls[i].config(text=f"actual {math.degrees(cur[i]):6.1f}°")
            if self.kin is not None:
                pos, _ = self.kin.fk(cur)
                self.fk_lbl.config(
                    text=f"Current position:  X {pos[0]:.3f}   Y {pos[1]:.3f}   Z {pos[2]:.3f}  m")
        elif self.ticks > 60:
            self.status.config(
                text=f"No data on {self.node.state_topic} - check `ros2 topic list` "
                     f"and pass -p state_topic:=<name>",
                foreground="#c62828")
        self.root.after(50, self.poll)


def main():
    rclpy.init()
    node = JointController()

    global JOINTS
    kin = None
    model_path = str(node.get_parameter("model_path").value)
    if model_path:
        try:
            kin = Kinematics(model_path, N, str(node.get_parameter("ee_name").value))
            JOINTS = kin.limits
            node.get_logger().info("Joint limits from model: " + ", ".join(
                f"{n} [{math.degrees(lo):.0f}, {math.degrees(hi):.0f}]" for n, lo, hi in JOINTS))
            node.get_logger().info(f"End effector: {kin.ee_label}")
        except Exception as e:
            node.get_logger().error(
                f"Could not load model for kinematics ({e}). "
                f"Using fallback limits; IK disabled.")
    else:
        node.get_logger().warn("No model_path given: IK disabled, fallback joint limits used.")

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    root = tk.Tk()
    SliderGUI(root, node, kin)
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()