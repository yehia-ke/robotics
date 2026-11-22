#!/usr/bin/env python3
"""
Motor-level controller node (Milestone 4): closes the loop around the MuJoCo simulation.

    /joint_state (sim)  --->  [ reference: trajectory | jog ]  --->  PID + G(q) + M(q)q''  --->  /joint_commands (torques)

* Runs once every `control_period` of SIMULATION time (default 4 ms = 250 Hz), triggered by
  the joint-state messages (their stamps are simulation time). mujoco_node runs in lockstep
  (torque_steps_per_command = 4), so each torque is applied for exactly 4 simulation steps.
* Reference = a planned trajectory (from the GUI) or, when none is running, a smooth set-point
  that follows the GUI joint sliders (jog).
* Every trajectory execution is logged to CSV (reference, measured, torques, TCP from the
  simulator) for the validation plots (validation/plot_validation.py).

Topics
  sub  /joint_state               sensor_msgs/JointState      measured q, q' (sim-time stamps)
  sub  /ee_pose                   geometry_msgs/PoseStamped   TCP pose sensed in the simulator
  sub  /controller/trajectory     trajectory_msgs/JointTrajectory  planned (t, q, q', q'') samples
                                  header.frame_id = JSON {"label": ..., "kind": "joint"|"task"}
  sub  /controller/jog_target     std_msgs/Float64MultiArray  6 target joint angles (sliders)
  sub  /controller/gripper        std_msgs/Float64MultiArray  [finger force N (+close), payload kg]
  sub  /controller/command        std_msgs/String (JSON)      gains / options / stop / sync / jog limits
  pub  /joint_commands            std_msgs/Float64MultiArray  6 joint torques (N*m) + gripper force (N)
  pub  /controller/state          std_msgs/Float64MultiArray  see STATE_LAYOUT (50 Hz)
"""
import csv
import json
import os
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory

import controller as ctl
import kinematics as kin
from trajectory import SampledTrajectory

N = kin.N_JOINTS

# Index of each field in the /controller/state message
STATE_LAYOUT = {
    "time": 0,               # simulation time (s)
    "mode": 1,               # 0 = hold / jog, 1 = trajectory running
    "trajectory_id": 2,      # id of the running (or last) trajectory
    "progress": 3,           # 0..1 while running
    "finished_id": 4,        # id of the last trajectory that finished (incl. settling)
    "q_ref": slice(5, 11),
    "qd_ref": slice(11, 17),
    "q": slice(17, 23),
    "qd": slice(23, 29),
    "torque": slice(29, 35),
    "tcp_sim": slice(35, 38),      # TCP position sensed in the simulator (/ee_pose)
    "tcp_ref": slice(38, 41),      # TCP position of the reference, FK(q_ref)
    "gripper_force": 41,
    "max_error": 42,               # largest |q_ref - q| of the running / last trajectory (rad)
    "saturated": 43,               # number of saturated motors this tick
}
STATE_SIZE = 44

SETTLE_TIME = 0.25            # s of extra logging/holding after a trajectory ends
STATE_PUBLISH_PERIOD = 0.02   # s (simulation time) between /controller/state messages


def _stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class ControllerNode(Node):

    def __init__(self):
        super().__init__("lite6_controller")
        self.declare_parameter("state_topic", "/joint_state")
        self.declare_parameter("command_topic", "/joint_commands")
        self.declare_parameter("ee_pose_topic", "/ee_pose")
        self.declare_parameter("control_period", 0.004)
        default_log_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "validation", "runs"))
        self.declare_parameter("log_dir", default_log_dir)
        self.declare_parameter("logging", True)

        self.dt = float(self.get_parameter("control_period").value)
        self.log_dir = str(self.get_parameter("log_dir").value)
        self.logging = bool(self.get_parameter("logging").value)

        self.controller = ctl.JointTorqueController()
        self.jog = None                      # ReferenceGenerator, created on the first state
        self.gripper_force = 0.0

        self.trajectory = None               # SampledTrajectory running now
        self.pending_trajectory = None       # received, starts at the next control tick
        self.trajectory_start = 0.0
        self.trajectory_id = 0
        self.finished_id = 0
        self.max_error = 0.0
        self.log_rows = []
        self.run_index = self._next_run_index()

        self.last_control_time = None
        self.last_state_publish = -1.0
        self.tcp_sim = np.full(3, np.nan)
        self.tcp_history = {}                # stamp (ns) -> TCP position sensed in MuJoCo
        self.rows_waiting_for_tcp = {}       # stamp (ns) -> log row index
        self.q_ref = None
        self.qd_ref = np.zeros(N)
        self.qdd_ref = np.zeros(N)

        self.command_publisher = self.create_publisher(
            Float64MultiArray, str(self.get_parameter("command_topic").value), 10)
        self.state_publisher = self.create_publisher(Float64MultiArray, "/controller/state", 10)
        self.create_subscription(JointState, str(self.get_parameter("state_topic").value),
                                 self._on_joint_state, 10)
        self.create_subscription(PoseStamped, str(self.get_parameter("ee_pose_topic").value),
                                 self._on_ee_pose, 10)
        self.create_subscription(JointTrajectory, "/controller/trajectory", self._on_trajectory, 10)
        self.create_subscription(Float64MultiArray, "/controller/jog_target", self._on_jog_target, 10)
        self.create_subscription(Float64MultiArray, "/controller/gripper", self._on_gripper, 10)
        self.create_subscription(String, "/controller/command", self._on_command, 10)

        self.get_logger().info(
            f"PID + gravity compensation controller at {1.0 / self.dt:.0f} Hz (sim time). "
            f"Kp={self.controller.kp.tolist()} Ki={self.controller.ki.tolist()} "
            f"Kd={self.controller.kd.tolist()}. Logs: {self.log_dir}")

    # ------------------------------------------------------------------ inputs
    def _on_ee_pose(self, message):
        p = message.pose.position
        self.tcp_sim = np.array([p.x, p.y, p.z])
        # The pose of a simulation step can arrive just after its joint state: keep a short
        # history by stamp and fill in log rows that were written before it arrived, so the
        # logged TCP is sensed at exactly the same simulation instant as the joint angles.
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        self.tcp_history[stamp] = self.tcp_sim
        if len(self.tcp_history) > 64:
            self.tcp_history.pop(next(iter(self.tcp_history)))
        row = self.rows_waiting_for_tcp.pop(stamp, None)
        if row is not None and row < len(self.log_rows):
            self.log_rows[row][self.TCP_SIM_COLUMNS] = self.tcp_sim

    def _on_trajectory(self, message):
        try:
            meta = json.loads(message.header.frame_id) if message.header.frame_id else {}
        except json.JSONDecodeError:
            meta = {"label": message.header.frame_id}
        points = message.points
        if len(points) < 2:
            self.get_logger().warn("Ignoring trajectory with fewer than 2 points.")
            return
        t = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in points]
        q = [list(p.positions) for p in points]
        qd = [list(p.velocities) if len(p.velocities) == N else [0.0] * N for p in points]
        qdd = [list(p.accelerations) if len(p.accelerations) == N else [0.0] * N for p in points]
        trajectory = SampledTrajectory(meta.get("label", "trajectory"), t, q, qd, qdd,
                                       tcp_position=np.zeros((len(t), 3)),
                                       kind=meta.get("kind", "joint"))
        trajectory.meta = meta
        if self.q_ref is not None:
            jump = float(np.abs(trajectory.q[0] - self.q_ref).max())
            if jump > 0.05:
                self.get_logger().warn(
                    f"Trajectory '{trajectory.label}' starts {jump:.3f} rad away from the current "
                    f"reference: rejected (plan it from the current reference).")
                return
        self.pending_trajectory = trajectory

    def _on_jog_target(self, message):
        if len(message.data) != N or self.jog is None:
            return
        if self.trajectory is not None:
            self._abort_trajectory("jog")
        self.jog.target = np.clip(np.array(message.data, float), kin.JOINT_LOWER, kin.JOINT_UPPER)

    def _on_gripper(self, message):
        if len(message.data) >= 1:
            self.gripper_force = float(message.data[0])
        if len(message.data) >= 2:
            self.controller.payload_mass = max(0.0, float(message.data[1]))

    def _on_command(self, message):
        try:
            command = json.loads(message.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"Bad command: {message.data}")
            return
        kind = command.get("cmd")
        if kind == "gains":
            self.controller.set_gains(command.get("kp"), command.get("ki"), command.get("kd"))
            self.get_logger().info(f"Gains: Kp={self.controller.kp.tolist()} "
                                   f"Ki={self.controller.ki.tolist()} Kd={self.controller.kd.tolist()}")
        elif kind == "options":
            if "gravity" in command:
                self.controller.use_gravity = bool(command["gravity"])
            if "feedforward" in command:
                self.controller.use_feedforward = bool(command["feedforward"])
            if "logging" in command:
                self.logging = bool(command["logging"])
            self.get_logger().info(f"Options: gravity={self.controller.use_gravity} "
                                   f"feedforward={self.controller.use_feedforward} logging={self.logging}")
        elif kind == "stop":
            self._abort_trajectory("stop")
        elif kind == "sync" and self.jog is not None and self.last_q is not None:
            self._abort_trajectory("sync")
            self.jog.reset(self.last_q)
            self.controller.reset()
        elif kind == "jog_limits" and self.jog is not None:
            self.jog.max_speed = float(command.get("max_speed", self.jog.max_speed))
            self.jog.max_acceleration = float(command.get("max_acceleration", self.jog.max_acceleration))

    def _abort_trajectory(self, reason):
        """Stop a running trajectory smoothly: brake from the current reference velocity."""
        self.pending_trajectory = None
        if self.trajectory is None or self.q_ref is None:
            return
        self.get_logger().info(f"Trajectory '{self.trajectory.label}' aborted ({reason}).")
        self._finish_trajectory(aborted=True)
        acceleration = self.jog.max_acceleration
        stopping = self.qd_ref * np.abs(self.qd_ref) / (2.0 * acceleration)
        self.jog.position = self.q_ref.copy()
        self.jog.velocity = self.qd_ref.copy()
        self.jog.target = np.clip(self.q_ref + stopping, kin.JOINT_LOWER, kin.JOINT_UPPER)

    # ------------------------------------------------------------------ control loop
    last_q = None

    def _on_joint_state(self, message):
        if len(message.position) < N:
            return
        t = _stamp_seconds(message.header.stamp)
        self.stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if self.last_control_time is not None and t < self.last_control_time + self.dt - 5e-4:
            return                                   # not a control instant yet
        if self.last_control_time is not None and t < self.last_control_time:
            self.last_control_time = None            # simulation was reset
        self.last_control_time = t

        q = np.array(message.position[:N])
        qd = np.array(message.velocity[:N]) if len(message.velocity) >= N else np.zeros(N)
        self.last_q = q
        if self.jog is None:                         # first state: hold where the arm is
            self.jog = ctl.ReferenceGenerator(q)
            self.q_ref = q.copy()

        self._update_reference(t)
        torque = self.controller.compute(q, qd, self.q_ref, self.qd_ref, self.qdd_ref, self.dt)

        command = Float64MultiArray()
        command.data = [float(x) for x in torque] + [self.gripper_force]
        self.command_publisher.publish(command)

        if self.trajectory is not None:
            error = float(np.abs(self.q_ref - q).max())
            self.max_error = max(self.max_error, error)
            if self.logging:
                self._log_row(t - self.trajectory_start, q, qd)
            if t - self.trajectory_start >= self.trajectory.duration + SETTLE_TIME:
                self._finish_trajectory(aborted=False)
        if t - self.last_state_publish >= STATE_PUBLISH_PERIOD - 1e-6:
            self.last_state_publish = t
            self._publish_state(t, q, qd, torque)

    def _update_reference(self, t):
        if self.pending_trajectory is not None:
            self.trajectory = self.pending_trajectory
            self.pending_trajectory = None
            self.trajectory_start = t
            self.trajectory_id += 1
            self.max_error = 0.0
            self.log_rows = []
            self.rows_waiting_for_tcp = {}
            self.get_logger().info(f"[{self.trajectory_id}] running '{self.trajectory.label}' "
                                   f"({self.trajectory.duration:.2f} s)")
        if self.trajectory is not None:
            q_ref, qd_ref, qdd_ref = self.trajectory.sample(t - self.trajectory_start)
            self.q_ref, self.qd_ref, self.qdd_ref = (np.array(x, float) for x in (q_ref, qd_ref, qdd_ref))
        else:
            self.q_ref, self.qd_ref, self.qdd_ref = self.jog.step(self.dt)

    def _finish_trajectory(self, aborted):
        trajectory = self.trajectory
        self.trajectory = None
        self.finished_id = self.trajectory_id
        # continue holding exactly at the trajectory's last reference
        self.jog.reset(self.q_ref if aborted else trajectory.final_position)
        if not aborted:
            self.q_ref = trajectory.final_position
            self.qd_ref = np.zeros(N)
            self.qdd_ref = np.zeros(N)
        self.get_logger().info(
            f"[{self.trajectory_id}] {'aborted' if aborted else 'finished'} '{trajectory.label}': "
            f"max joint error {self.max_error * 1000:.2f} mrad")
        if self.logging and self.log_rows:
            self._write_log(trajectory, aborted)
        self.rows_waiting_for_tcp = {}

    # ------------------------------------------------------------------ outputs
    def _publish_state(self, t, q, qd, torque):
        state = np.zeros(STATE_SIZE)
        state[STATE_LAYOUT["time"]] = t
        state[STATE_LAYOUT["mode"]] = 1.0 if self.trajectory is not None else 0.0
        state[STATE_LAYOUT["trajectory_id"]] = self.trajectory_id
        state[STATE_LAYOUT["progress"]] = (
            min(1.0, (t - self.trajectory_start) / max(self.trajectory.duration, 1e-6))
            if self.trajectory is not None else -1.0)
        state[STATE_LAYOUT["finished_id"]] = self.finished_id
        state[STATE_LAYOUT["q_ref"]] = self.q_ref
        state[STATE_LAYOUT["qd_ref"]] = self.qd_ref
        state[STATE_LAYOUT["q"]] = q
        state[STATE_LAYOUT["qd"]] = qd
        state[STATE_LAYOUT["torque"]] = torque
        state[STATE_LAYOUT["tcp_sim"]] = self.tcp_sim
        state[STATE_LAYOUT["tcp_ref"]] = kin.fk(self.q_ref)[0]
        state[STATE_LAYOUT["gripper_force"]] = self.gripper_force
        state[STATE_LAYOUT["max_error"]] = self.max_error
        terms = self.controller.last_terms
        state[STATE_LAYOUT["saturated"]] = float(np.sum(terms["saturated"])) if terms else 0.0
        message = Float64MultiArray()
        message.data = [float(x) for x in state]
        self.state_publisher.publish(message)

    LOG_HEADER = (["t"] + [f"q_ref{i}" for i in range(1, 7)] + [f"qd_ref{i}" for i in range(1, 7)]
                  + [f"qdd_ref{i}" for i in range(1, 7)] + [f"q{i}" for i in range(1, 7)]
                  + [f"qd{i}" for i in range(1, 7)] + [f"tau{i}" for i in range(1, 7)]
                  + [f"tau_gravity{i}" for i in range(1, 7)] + [f"tau_feedforward{i}" for i in range(1, 7)]
                  + ["tcp_ref_x", "tcp_ref_y", "tcp_ref_z", "tcp_sim_x", "tcp_sim_y", "tcp_sim_z",
                     "tcp_fk_x", "tcp_fk_y", "tcp_fk_z", "gripper_force"])

    TCP_SIM_COLUMNS = slice(LOG_HEADER.index("tcp_sim_x"), LOG_HEADER.index("tcp_sim_z") + 1)

    def _log_row(self, t, q, qd):
        terms = self.controller.last_terms
        tcp_sim = self.tcp_history.get(self.stamp_ns)
        if tcp_sim is None:                  # not received yet: filled in by _on_ee_pose
            self.rows_waiting_for_tcp[self.stamp_ns] = len(self.log_rows)
            tcp_sim = np.full(3, np.nan)
        self.log_rows.append(np.concatenate([
            [t], self.q_ref, self.qd_ref, self.qdd_ref, q, qd, terms["torque"], terms["gravity"],
            terms["feedforward"], kin.fk(self.q_ref)[0], tcp_sim, kin.fk(q)[0], [self.gripper_force]]))

    def _next_run_index(self):
        try:
            names = os.listdir(self.log_dir)
        except FileNotFoundError:
            return 1
        indices = [int(n[:4]) for n in names if n[:4].isdigit()]
        return max(indices, default=0) + 1

    def _write_log(self, trajectory, aborted):
        os.makedirs(self.log_dir, exist_ok=True)
        safe = "".join(c if c.isalnum() else "_" for c in trajectory.label)[:60].strip("_")
        base = os.path.join(self.log_dir, f"{self.run_index:04d}_{safe}")
        self.run_index += 1
        with open(base + ".csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.LOG_HEADER)
            writer.writerows(np.round(np.array(self.log_rows), 7).tolist())
        meta = dict(getattr(trajectory, "meta", {}))
        meta.update({
            "label": trajectory.label, "kind": trajectory.kind, "duration": trajectory.duration,
            "aborted": aborted, "max_joint_error_rad": self.max_error,
            "control_period": self.dt, "kp": self.controller.kp.tolist(),
            "ki": self.controller.ki.tolist(), "kd": self.controller.kd.tolist(),
            "gravity_compensation": self.controller.use_gravity,
            "feedforward": self.controller.use_feedforward,
            "payload_mass": self.controller.payload_mass,
            "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        with open(base + ".json", "w") as handle:
            json.dump(meta, handle, indent=2)
        self.log_rows = []


def main():
    rclpy.init()
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:                    # e.g. a publish racing with Ctrl-C shutdown
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
