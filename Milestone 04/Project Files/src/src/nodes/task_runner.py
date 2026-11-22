"""
ROS client side of the controller: send trajectories / gripper commands and wait for them.

Used by the GUI (sliders_node.py) and by the headless validation script (run_task.py).
The client never computes torques; it plans trajectories (trajectory.py / application.py)
and hands them to controller_node.py, then waits until the controller reports them finished.
"""
import json
import threading
import time

import numpy as np
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import application as app
import trajectory as traj
from controller_node import STATE_LAYOUT

N = 6


class TaskAborted(RuntimeError):
    pass


class TaskClient:
    """Wraps the controller topics for an existing rclpy node (which must be spinning)."""

    def __init__(self, node, collision_checker=None):
        self.node = node
        self.collision_checker = collision_checker      # collision.CollisionChecker or None
        self.lock = threading.Lock()
        self.state = None
        self.state_event = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.trajectory_publisher = node.create_publisher(JointTrajectory, "/controller/trajectory", 10)
        self.jog_publisher = node.create_publisher(Float64MultiArray, "/controller/jog_target", 10)
        self.gripper_publisher = node.create_publisher(Float64MultiArray, "/controller/gripper", 10)
        self.command_publisher = node.create_publisher(String, "/controller/command", 10)
        node.create_subscription(Float64MultiArray, "/controller/state", self._on_state, 10)

    # ------------------------------------------------------------------ state
    def _on_state(self, message):
        with self.lock:
            self.state = np.array(message.data)
            self.state_event.notify_all()

    def get_state(self):
        with self.lock:
            return None if self.state is None else self.state.copy()

    def field(self, name, state=None):
        state = self.get_state() if state is None else state
        return None if state is None else state[STATE_LAYOUT[name]]

    def wait_for_state(self, timeout=30.0):
        deadline = time.time() + timeout
        with self.lock:
            while self.state is None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("no /controller/state: is controller_node.py running?")
                self.state_event.wait(remaining)
        return self.get_state()

    def reference(self):
        """Current reference joint angles of the controller (plan new motions from here)."""
        q_ref = self.field("q_ref")
        if q_ref is None:
            raise TimeoutError("no /controller/state yet: is controller_node.py running?")
        return np.array(q_ref)

    def wait_sim_time(self, seconds):
        """Wait `seconds` of SIMULATION time (the simulation may run slower than real time)."""
        start = self.field("time")
        while self.field("time") - start < seconds:
            self._check_stop()
            time.sleep(0.01)

    def _check_stop(self):
        if self.stop_event.is_set():
            raise TaskAborted("stopped by the user")

    # ------------------------------------------------------------------ commands
    def send_command(self, **command):
        message = String()
        message.data = json.dumps(command)
        self.command_publisher.publish(message)

    def set_options(self, **options):
        self.send_command(cmd="options", **options)

    def set_gains(self, kp, ki, kd):
        self.send_command(cmd="gains", kp=list(map(float, kp)), ki=list(map(float, ki)),
                          kd=list(map(float, kd)))

    def stop(self):
        """Abort a running task/trajectory: the controller brakes smoothly and holds."""
        self.stop_event.set()
        self.send_command(cmd="stop")

    def jog(self, joint_angles):
        message = Float64MultiArray()
        message.data = [float(x) for x in joint_angles]
        self.jog_publisher.publish(message)

    def gripper(self, force, payload=0.0):
        message = Float64MultiArray()
        message.data = [float(force), float(payload)]
        self.gripper_publisher.publish(message)

    def send_trajectory(self, trajectory, meta=None):
        message = JointTrajectory()
        info = {"label": trajectory.label, "kind": trajectory.kind}
        info.update(meta or {})
        message.header.frame_id = json.dumps(info)
        message.joint_names = [f"joint{i}" for i in range(1, N + 1)]
        for k in range(len(trajectory.t)):
            point = JointTrajectoryPoint()
            point.positions = [float(x) for x in trajectory.q[k]]
            point.velocities = [float(x) for x in trajectory.qd[k]]
            point.accelerations = [float(x) for x in trajectory.qdd[k]]
            seconds = float(trajectory.t[k])
            point.time_from_start.sec = int(seconds)
            point.time_from_start.nanosec = int(round((seconds - int(seconds)) * 1e9)) % 1_000_000_000
            message.points.append(point)
        self.trajectory_publisher.publish(message)

    def run_trajectory(self, trajectory, meta=None, timeout_margin=15.0):
        """Send a trajectory and block until the controller has finished (and settled) it.

        Raises trajectory.TrajectoryError (nothing is sent) if the collision check fails.
        Returns the controller state at the end (includes the max tracking error)."""
        self._check_stop()
        if self.collision_checker is not None:
            problem = self.collision_checker.check_trajectory(trajectory)
            if problem:
                raise traj.TrajectoryError(f"'{trajectory.label}': {problem}")
        before = int(self.field("finished_id"))
        self.send_trajectory(trajectory, meta)
        deadline = time.time() + trajectory.duration * 3.0 + timeout_margin
        while int(self.field("finished_id")) <= before:
            self._check_stop()
            if time.time() > deadline:
                raise TimeoutError(f"'{trajectory.label}' did not finish (controller running?)")
            time.sleep(0.01)
        return self.get_state()

    # ------------------------------------------------------------------ tasks
    def execute(self, steps, report=None, meta=None):
        """Run application steps (application.Step) one after the other.

        report(index, step, message) is called before/after each step (for the GUI).
        Each motion is planned from the end of the previous one, so the joint references
        chain exactly."""
        self.stop_event.clear()
        report = report or (lambda index, step, message: None)
        q_plan = self.reference()
        results = []
        for index, step in enumerate(steps):
            self._check_stop()
            report(index, step, "running")
            if step.kind == "gripper":
                self.gripper(step.force, step.payload)
                self.wait_sim_time(step.duration or app.GRIP_SETTLE_TIME)
                report(index, step, "done")
                continue
            trajectory = app.plan_motion(step, q_plan)
            state = self.run_trajectory(trajectory, dict(meta or {}, step=step.description))
            q_plan = trajectory.final_position
            error = float(self.field("max_error", state))
            results.append((step.description, trajectory.label, error))
            report(index, step, f"done, max joint error {error * 1000:.2f} mrad")
        return results
