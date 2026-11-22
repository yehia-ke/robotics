#!/usr/bin/env python3
"""
Lite6 lab-cell GUI (Milestones 2-4), MuJoCo + ROS2.

Milestone 4 turns the Milestone 3 slider window into a tabbed control panel. The GUI no
longer sends joint commands itself: it PLANS (trajectory.py, application.py) and hands the
trajectories to the motor-level controller (controller_node.py), which closes the loop with
joint torques.

Tabs
  Joints        : jog sliders (smooth set-point), forward kinematics readout (DH vs simulator),
                  inverse kinematics "go to point", live end-effector velocity, singularity
  Trajectories  : joint-space cubic / quintic to any target; task-space straight line
                  (cubic / quintic time scaling); gripper open / close
  Controller    : PID gains, gravity compensation / feed-forward switches, live tracking
                  errors + torques, live plot, validation plots
  Application   : the industrial task: sort the sample vials by cap colour; flask lifting demo

    python3 sliders_node.py --ros-args -p model_path:=/path/to/lab-scene.xml

Requires: pip install mujoco numpy matplotlib
"""
import math
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import ttk

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node

import application as app
import controller as ctl
import kinematics as kin
import trajectory as traj
from collision import CollisionChecker
from task_runner import TaskAborted, TaskClient

NUMBER_OF_JOINTS = 6
JOINT_NAMES = [f"Joint {i}" for i in range(1, 7)]

COLOR_OK = "#2e7d32"
COLOR_WARNING = "#b36b00"
COLOR_ERROR = "#c62828"
COLOR_RUNNING = "#1565c0"

VALIDATION_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "..", "..", "..", "validation"))


def clamp_value(value, lowest, highest):
    return max(lowest, min(highest, value))


class GuiNode(Node):
    def __init__(self):
        super().__init__("lite6_gui")
        self.declare_parameter("model_path", "")


# ===========================================================================
# GUI window
# ===========================================================================
class ControllerWindow:
    """The Tkinter window. All ROS traffic goes through TaskClient (task_runner.py)."""

    def __init__(self, root, client, layout):
        self.root = root
        self.client = client
        self.layout = layout                     # None if the scene could not be loaded
        self.worker = None                       # background thread (planning / task)
        self.slider_targets = None               # joint targets shown on the sliders
        self.previous_mode = 0
        self.refresh_count = 0
        self.plot_history = deque(maxlen=600)    # (t, q_ref, q) for the live plot, 12 s at 50 Hz

        self.root.title("Lite6 Lab Cell - Milestone 4")
        self.root.minsize(760, 0)
        self._build_status_bar()
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._build_joints_tab()
        self._build_trajectory_tab()
        self._build_controller_tab()
        self._build_application_tab()
        self.presets = self._compute_presets()
        self.preset_box["values"] = list(self.presets)
        self._refresh()

    # ======================================================================
    # Status bar (always visible)
    # ======================================================================
    def _build_status_bar(self):
        bar = ttk.Frame(self.root, padding=(8, 8, 8, 4))
        bar.pack(fill="x")
        self.status_label = ttk.Label(bar, text="Waiting for controller_node.py ...",
                                      foreground=COLOR_WARNING, font=("TkDefaultFont", 10, "bold"))
        self.status_label.pack(side="left")
        tk.Button(bar, text="STOP", bg="#c62828", fg="white", activebackground="#8e0000",
                  font=("TkDefaultFont", 10, "bold"), command=self._stop).pack(side="right")
        self.mode_label = ttk.Label(bar, text="", foreground=COLOR_RUNNING)
        self.mode_label.pack(side="right", padx=12)

    def _stop(self):
        self.client.stop()
        self._set_message(self.trajectory_message, "Stopped: the arm brakes and holds.", COLOR_WARNING)

    # ======================================================================
    # Tab 1: joints (jog), kinematics
    # ======================================================================
    def _build_joints_tab(self):
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="Joints")
        tab.columnconfigure(1, weight=1)
        self.slider_variables, self.sliders, self.target_labels, self.actual_labels = [], [], [], []
        for joint in range(NUMBER_OF_JOINTS):
            lowest, highest = kin.JOINT_LOWER[joint], kin.JOINT_UPPER[joint]
            ttk.Label(tab, text=JOINT_NAMES[joint], width=8).grid(row=joint, column=0, sticky="w", pady=4)
            variable = tk.DoubleVar(value=0.0)
            slider = ttk.Scale(tab, from_=lowest, to=highest, variable=variable, orient="horizontal",
                               command=lambda value, index=joint: self._on_slider_moved(index, value))
            slider.grid(row=joint, column=1, sticky="ew", padx=8)
            slider.state(["disabled"])
            target = ttk.Label(tab, text="target    0.0°", width=14, anchor="e")
            target.grid(row=joint, column=2)
            actual = ttk.Label(tab, text="actual    0.0°", width=14, anchor="e")
            actual.grid(row=joint, column=3)
            self.slider_variables.append(variable)
            self.sliders.append(slider)
            self.target_labels.append(target)
            self.actual_labels.append(actual)

        row = NUMBER_OF_JOINTS
        ttk.Separator(tab).grid(row=row, column=0, columnspan=4, sticky="ew", pady=8)
        self.jog_speed = tk.DoubleVar(value=1.0)
        self.jog_acceleration = tk.DoubleVar(value=4.0)
        for offset, (name, variable, low, high, unit) in enumerate(
                (("Jog speed", self.jog_speed, 0.1, 3.0, "rad/s"),
                 ("Jog accel.", self.jog_acceleration, 0.5, 20.0, "rad/s²"))):
            ttk.Label(tab, text=name).grid(row=row + 1 + offset, column=0, sticky="w")
            label = ttk.Label(tab, text=f"{variable.get():.2f} {unit}", width=14, anchor="e")
            ttk.Scale(tab, from_=low, to=high, variable=variable, orient="horizontal",
                      command=lambda value, l=label, u=unit: self._on_jog_limits(l, value, u)
                      ).grid(row=row + 1 + offset, column=1, sticky="ew", padx=8)
            label.grid(row=row + 1 + offset, column=2)

        buttons = ttk.Frame(tab)
        buttons.grid(row=row + 3, column=0, columnspan=4, pady=(10, 0))
        ttk.Button(buttons, text="Sync to arm", command=self._sync_to_arm).pack(side="left", padx=4)
        ttk.Button(buttons, text="Home (quintic)",
                   command=lambda: self._run_joint_trajectory(app.HOME, "quintic", None)).pack(side="left", padx=4)

        # ---- kinematics box ----
        box = ttk.LabelFrame(tab, text="End effector (TCP = fingertip centre)", padding=10)
        box.grid(row=row + 4, column=0, columnspan=4, sticky="ew", pady=(14, 0))
        mono = ("TkFixedFont", 9)
        self.fk_label = ttk.Label(box, text="DH forward kinematics   --", font=mono)
        self.fk_label.grid(row=0, column=0, columnspan=8, sticky="w")
        self.sim_label = ttk.Label(box, text="Sensed in MuJoCo        --", font=mono)
        self.sim_label.grid(row=1, column=0, columnspan=8, sticky="w")
        self.velocity_label = ttk.Label(box, text="Velocity J(q) q'        --", font=mono)
        self.velocity_label.grid(row=2, column=0, columnspan=8, sticky="w")
        self.singularity_label = ttk.Label(box, text="Singularity             --", font=mono)
        self.singularity_label.grid(row=3, column=0, columnspan=8, sticky="w", pady=(0, 8))

        self.point_entries = self._xyz_entries(box, row=4)
        ttk.Button(box, text="Use current", command=lambda: self._fill_xyz(self.point_entries)
                   ).grid(row=4, column=6, padx=4)
        ttk.Button(box, text="Go: joint space", command=lambda: self._go_to_point("joint")
                   ).grid(row=5, column=0, columnspan=3, pady=(6, 0), sticky="w")
        ttk.Button(box, text="Go: straight line", command=lambda: self._go_to_point("line")
                   ).grid(row=5, column=3, columnspan=3, pady=(6, 0), sticky="w")
        self.joints_message = ttk.Label(box, text="", wraplength=640, justify="left")
        self.joints_message.grid(row=6, column=0, columnspan=8, sticky="w", pady=(6, 0))

    def _xyz_entries(self, parent, row, column=0, initial=(0.0, 0.0, 0.0), width=8):
        entries = []
        for k, axis in enumerate("XYZ"):
            ttk.Label(parent, text=f"{axis} (m)").grid(row=row, column=column + 2 * k, padx=(0, 4))
            entry = ttk.Entry(parent, width=width)
            entry.insert(0, f"{initial[k]:.3f}")
            entry.grid(row=row, column=column + 2 * k + 1, padx=(0, 10))
            entries.append(entry)
        return entries

    def _fill_xyz(self, entries):
        state = self.client.get_state()
        if state is None:
            return
        position, _ = kin.fk(self.client.field("q_ref", state))
        for entry, value in zip(entries, position):
            entry.delete(0, "end")
            entry.insert(0, f"{value:.3f}")

    def _read_numbers(self, entries, message_label, what):
        try:
            return np.array([float(e.get()) for e in entries])
        except ValueError:
            self._set_message(message_label, f"{what} must be numbers.", COLOR_ERROR)
            return None

    def _on_slider_moved(self, joint, value):
        if self.slider_targets is None:
            return
        self.slider_targets[joint] = float(value)
        self.target_labels[joint].config(text=f"target {math.degrees(float(value)):6.1f}°")
        self.client.jog(self.slider_targets)

    def _on_jog_limits(self, label, value, unit):
        label.config(text=f"{float(value):.2f} {unit}")
        self.client.send_command(cmd="jog_limits", max_speed=self.jog_speed.get(),
                                 max_acceleration=self.jog_acceleration.get())

    def _show_on_sliders(self, angles):
        self.slider_targets = np.array(angles, float)
        for joint in range(NUMBER_OF_JOINTS):
            self.slider_variables[joint].set(clamp_value(angles[joint], kin.JOINT_LOWER[joint],
                                                         kin.JOINT_UPPER[joint]))
            self.target_labels[joint].config(text=f"target {math.degrees(angles[joint]):6.1f}°")

    def _sync_to_arm(self):
        self.client.send_command(cmd="sync")
        q = self.client.field("q")
        if q is not None:
            self._show_on_sliders(q)

    def _go_to_point(self, kind):
        position = self._read_numbers(self.point_entries, self.joints_message, "X, Y, Z")
        if position is None:
            return

        def job():
            q_start = self.client.reference()
            _, R_now = kin.fk(q_start)
            if kind == "line":
                self._plan_and_run_line(q_start, position, R_now, None, "quintic", self.joints_message)
                return
            q_goal, ok = kin.ik_pose(position, R_now, [q_start, app.HOME])
            if not ok:                         # orientation cannot be kept: position only
                q_goal, ok = kin.ik_position(position, q_start)
            if not ok:
                raise ValueError("unreachable point")
            self._run_checked(traj.joint_space(q_start, q_goal, None, "quintic"), self.joints_message)
        self._start_worker(job, self.joints_message)

    # ======================================================================
    # Tab 2: trajectories
    # ======================================================================
    def _build_trajectory_tab(self):
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="Trajectories")
        tab.columnconfigure(0, weight=1)

        # ---- joint space ----
        box = ttk.LabelFrame(tab, text="Joint-space trajectory (every joint: rest-to-rest polynomial)",
                             padding=10)
        box.grid(row=0, column=0, sticky="ew")
        self.joint_profile = tk.StringVar(value="quintic")
        ttk.Radiobutton(box, text="Cubic", value="cubic", variable=self.joint_profile).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(box, text="Quintic", value="quintic", variable=self.joint_profile).grid(row=0, column=1, sticky="w")
        ttk.Label(box, text="Duration (s, empty = auto)").grid(row=0, column=2, padx=(20, 4))
        self.joint_duration = ttk.Entry(box, width=6)
        self.joint_duration.grid(row=0, column=3)
        ttk.Label(box, text="Preset").grid(row=0, column=4, padx=(20, 4))
        self.preset_box = ttk.Combobox(box, width=22, state="readonly")
        self.preset_box.grid(row=0, column=5)
        self.preset_box.bind("<<ComboboxSelected>>", self._on_preset)

        self.joint_entries = []
        for joint in range(NUMBER_OF_JOINTS):
            ttk.Label(box, text=f"q{joint + 1} (°)").grid(row=1, column=joint, pady=(8, 0))
            entry = ttk.Entry(box, width=8)
            entry.insert(0, "0.0")
            entry.grid(row=2, column=joint, padx=2)
            self.joint_entries.append(entry)
        buttons = ttk.Frame(box)
        buttons.grid(row=3, column=0, columnspan=6, sticky="w", pady=(8, 0))
        ttk.Button(buttons, text="Use current", command=self._fill_joint_entries).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Plan & run joint trajectory", command=self._on_run_joint).pack(side="left")

        # ---- task space ----
        box = ttk.LabelFrame(tab, text="Task-space trajectory (TCP on a straight line)", padding=10)
        box.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self.line_profile = tk.StringVar(value="quintic")
        ttk.Radiobutton(box, text="Cubic s(t)", value="cubic", variable=self.line_profile).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(box, text="Quintic s(t)", value="quintic", variable=self.line_profile).grid(row=0, column=1, sticky="w")
        ttk.Label(box, text="Duration (s, empty = auto)").grid(row=0, column=2, columnspan=2, padx=(20, 4))
        self.line_duration = ttk.Entry(box, width=6)
        self.line_duration.grid(row=0, column=4)
        self.line_entries = self._xyz_entries(box, row=1)
        ttk.Button(box, text="Use current", command=lambda: self._fill_xyz(self.line_entries)
                   ).grid(row=1, column=6, padx=4)
        self.line_orientation = tk.StringVar(value="keep")
        ttk.Radiobutton(box, text="Keep orientation", value="keep", variable=self.line_orientation
                        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Radiobutton(box, text="Gripper down, yaw (°):", value="down", variable=self.line_orientation
                        ).grid(row=2, column=2, columnspan=2, sticky="w", pady=(6, 0))
        self.line_yaw = ttk.Entry(box, width=6)
        self.line_yaw.insert(0, "0")
        self.line_yaw.grid(row=2, column=4, pady=(6, 0))
        ttk.Label(box, text="Relative move (mm):").grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        nudges = ttk.Frame(box)
        nudges.grid(row=3, column=2, columnspan=5, sticky="w", pady=(6, 0))
        for axis, sign in (("X", 1), ("X", -1), ("Y", 1), ("Y", -1), ("Z", 1), ("Z", -1)):
            ttk.Button(nudges, text=f"{'+' if sign > 0 else '-'}{axis} 50", width=6,
                       command=lambda a="XYZ".index(axis), s=sign: self._nudge(a, s * 0.05)).pack(side="left", padx=1)
        ttk.Button(box, text="Plan & run straight line", command=self._on_run_line
                   ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))

        # ---- gripper ----
        box = ttk.LabelFrame(tab, text="Gripper", padding=10)
        box.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(box, text="Open", command=lambda: self.client.gripper(app.GRIPPER_OPEN_FORCE)
                   ).pack(side="left", padx=4)
        ttk.Button(box, text="Close", command=lambda: self.client.gripper(app.GRIPPER_CLOSE_FORCE)
                   ).pack(side="left", padx=4)
        ttk.Label(box, text="finger motor force: +5 N closes, -5 N opens").pack(side="left", padx=10)

        self.trajectory_message = ttk.Label(tab, text="", wraplength=700, justify="left")
        self.trajectory_message.grid(row=3, column=0, sticky="w", pady=(10, 0))

    def _compute_presets(self):
        presets = {"Home": app.HOME.copy()}
        if self.layout is None:
            return presets
        try:
            for name, colour, position in self.layout.vials:
                xy = position[:2]
                presets[f"Above {name.replace('lab_sample_', '')} ({colour})"] = app.solve_pose(
                    np.array([*xy, app.TRANSIT_HEIGHT]), app.radial_yaw(xy), app.HOME)
            for colour, position in self.layout.bins.items():
                xy = position[:2]
                presets[f"Above {colour} bin"] = app.solve_pose(
                    np.array([*xy, app.TRANSIT_HEIGHT]), app.radial_yaw(xy), app.HOME)
        except ValueError as error:
            print(f"preset skipped: {error}", file=sys.stderr)
        return presets

    def _on_preset(self, _event=None):
        angles = self.presets.get(self.preset_box.get())
        if angles is None:
            return
        for entry, angle in zip(self.joint_entries, angles):
            entry.delete(0, "end")
            entry.insert(0, f"{math.degrees(angle):.1f}")

    def _fill_joint_entries(self):
        q = self.client.field("q_ref")
        if q is None:
            return
        for entry, angle in zip(self.joint_entries, q):
            entry.delete(0, "end")
            entry.insert(0, f"{math.degrees(angle):.1f}")

    def _read_duration(self, entry, label):
        text = entry.get().strip()
        if not text:
            return None
        try:
            duration = float(text)
        except ValueError:
            self._set_message(label, "Duration must be a number (or empty for automatic).", COLOR_ERROR)
            return False
        if duration <= 0.05:
            self._set_message(label, "Duration must be more than 0.05 s.", COLOR_ERROR)
            return False
        return duration

    def _on_run_joint(self):
        degrees = self._read_numbers(self.joint_entries, self.trajectory_message, "Joint angles")
        duration = self._read_duration(self.joint_duration, self.trajectory_message)
        if degrees is None or duration is False:
            return
        self._run_joint_trajectory(np.radians(degrees), self.joint_profile.get(), duration)

    def _run_joint_trajectory(self, q_goal, profile, duration, label=None):
        label = label or self.trajectory_message

        def job():
            trajectory = traj.joint_space(self.client.reference(), q_goal, duration, profile)
            self._run_checked(trajectory, label)
        self._start_worker(job, label)

    def _on_run_line(self):
        position = self._read_numbers(self.line_entries, self.trajectory_message, "X, Y, Z")
        duration = self._read_duration(self.line_duration, self.trajectory_message)
        if position is None or duration is False:
            return
        R_goal = None
        if self.line_orientation.get() == "down":
            try:
                R_goal = kin.top_down_rotation(math.radians(float(self.line_yaw.get())))
            except ValueError:
                self._set_message(self.trajectory_message, "Yaw must be a number.", COLOR_ERROR)
                return
        profile = self.line_profile.get()
        self._start_worker(lambda: self._plan_and_run_line(
            self.client.reference(), position, R_goal, duration, profile), self.trajectory_message)

    def _nudge(self, axis, distance):
        profile = self.line_profile.get()

        def job():
            q_start = self.client.reference()
            position, R = kin.fk(q_start)
            position[axis] += distance
            self._plan_and_run_line(q_start, position, R, None, profile)
        self._start_worker(job, self.trajectory_message)

    def _plan_and_run_line(self, q_start, position, R_goal, duration, profile, label=None):
        label = label or self.trajectory_message
        self._post_message(label, "Planning the straight line (IK + inverse velocity/acceleration "
                                  "kinematics at every sample) ...", COLOR_RUNNING)
        trajectory = traj.task_space_line(q_start, position, R_goal, duration, profile)
        self._run_checked(trajectory, label)

    def _run_checked(self, trajectory, label):
        report = trajectory.check()
        if not report["ok"]:
            self._post_message(label, f"Rejected '{trajectory.label}': {report}", COLOR_ERROR)
            return
        self._post_message(
            label, f"Running {trajectory.label}: peak joint speed {report['peak_joint_speed']:.2f} rad/s, "
                   f"peak accel. {report['peak_joint_acceleration']:.2f} rad/s², "
                   f"min singular value {report['min_singular_value']:.3f}", COLOR_RUNNING)
        state = self.client.run_trajectory(trajectory)
        error = self.client.field("max_error", state)
        self._post_message(label, f"Finished {trajectory.label}: max joint tracking error "
                                  f"{error * 1000:.2f} mrad.", COLOR_OK)

    # ======================================================================
    # Tab 3: controller
    # ======================================================================
    def _build_controller_tab(self):
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="Controller")
        tab.columnconfigure(0, weight=1)

        box = ttk.LabelFrame(tab, text="tau = Kp e + Ki ∫e + Kd e'  + G(q)  + M(q) q''_ref + B q'_ref",
                             padding=10)
        box.grid(row=0, column=0, sticky="ew")
        headers = ["", "Kp (N·m/rad)", "Ki (N·m/rad·s)", "Kd (N·m·s/rad)", "error (mrad)", "torque (N·m)"]
        for column, text in enumerate(headers):
            ttk.Label(box, text=text).grid(row=0, column=column, padx=4)
        self.gain_entries = {"kp": [], "ki": [], "kd": []}
        self.error_labels, self.torque_labels = [], []
        defaults = {"kp": ctl.DEFAULT_KP, "ki": ctl.DEFAULT_KI, "kd": ctl.DEFAULT_KD}
        for joint in range(NUMBER_OF_JOINTS):
            ttk.Label(box, text=JOINT_NAMES[joint]).grid(row=joint + 1, column=0, sticky="w")
            for column, key in enumerate(("kp", "ki", "kd"), start=1):
                entry = ttk.Entry(box, width=9)
                entry.insert(0, f"{defaults[key][joint]:g}")
                entry.grid(row=joint + 1, column=column, padx=4, pady=1)
                self.gain_entries[key].append(entry)
            error = ttk.Label(box, text="--", width=10, anchor="e", font=("TkFixedFont", 9))
            error.grid(row=joint + 1, column=4)
            torque = ttk.Label(box, text="--", width=10, anchor="e", font=("TkFixedFont", 9))
            torque.grid(row=joint + 1, column=5)
            self.error_labels.append(error)
            self.torque_labels.append(torque)

        options = ttk.Frame(box)
        options.grid(row=NUMBER_OF_JOINTS + 1, column=0, columnspan=6, sticky="w", pady=(8, 0))
        ttk.Button(options, text="Apply gains", command=self._apply_gains).pack(side="left", padx=(0, 6))
        ttk.Button(options, text="Defaults", command=self._default_gains).pack(side="left", padx=(0, 16))
        self.use_gravity = tk.BooleanVar(value=True)
        self.use_feedforward = tk.BooleanVar(value=True)
        self.use_logging = tk.BooleanVar(value=True)
        for text, variable in (("Gravity compensation G(q)", self.use_gravity),
                               ("Inertia feed-forward M(q)q''", self.use_feedforward),
                               ("Log trajectories", self.use_logging)):
            ttk.Checkbutton(options, text=text, variable=variable, command=self._apply_options
                            ).pack(side="left", padx=4)
        self.controller_info = ttk.Label(box, text="", font=("TkFixedFont", 9))
        self.controller_info.grid(row=NUMBER_OF_JOINTS + 2, column=0, columnspan=6, sticky="w", pady=(6, 0))

        # ---- live plot ----
        plot_box = ttk.LabelFrame(tab, text="Live tracking", padding=6)
        plot_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        tab.rowconfigure(1, weight=1)
        controls = ttk.Frame(plot_box)
        controls.pack(fill="x")
        self.live_plot = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Live plot", variable=self.live_plot).pack(side="left")
        ttk.Label(controls, text="Joint").pack(side="left", padx=(12, 4))
        self.plot_joint = ttk.Combobox(controls, width=4, state="readonly", values=[1, 2, 3, 4, 5, 6])
        self.plot_joint.set(2)
        self.plot_joint.pack(side="left")
        ttk.Button(controls, text="Make validation plots (latest 10 runs)",
                   command=self._make_validation_plots).pack(side="right")
        self._build_plot(plot_box)

    def _build_plot(self, parent):
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            ttk.Label(parent, text="matplotlib not installed: no live plot").pack()
            self.canvas = None
            return
        self.figure = Figure(figsize=(7.2, 3.6), dpi=90)
        self.axis_angle = self.figure.add_subplot(2, 1, 1)
        self.axis_error = self.figure.add_subplot(2, 1, 2)
        (self.line_ref,) = self.axis_angle.plot([], [], "--", label="reference")
        (self.line_actual,) = self.axis_angle.plot([], [], "-", label="measured")
        self.axis_angle.set_ylabel("angle [deg]")
        self.axis_angle.legend(loc="upper left", fontsize=8)
        self.error_lines = [self.axis_error.plot([], [], lw=1, label=f"e{j + 1}")[0] for j in range(6)]
        self.axis_error.set_ylabel("error [mrad]")
        self.axis_error.set_xlabel("simulation time [s]")
        self.axis_error.legend(loc="upper left", ncol=6, fontsize=7)
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _update_plot(self):
        if self.canvas is None or not self.live_plot.get() or len(self.plot_history) < 2:
            return
        if self.notebook.index(self.notebook.select()) != 2:     # only when the tab is visible
            return
        joint = int(self.plot_joint.get()) - 1
        data = np.array([np.concatenate([[t], r, q]) for t, r, q in self.plot_history])
        t = data[:, 0]
        q_ref, q = data[:, 1:7], data[:, 7:13]
        self.line_ref.set_data(t, np.degrees(q_ref[:, joint]))
        self.line_actual.set_data(t, np.degrees(q[:, joint]))
        for j, line in enumerate(self.error_lines):
            line.set_data(t, 1000 * (q_ref[:, j] - q[:, j]))
        for axis in (self.axis_angle, self.axis_error):
            axis.relim()
            axis.autoscale_view()
            axis.set_xlim(t[0], max(t[-1], t[0] + 1.0))
        self.canvas.draw_idle()

    def _apply_gains(self):
        try:
            gains = {key: [float(e.get()) for e in entries] for key, entries in self.gain_entries.items()}
        except ValueError:
            self.controller_info.config(text="Gains must be numbers.", foreground=COLOR_ERROR)
            return
        self.client.set_gains(gains["kp"], gains["ki"], gains["kd"])
        self.controller_info.config(text="Gains sent to the controller.", foreground=COLOR_OK)

    def _default_gains(self):
        for key, values in (("kp", ctl.DEFAULT_KP), ("ki", ctl.DEFAULT_KI), ("kd", ctl.DEFAULT_KD)):
            for entry, value in zip(self.gain_entries[key], values):
                entry.delete(0, "end")
                entry.insert(0, f"{value:g}")
        self._apply_gains()

    def _apply_options(self):
        self.client.set_options(gravity=self.use_gravity.get(), feedforward=self.use_feedforward.get(),
                                logging=self.use_logging.get())

    def _make_validation_plots(self):
        script = os.path.join(VALIDATION_DIR, "plot_validation.py")
        if not os.path.exists(script):
            self.controller_info.config(text=f"Not found: {script}", foreground=COLOR_ERROR)
            return
        subprocess.Popen([sys.executable, script, "--latest", "10"])
        self.controller_info.config(text=f"Plotting in the background -> {VALIDATION_DIR}/figures",
                                    foreground=COLOR_OK)

    # ======================================================================
    # Tab 4: application
    # ======================================================================
    def _build_application_tab(self):
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="Application")
        tab.columnconfigure(0, weight=1)
        ttk.Label(tab, wraplength=720, justify="left", text=(
            "Laboratory sample sorting: each vial is picked from the tray and placed in the bin that "
            "matches its cap colour. Large moves are joint-space quintic trajectories; approaching, "
            "lifting and placing are task-space straight lines. Positions come from the scene "
            "(pre-programmed, no camera).")).grid(row=0, column=0, sticky="w")
        buttons = ttk.Frame(tab)
        buttons.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Button(buttons, text="Sort all vials", command=lambda: self._run_task("sort")).pack(side="left", padx=(0, 6))
        if self.layout is not None:
            for name, colour, _ in self.layout.vials:
                short = name.replace("lab_sample_", "").replace("_", " ")
                ttk.Button(buttons, text=f"{short} ({colour})",
                           command=lambda n=name: self._run_task("sort", [n])).pack(side="left", padx=2)
        buttons2 = ttk.Frame(tab)
        buttons2.grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Button(buttons2, text="Lift demo: Erlenmeyer flask (150 g)",
                   command=lambda: self._run_task("lift")).pack(side="left", padx=(0, 6))
        ttk.Button(buttons2, text="Home", command=lambda: self._run_joint_trajectory(
            app.HOME, "quintic", None, self.task_message)).pack(side="left")

        self.step_list = tk.Listbox(tab, height=16, font=("TkFixedFont", 9), activestyle="none")
        self.step_list.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        tab.rowconfigure(3, weight=1)
        self.task_message = ttk.Label(tab, text="", wraplength=720, justify="left")
        self.task_message.grid(row=4, column=0, sticky="w", pady=(6, 0))
        if self.layout is None:
            self._set_message(self.task_message, "Scene not loaded (model_path): application disabled.",
                              COLOR_ERROR)

    def _run_task(self, which, vial_names=None):
        if self.layout is None:
            return
        steps = (app.plan_sorting(self.layout, vial_names) if which == "sort"
                 else app.plan_lift_demo(self.layout))
        self.step_list.delete(0, "end")
        for k, step in enumerate(steps, 1):
            self.step_list.insert("end", f"{k:2d}. [{step.kind:7s}] {step.description}")

        def report(index, step, message):
            def update():
                text = f"{index + 1:2d}. [{step.kind:7s}] {step.description}"
                if message != "running":
                    text += f"   -> {message}"
                self.step_list.delete(index)
                self.step_list.insert(index, text)
                self.step_list.itemconfig(index, foreground=COLOR_RUNNING if message == "running" else COLOR_OK)
                self.step_list.see(index)
            self.root.after(0, update)

        def job():
            start = time.time()
            results = self.client.execute(steps, report, meta={"experiment": which})
            worst = max((r[2] for r in results), default=0.0)
            self._post_message(self.task_message,
                               f"Task finished in {time.time() - start:.0f} s: {len(results)} trajectories, "
                               f"worst joint tracking error {worst * 1000:.2f} mrad.", COLOR_OK)
        self._start_worker(job, self.task_message)

    # ======================================================================
    # Background work
    # ======================================================================
    def _start_worker(self, job, label):
        if self.worker is not None and self.worker.is_alive():
            self._set_message(label, "Busy: wait for the current motion (or press STOP).", COLOR_WARNING)
            return
        if self.client.get_state() is None:
            self._set_message(label, "No controller state yet: is controller_node.py running?", COLOR_ERROR)
            return
        self.client.stop_event.clear()

        def wrapper():
            try:
                job()
            except TaskAborted:
                self._post_message(label, "Stopped.", COLOR_WARNING)
            except (traj.TrajectoryError, ValueError, TimeoutError) as error:
                self._post_message(label, f"Not executed: {error}", COLOR_ERROR)
        self.worker = threading.Thread(target=wrapper, daemon=True)
        self.worker.start()

    def _post_message(self, label, text, color):
        self.root.after(0, lambda: self._set_message(label, text, color))

    @staticmethod
    def _set_message(label, text, color):
        label.config(text=text, foreground=color)

    # ======================================================================
    # Periodic refresh (every 50 ms)
    # ======================================================================
    def _refresh(self):
        self.refresh_count += 1
        state = self.client.get_state()
        if state is not None:
            self._update_from_state(state)
        elif self.refresh_count > 60:
            self.status_label.config(text="No /controller/state: start controller_node.py",
                                     foreground=COLOR_ERROR)
        if self.refresh_count % 4 == 0:
            self._update_plot()
        self.root.after(50, self._refresh)

    def _update_from_state(self, state):
        field = lambda name: self.client.field(name, state)
        q, q_ref, qd = field("q"), field("q_ref"), field("qd")
        mode = int(field("mode"))
        t = float(field("time"))

        if self.slider_targets is None:                       # first data: unlock the sliders
            self._show_on_sliders(q_ref)
            for slider in self.sliders:
                slider.state(["!disabled"])
            self._fill_xyz(self.point_entries)
            self._fill_xyz(self.line_entries)
            self._fill_joint_entries()
        busy = self.worker is not None and self.worker.is_alive()
        if mode == 1 or busy or self.previous_mode == 1:      # sliders follow planned motions
            self._show_on_sliders(q_ref)
        self.previous_mode = mode

        self.status_label.config(text=f"Connected   sim time {t:7.2f} s", foreground=COLOR_OK)
        if mode == 1:
            self.mode_label.config(text=f"TRAJECTORY #{int(field('trajectory_id'))}  "
                                        f"{100 * field('progress'):3.0f}%")
        else:
            self.mode_label.config(text="HOLD / JOG" + ("  (task running)" if busy else ""))

        for joint in range(NUMBER_OF_JOINTS):
            self.actual_labels[joint].config(text=f"actual {math.degrees(q[joint]):6.1f}°")
            self.error_labels[joint].config(text=f"{1000 * (q_ref[joint] - q[joint]):+8.3f}")
            self.torque_labels[joint].config(text=f"{field('torque')[joint]:+8.2f}")

        position, _ = kin.fk(q)
        sensed = field("tcp_sim")
        self.fk_label.config(text=f"DH forward kinematics   X {position[0]:+.4f}  Y {position[1]:+.4f}  "
                                  f"Z {position[2]:+.4f} m")
        if np.all(np.isfinite(sensed)):
            gap = 1e6 * np.linalg.norm(position - sensed)
            self.sim_label.config(text=f"Sensed in MuJoCo        X {sensed[0]:+.4f}  Y {sensed[1]:+.4f}  "
                                       f"Z {sensed[2]:+.4f} m   (|diff| {gap:.0f} µm)")
        twist = kin.fk_velocity(q, qd)
        self.velocity_label.config(
            text=f"Velocity J(q) q'        v [{twist[0]:+.3f} {twist[1]:+.3f} {twist[2]:+.3f}] m/s   "
                 f"w [{twist[3]:+.2f} {twist[4]:+.2f} {twist[5]:+.2f}] rad/s")
        sigma = kin.singularity_measure(q)["sigma_min"]
        near = sigma < kin.SINGULARITY_EPS
        self.singularity_label.config(
            text=f"Singularity             smallest singular value {sigma:.4f}"
                 + ("   NEAR SINGULARITY" if near else ""), foreground=COLOR_ERROR if near else "")
        self.controller_info.config(
            text=f"last/running trajectory #{int(field('trajectory_id'))}: max |q_ref - q| = "
                 f"{1000 * field('max_error'):.3f} mrad    saturated motors now: {int(field('saturated'))}    "
                 f"gripper force {field('gripper_force'):+.1f} N", foreground="")
        if not self.plot_history or t > self.plot_history[-1][0]:
            self.plot_history.append((t, q_ref.copy(), q.copy()))
        elif t < self.plot_history[-1][0]:
            self.plot_history.clear()                         # simulation restarted


# ===========================================================================
# Program entry point
# ===========================================================================
def _spin_quietly(executor):
    """Spin until shutdown; Ctrl-C (ROS shuts the context down) is a normal way to quit."""
    try:
        executor.spin()
    except ExternalShutdownException:
        pass


def main():
    rclpy.init()
    node = GuiNode()
    model_path = str(node.get_parameter("model_path").value)
    layout, checker = None, None
    if model_path:
        try:
            layout = app.load_layout(model_path)
            checker = CollisionChecker(model_path)
        except Exception as error:                            # GUI still works without the scene
            node.get_logger().error(f"Could not load the scene ({error}): no application, "
                                    f"no collision check.")
    else:
        node.get_logger().warn("No model_path: application tab and collision check disabled.")
    client = TaskClient(node, checker)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=_spin_quietly, args=(executor,), daemon=True)
    spinner.start()

    root = tk.Tk()
    ControllerWindow(root, client, layout)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()              # stop spinning before the node goes away
        spinner.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
