#!/usr/bin/env python3
"""
Run Milestone 4 scenarios through ROS2 without the GUI (mujoco_node + controller_node must run).

    python3 run_task.py compare   # validation experiments (profiles, control laws, joint vs task space)
    python3 run_task.py sort      # the industrial application: sort the 4 vials
    python3 run_task.py lift      # lifting demo: Erlenmeyer flask
    python3 run_task.py all       # compare + sort + lift

    optional: --ros-args -p model_path:=/path/to/lab-scene.xml

Every trajectory is logged by controller_node.py (validation/runs/); make the plots with
    python3 ../../../validation/plot_validation.py
"""
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node

import application as app
import kinematics as kin
import trajectory as traj
from collision import CollisionChecker
from task_runner import TaskClient

DEFAULT_MODEL = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                             "model", "ufactory_lite6", "lab-scene.xml"))


def print_result(label, state, client):
    print(f"  {label:58s} max joint error {client.field('max_error', state) * 1000:7.3f} mrad")


def compare(client, layout):
    """Validation experiments. All start and end at the same poses so they can be compared."""
    above_vial = app.solve_pose(np.array([*layout.vials[0][2][:2], app.TRANSIT_HEIGHT]),
                                app.radial_yaw(layout.vials[0][2]), app.HOME)
    bin_xy = layout.bins["blue"][:2]
    above_bin = app.solve_pose(np.array([*bin_xy, app.TRANSIT_HEIGHT]), app.radial_yaw(bin_xy), above_vial)

    def to(q_goal, duration=2.5, profile="quintic", tag=""):
        trajectory = traj.joint_space(client.reference(), q_goal, duration, profile)
        state = client.run_trajectory(trajectory, {"experiment": tag})
        print_result(f"[{tag}] {trajectory.label}", state, client)

    client.set_options(gravity=True, feedforward=True)
    to(app.HOME, tag="reset")

    print("1) joint-space profiles: cubic vs quintic (same move, same duration)")
    for profile in ("cubic", "quintic"):
        to(above_vial, profile=profile, tag=f"profile-{profile}")
        to(app.HOME, profile=profile, tag=f"profile-{profile}-back")

    print("2) control law: PID only / PID + gravity compensation / + inertia feed-forward")
    for name, gravity, feedforward in (("pid-only", False, False), ("pid+gravity", True, False),
                                       ("pid+gravity+feedforward", True, True)):
        client.set_options(gravity=gravity, feedforward=feedforward)
        time.sleep(0.2)
        client.wait_sim_time(1.0)          # let the integral term absorb the load change
        to(above_vial, profile="quintic", tag=f"control-{name}")
        to(app.HOME, profile="quintic", tag=f"control-{name}-back")
    client.set_options(gravity=True, feedforward=True)
    client.wait_sim_time(0.5)
    to(app.HOME, duration=1.0, tag="reset")

    print("3) same start/end pose: joint-space quintic (curved TCP path) vs task-space line (straight)")
    to(above_vial, tag="reset")
    p_bin, R_bin = kin.fk(above_bin)
    for name in ("joint-space", "task-space"):
        if name == "joint-space":
            trajectory = traj.joint_space(client.reference(), above_bin, 3.0, "quintic")
        else:
            trajectory = traj.task_space_line(client.reference(), p_bin, R_bin, 3.0, "quintic")
        state = client.run_trajectory(trajectory, {"experiment": f"path-{name}"})
        print_result(f"[path-{name}] {trajectory.label}", state, client)
        to(above_vial, duration=2.0, tag="reset")

    print("4) task-space line: cubic vs quintic time scaling (straight down 12 cm)")
    p, R = kin.fk(client.reference())
    for profile in ("cubic", "quintic"):
        trajectory = traj.task_space_line(client.reference(), p - [0, 0, 0.12], R, 1.5, profile)
        state = client.run_trajectory(trajectory, {"experiment": f"line-{profile}"})
        print_result(f"[line-{profile}] {trajectory.label}", state, client)
        trajectory = traj.task_space_line(client.reference(), p, R, 1.5, profile)
        client.run_trajectory(trajectory, {"experiment": f"line-{profile}-back"})
    to(app.HOME, tag="reset")


def run_steps(client, steps, tag):
    def report(index, step, message):
        if message == "running":
            print(f"  [{index + 1:2d}/{len(steps)}] {step.description}")
    results = client.execute(steps, report, meta={"experiment": tag})
    worst = max((r[2] for r in results), default=0.0)
    print(f"  {tag}: {len(results)} trajectories, worst joint tracking error {worst * 1000:.2f} mrad")


def _spin_quietly(executor):
    """Spin until shutdown; Ctrl-C (ROS shuts the context down) is a normal way to quit."""
    try:
        executor.spin()
    except ExternalShutdownException:
        pass


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "all"
    rclpy.init(args=sys.argv)
    node = Node("lite6_task_runner")
    node.declare_parameter("model_path", DEFAULT_MODEL)
    model_path = str(node.get_parameter("model_path").value)
    client = TaskClient(node, CollisionChecker(model_path))
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=_spin_quietly, args=(executor,), daemon=True)
    spinner.start()
    try:
        client.wait_for_state()
        layout = app.load_layout(model_path)
        if scenario in ("compare", "all"):
            compare(client, layout)
        if scenario in ("sort", "all"):
            print("Sorting the sample vials")
            run_steps(client, app.plan_sorting(layout), "sorting")
        if scenario in ("lift", "all"):
            print("Lifting demo: Erlenmeyer flask")
            run_steps(client, app.plan_lift_demo(layout), "lift")
    finally:
        executor.shutdown()              # stop spinning before the node goes away
        spinner.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
