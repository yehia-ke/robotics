# Milestone 04: Trajectory planning, motor-level control and the industrial application

UFactory Lite6 (6 DoF) with the UFactory gripper, ROS2 Jazzy + MuJoCo 3.13, Python.
Application: **laboratory sample sorting**. Four capped vials are picked from a tray and placed in the
bin that matches their cap colour. A second demo lifts an Erlenmeyer flask (150 g) to show the arm
carrying a heavier load.

## What is implemented

| Requirement | Where |
|---|---|
| Task-space trajectories (straight line, cubic / quintic time scaling) | `src/src/nodes/trajectory.py` (`StraightLine`, `task_space_line`) |
| Joint-space trajectories (cubic, quintic) | `src/src/nodes/trajectory.py` (`cubic_coefficients`, `quintic_coefficients`, `joint_space`) |
| Motor-level control (PID + gravity compensation + inertia feed-forward, torques) | `controller.py` (control law), `dynamics.py` (G(q), M(q) from DH), `controller_node.py` (ROS2 loop) |
| Validate task- and joint-space trajectories on ROS2 + MuJoCo | `run_task.py compare`, logs in `validation/runs`, `validation/plot_validation.py` |
| Send the joint angles to the simulator from the GUI | `sliders_node.py`: *Trajectories* and *Application* tabs |
| Closed loop tested with the visualisation | torque control in the loop. The GUI shows live errors, torques and a plot. `/ee_pose` comes from MuJoCo. Planned motions are collision-checked in the scene first (`collision.py`) |
| Industrial application | `application.py` (sorting + flask lift), *Application* tab or `run_task.py sort` |
| Kinematics (restored DH code from Milestones 2-3, now with the gripper TCP) | `kinematics.py` |

## Architecture

```
 sliders_node.py (Tk GUI)          run_task.py (headless scenarios)
   plans: trajectory.py / application.py, checks: collision.py
          |  /controller/trajectory  (trajectory_msgs/JointTrajectory: t, q, q', q'')
          |  /controller/jog_target, /controller/gripper, /controller/command (JSON)
          v
 controller_node.py  (Python, 250 Hz of simulation time)
   tau = Kp e + Ki ∫e + Kd e' + G(q) + M(q) q''_ref + B q'_ref        (controller.py, dynamics.py)
          |  /joint_commands  (6 joint torques N*m + gripper force N)
          v                                         ^  /joint_state (sim-time stamps), /ee_pose
 mujoco_node (C++, 1 kHz, TORQUE mode, lockstep: 4 steps per command)  --------------'
```

* **Torque control:** the six arm actuators are `<motor>`s (`lite6_gripper_torque.xml`), so `ctrl` is a
  torque. All of the control is done by the Python controller. In `TORQUE` mode `mujoco_node` holds each
  command until the next one arrives and adds no gravity compensation of its own. Until the controller
  connects, and whenever commands stop for 0.5 s, the node holds the arm in place ("safe hold").
* **Lockstep:** the simulator advances exactly 4 steps (4 ms) per torque command, then waits. ROS or
  Python latency therefore never adds loop delay. Without lockstep, a 12 ms latency spike makes the loop
  unstable; this was tested offline.
* **Simulation-time stamps:** `/joint_state` and `/ee_pose` carry MuJoCo time. The controller and the
  logs use it as their clock.

## Build and run

```bash
cd "Milestone 04/Project Files"
source /opt/ros/jazzy/setup.bash
colcon build --packages-select mujoco_ros2      # MuJoCo is found automatically (repo copy), or
                                                # add --cmake-args -DMUJOCO_PATH=/path/to/mujoco-3.13.0
source install/setup.bash
ros2 launch mujoco_ros2 example.py               # MuJoCo window + controller + GUI
```

Rebuild after changing anything in `src/model` (the launch file uses the installed copy). Python
needs `mujoco numpy matplotlib`.

Launch options: `gui:=false` (no Tk window), `headless:=true` (no MuJoCo window).

MuJoCo window: left-drag rotates, right-drag pans (Shift for the horizontal plane), scroll zooms, and
**C** cycles through the scene cameras (overview, robot cell, ...).

### GUI tabs
* **Joints**: jog sliders (smooth trapezoidal set-point), TCP from DH forward kinematics next to the
  TCP sensed in MuJoCo, live TCP velocity J(q)q', singularity measure, IK "go to point" (joint space or
  straight line).
* **Trajectories**: joint-space cubic/quintic to typed angles or presets (above each vial / bin),
  task-space straight line to an XYZ target (keep orientation or gripper-down with a yaw), ±50 mm
  nudges, gripper open/close. Each plan is checked before it runs: joint limits, speed, singularity
  margin and collisions.
* **Controller**: Kp/Ki/Kd per joint, switches for gravity compensation, feed-forward and logging, live
  error and torque per joint, live plot, and a button that runs the validation plots.
* **Application**: sort all vials or a single one, the flask lift demo, Home, and a step list showing
  progress.
* **STOP** (top right): the arm brakes smoothly and holds.

## Validation (ROS2 + MuJoCo)

```bash
# terminal 1
ros2 launch mujoco_ros2 example.py headless:=true gui:=false
# terminal 2
cd src/src/nodes
python3 run_task.py all          # compare | sort | lift | all
python3 ../../../validation/plot_validation.py   # -> validation/figures/*.png, validation/summary.md
```

`controller_node.py` logs every trajectory to `validation/runs/` (CSV + JSON; ignored by git, so
regenerate them with the commands above). Each log holds the reference, the measured joints, the
torques, the gravity term, the TCP reference and the TCP sensed in MuJoCo. The results of the last run
are in `validation/summary.md` and `validation/figures/`.

Results (PID + G + FF = the default controller):

| Test | Result |
|---|---|
| DH FK of the measured angles vs the TCP sensed in MuJoCo, all runs | ≤ 0.002 mm |
| Sorting task (25 trajectories) | worst joint error 0.64 mrad, all 4 vials in the right bin |
| Flask lift (7 trajectories, 150 g payload) | worst joint error 1.07 mrad |
| Task-space lines: sensed TCP distance from the commanded line | ≤ 0.17 mm |
| Same joint-space move: PID only / PID+G / PID+G+FF | 7.1 / 5.9 / 0.30 mrad max error |
| Joint-space vs task-space between the same two poses | joint-space path bows 77 mm off the straight line; task-space ≤ 0.05 mm |
| Cubic vs quintic | cubic acceleration (and torque) jumps at the start and end; quintic is smooth |

Self-tests that need no ROS:

```bash
cd src/src/nodes
python3 kinematics.py ../../model/ufactory_lite6/lab-scene.xml   # DH vs MuJoCo: position/velocity/acceleration, forward+inverse
python3 dynamics.py   ../../model/ufactory_lite6/lab-scene.xml   # G(q), M(q) vs MuJoCo
python3 trajectory.py                                            # polynomial boundary conditions, straight-line accuracy
python3 collision.py                                             # collision checker
```

## Changes to the model and scene (vs Milestone 3)
* `lite6_gripper_torque.xml` is new. It is the Menagerie Lite6 with the wide gripper, the arm
  actuators changed to torque motors, and the false link4–link6 contact excluded. That contact came
  from overlapping convex hulls and blocked joint 5 above ~93° (the real limit is 124°).
* `lab-scene.xml` now includes the gripper robot, and the `lab_home` keyframe has the 2 finger joints
  and 7 zero controls. The three sort bins moved 4–6 cm closer to the robot (y = −0.31 → −0.27). The red
  bin was at the edge of the reach, with the elbow almost straight and near a singularity.
* `mujoco_node` gained: torque-mode hold and lockstep, safe hold, simulation-time stamps, `/ee_pose`
  (site `end_effector`), the `initial_keyframe` and `headless` parameters, and camera mouse/keyboard
  control. POSITION and VELOCITY modes are unchanged.
