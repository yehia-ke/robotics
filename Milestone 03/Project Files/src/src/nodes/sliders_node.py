#!/usr/bin/env python3
"""
Joint sliders + end-effector (Cartesian) control for MuJoCo + ROS2
(woolfrey/mujoco_ros2, UFactory Lite6).

What it does
------------
* Sliders set TARGET JOINT ANGLES. The node reads /joint_state and publishes
  smooth, speed-limited commands on /joint_commands.
* Forward kinematics: the live end-effector position is shown in the window.
* Inverse kinematics: type X Y Z (metres, world frame), press "Go to point";
  the solved joint angles are applied to the sliders and the arm moves there.

The kinematics use the MuJoCo model itself, so pass the scene path:

    python3 joint_slider_gui.py --ros-args \
        -p mode:=position -p model_path:=/path/to/scene.xml

Optional parameters:
    -p ee_name:=<site or body name>   end effector to control (default: auto-detect)
    -p state_topic:=/joint_state      where joint angles are read from
    -p command_topic:=/joint_commands where commands are published
    -p mode:=velocity|position        must match control_mode in the launch file

Requires: pip install mujoco numpy
"""

import math
import threading
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUMBER_OF_JOINTS = 6

# (name, lowest_angle_rad, highest_angle_rad)
# Fallback UFactory Lite6 limits, only used when the model can't be loaded.
FALLBACK_JOINT_LIMITS = [
    ("Joint 1", -2 * math.pi, 2 * math.pi),
    ("Joint 2", -2.61799, 2.61799),
    ("Joint 3", -0.061087, 5.235988),
    ("Joint 4", -2 * math.pi, 2 * math.pi),
    ("Joint 5", -2.1642, 2.1642),
    ("Joint 6", -2 * math.pi, 2 * math.pi),
]

COLOR_OK = "#2e7d32"
COLOR_WARNING = "#b36b00"
COLOR_ERROR = "#c62828"

POSITION_TOLERANCE_METRES = 5e-4        # IK loop stops below this position error
ROTATION_TOLERANCE = 5e-3               # IK loop stops below this orientation error
ACCEPTABLE_POSITION_ERROR_METRES = 1e-3 # IK result counts as "reached" below this
ACCEPTABLE_ROTATION_ERROR = 1e-2
MAX_JOINT_STEP_RADIANS = 0.3            # largest joint change per IK iteration


USE_VELOCITY_ACCELERATION_COMMANDS = True
END_EFFECTOR_VELOCITY_TOPIC = "/ee_velocity"          # [vx vy vz wx wy wz]   m/s, rad/s
END_EFFECTOR_ACCELERATION_TOPIC = "/ee_acceleration"  # [ax ay az alx aly alz] m/s^2, rad/s^2
# Lite6: the smallest singular value of J is ~0.05 in a typical pose and 0 when singular
# (e.g. joint 5 = 0). Below the threshold the inverse is damped so it stays finite;
# above it the inverse is exact.
SINGULARITY_THRESHOLD = 0.01            # smallest singular value of J below which damping starts
MAX_SINGULARITY_DAMPING = 0.01          # damping used exactly at a singularity


def clamp_value(value, lowest, highest):
    """Return `value` forced into the range [lowest, highest]."""
    return max(lowest, min(highest, value))


# ===========================================================================
# Kinematics: forward + inverse, computed with the MuJoCo model
# ===========================================================================
class ArmKinematics:
    """Forward and inverse kinematics for the robot arm using the MuJoCo model."""

    # Site names tried (in order) when no end effector name is given.
    COMMON_END_EFFECTOR_NAMES = ["attachment_site", "ee_site", "end_effector", "tcp", "tool0"]

    def __init__(self, model_path, number_of_joints, end_effector_name=""):
        """Load the model, locate the arm joints, read their limits, find the end effector."""
        import numpy
        import mujoco

        self.numpy = numpy
        self.mujoco = mujoco

        self._load_model_and_data(model_path)
        arm_joint_ids = self._find_arm_joint_ids(number_of_joints)
        self._store_joint_memory_addresses(arm_joint_ids)
        self._read_joint_limits(arm_joint_ids)
        self._find_end_effector(end_effector_name, arm_joint_ids)

    # ----- setup helpers -------------------------------------------------
    def _load_model_and_data(self, model_path):
        """Parse the MJCF/XML file into a model, and create a scratch data object.

        The data object is private to this class (it is NOT the running
        simulation); it is only used to compute poses for hypothetical angles.
        """
        self.model = self.mujoco.MjModel.from_xml_path(model_path)
        self.data = self.mujoco.MjData(self.model)

    def _find_arm_joint_ids(self, number_of_joints):
        """Return the ids of the first N hinge/slide joints in the model.

        Ball and free joints are skipped. The first N movable joints are
        assumed to be the arm joints, in the same order as the joint_state topic.
        """
        movable_joint_types = (int(self.mujoco.mjtJoint.mjJNT_HINGE),
                               int(self.mujoco.mjtJoint.mjJNT_SLIDE))
        movable_joint_ids = [joint_id for joint_id in range(self.model.njnt)
                             if int(self.model.jnt_type[joint_id]) in movable_joint_types]
        if len(movable_joint_ids) < number_of_joints:
            raise RuntimeError(
                f"Model has only {len(movable_joint_ids)} joints, need {number_of_joints}")
        return movable_joint_ids[:number_of_joints]

    def _store_joint_memory_addresses(self, arm_joint_ids):
        """Remember where each arm joint lives inside MuJoCo's qpos / velocity arrays.

        MuJoCo packs all joint values in one array, so we need each joint's index
        to read or write it, and its Jacobian column.
        """
        self.angle_addresses = [int(self.model.jnt_qposadr[j]) for j in arm_joint_ids]
        self.velocity_addresses = [int(self.model.jnt_dofadr[j]) for j in arm_joint_ids]

    def _read_joint_limits(self, arm_joint_ids):
        """Read each joint's name and min/max angle from the model.

        Joints without limits in the XML get a +/-180 degree range. Results are
        stored both as a readable list and as numpy arrays for fast clamping.
        """
        self.joint_limits = []
        for joint_id in arm_joint_ids:
            name = (self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                    or f"Joint {len(self.joint_limits) + 1}")
            if self.model.jnt_limited[joint_id]:
                lowest = float(self.model.jnt_range[joint_id][0])
                highest = float(self.model.jnt_range[joint_id][1])
            else:
                lowest, highest = -math.pi, math.pi
            self.joint_limits.append((name, lowest, highest))
        self.lowest_angles = self.numpy.array([limit[1] for limit in self.joint_limits])
        self.highest_angles = self.numpy.array([limit[2] for limit in self.joint_limits])

    def _find_end_effector(self, end_effector_name, arm_joint_ids):
        """Pick the point on the arm that kinematics refers to.

        Order of preference: the name you gave, then common tool-site names,
        and finally the body attached to the last arm joint.
        """
        self.end_effector_site_id = -1
        self.end_effector_body_id = -1

        names_to_try = [end_effector_name] if end_effector_name else self.COMMON_END_EFFECTOR_NAMES
        for name in names_to_try:
            if self._try_use_named_element(name):
                return
        self._use_last_arm_link(arm_joint_ids[-1])

    def _try_use_named_element(self, name):
        """Look for a site, then a body, called `name`; remember it if found.

        Uses MuJoCo's name lookup, which returns -1 when nothing matches.
        Returns True if the element was found.
        """
        site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            self.end_effector_site_id = site_id
            self.end_effector_description = f"site '{name}'"
            return True
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            self.end_effector_body_id = body_id
            self.end_effector_description = f"body '{name}'"
            return True
        return False

    def _use_last_arm_link(self, last_joint_id):
        """Fallback end effector: the body that the last arm joint moves."""
        self.end_effector_body_id = int(self.model.jnt_bodyid[last_joint_id])
        body_name = self.mujoco.mj_id2name(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, self.end_effector_body_id)
        self.end_effector_description = (
            f"body '{body_name}' (last arm link; pass -p ee_name:=<site> for the tool tip)")

    # ----- forward kinematics ---------------------------------------------
    def forward_kinematics(self, joint_angles):
        """Return (position, rotation_matrix) of the end effector for given joint angles.

        How: write the angles into the scratch data, ask MuJoCo to propagate
        them down the kinematic chain (mj_fwdPosition), then read the end
        effector's world position and 3x3 orientation.
        """
        self.data.qpos[self.angle_addresses] = joint_angles
        self.mujoco.mj_fwdPosition(self.model, self.data)

        if self.end_effector_site_id >= 0:
            site_id = self.end_effector_site_id
            return (self.data.site_xpos[site_id].copy(),
                    self.data.site_xmat[site_id].reshape(3, 3).copy())
        body_id = self.end_effector_body_id
        return (self.data.xpos[body_id].copy(),
                self.data.xmat[body_id].reshape(3, 3).copy())

    # ----- inverse kinematics: building blocks ------------------------------
    def _compute_jacobians(self):
        """Return (position_jacobian, rotation_jacobian), each 3 x number_of_joints.

        The Jacobian says how the end effector's position / rotation changes
        per unit change of each joint. MuJoCo computes it for the whole model
        (columns for every degree of freedom); we keep only the arm's columns.
        Must be called right after forward_kinematics() for the same angles.
        """
        position_jacobian = self.numpy.zeros((3, self.model.nv))
        rotation_jacobian = self.numpy.zeros((3, self.model.nv))
        if self.end_effector_site_id >= 0:
            self.mujoco.mj_jacSite(self.model, self.data, position_jacobian,
                                   rotation_jacobian, self.end_effector_site_id)
        else:
            self.mujoco.mj_jacBody(self.model, self.data, position_jacobian,
                                   rotation_jacobian, self.end_effector_body_id)
        return (position_jacobian[:, self.velocity_addresses],
                rotation_jacobian[:, self.velocity_addresses])

    def _orientation_error(self, current_rotation, target_rotation):
        """Return a 3-vector: the axis*angle rotation that turns current into target.

        How: for each of the 3 axes (columns of the rotation matrices), the cross
        product of the current axis with the target axis points along the
        rotation needed. Summing and halving them gives a small-angle error.
        """
        axis_errors = [self.numpy.cross(current_rotation[:, axis], target_rotation[:, axis])
                       for axis in range(3)]
        return 0.5 * sum(axis_errors)

    def _stack_error_and_jacobian(self, position_error, orientation_error):
        """Combine position (and optionally orientation) into one error + Jacobian.

        With orientation: 6 numbers (3 position + 3 rotation) and a 6 x N Jacobian.
        Without: just position (3 numbers, 3 x N). Pass orientation_error=None to skip it.
        """
        position_jacobian, rotation_jacobian = self._compute_jacobians()
        if orientation_error is None:
            return position_error, position_jacobian
        error = self.numpy.concatenate([position_error, orientation_error])
        jacobian = self.numpy.vstack([position_jacobian, rotation_jacobian])
        return error, jacobian

    def _damped_least_squares_step(self, jacobian, error, damping=0.05):
        """Return the joint-angle change that reduces `error`.

        Formula: step = J^T (J J^T + damping^2 I)^-1 error
        The damping term keeps the step finite near singular poses, where a
        plain inverse would explode.
        """
        rows = jacobian.shape[0]
        regularised = jacobian @ jacobian.T + damping ** 2 * self.numpy.eye(rows)
        return jacobian.T @ self.numpy.linalg.solve(regularised, error)

    def _limit_step_size(self, joint_step):
        """Scale the step down if any joint would move more than the allowed amount.

        Big steps make the linearised Jacobian inaccurate, so we take
        several small ones instead.
        """
        largest_change = self.numpy.max(self.numpy.abs(joint_step))
        if largest_change > MAX_JOINT_STEP_RADIANS:
            joint_step = joint_step * (MAX_JOINT_STEP_RADIANS / largest_change)
        return joint_step

    def _has_converged(self, position_error, orientation_error):
        """True when both position and (if used) orientation errors are tiny."""
        position_ok = self.numpy.linalg.norm(position_error) < POSITION_TOLERANCE_METRES
        if orientation_error is None:
            return position_ok
        return position_ok and self.numpy.linalg.norm(orientation_error) < ROTATION_TOLERANCE

    def _solve_from_starting_angles(self, target_position, target_rotation,
                                    starting_angles, max_iterations=300):
        """Run the iterative solver once from one starting guess.

        How: repeat { compute pose -> measure error -> step the joints toward
        reducing it -> clamp to joint limits } until converged or out of
        iterations. Returns (angles, final_position_error, final_orientation_error).
        """
        joint_angles = self.numpy.clip(
            self.numpy.array(starting_angles, dtype=float),
            self.lowest_angles, self.highest_angles)

        for _ in range(max_iterations):
            position, rotation = self.forward_kinematics(joint_angles)
            position_error = target_position - position
            orientation_error = (None if target_rotation is None
                                 else self._orientation_error(rotation, target_rotation))
            if self._has_converged(position_error, orientation_error):
                break
            error, jacobian = self._stack_error_and_jacobian(position_error, orientation_error)
            joint_step = self._limit_step_size(self._damped_least_squares_step(jacobian, error))
            joint_angles = self.numpy.clip(joint_angles + joint_step,
                                           self.lowest_angles, self.highest_angles)

        return (joint_angles,) + self._measure_final_errors(
            joint_angles, target_position, target_rotation)

    def _measure_final_errors(self, joint_angles, target_position, target_rotation):
        """Return (position_error_metres, orientation_error) for the given angles."""
        position, rotation = self.forward_kinematics(joint_angles)
        position_error = float(self.numpy.linalg.norm(target_position - position))
        orientation_error = 0.0
        if target_rotation is not None:
            orientation_error = float(self.numpy.linalg.norm(
                self._orientation_error(rotation, target_rotation)))
        return position_error, orientation_error

    def _make_starting_guesses(self, current_angles, number_of_random_restarts):
        """List of starting guesses: the current pose first, then random poses.

        Starting from the current pose gives the smallest, most natural motion.
        Random poses are fallbacks in case that start gets stuck.
        The random generator is seeded so results are repeatable.
        """
        random_generator = self.numpy.random.default_rng(0)
        random_guesses = [random_generator.uniform(self.lowest_angles, self.highest_angles)
                          for _ in range(number_of_random_restarts)]
        return [self.numpy.array(current_angles, dtype=float)] + random_guesses

    def _is_acceptable(self, position_error, orientation_error, uses_orientation):
        """True if the solution is close enough to the target to count as reached."""
        if position_error >= ACCEPTABLE_POSITION_ERROR_METRES:
            return False
        return (not uses_orientation) or orientation_error < ACCEPTABLE_ROTATION_ERROR

    # ----- inverse kinematics: public ----------------------------------------
    def inverse_kinematics(self, target_position, current_angles,
                           target_rotation=None, number_of_random_restarts=10):
        """Find joint angles that put the end effector at `target_position`.

        Pass `target_rotation` (3x3) to also control orientation, or None for
        position only. Tries the current pose first, then random restarts, and
        returns the first acceptable answer.
        Returns (joint_angles, position_error_m, orientation_error, reached).
        If nothing reaches the target, returns the closest attempt with reached=False.
        """
        target_position = self.numpy.array(target_position, dtype=float)
        uses_orientation = target_rotation is not None
        closest = None

        for starting_angles in self._make_starting_guesses(current_angles,
                                                           number_of_random_restarts):
            angles, position_error, orientation_error = self._solve_from_starting_angles(
                target_position, target_rotation, starting_angles)
            if self._is_acceptable(position_error, orientation_error, uses_orientation):
                return angles, position_error, orientation_error, True
            if closest is None or (position_error + orientation_error
                                   < closest[1] + closest[2]):
                closest = (angles, position_error, orientation_error)

        return closest[0], closest[1], closest[2], False

    def jacobian(self, joint_angles):
        """Return the 6 x N Jacobian at the given angles: position rows, then rotation rows.

        Row order matches the end-effector velocity [vx vy vz wx wy wz]
        (linear in m/s, angular in rad/s, world frame).
        """
        self.forward_kinematics(joint_angles)
        position_jacobian, rotation_jacobian = self._compute_jacobians()
        return self.numpy.vstack([position_jacobian, rotation_jacobian])

    def smallest_singular_value(self, joint_angles):
        """How far the arm is from a singular pose (0 means singular)."""
        return float(self.numpy.linalg.svd(self.jacobian(joint_angles), compute_uv=False)[-1])

    def _solve_with_singularity_damping(self, jacobian, right_hand_side):
        """Solve J x = b: exact away from singular poses, damped close to them.

        Formula (SVD, J = U S V^T):  x = V diag(s / (s^2 + lambda^2)) U^T b
        lambda is 0 while the smallest singular value is above SINGULARITY_THRESHOLD,
        then grows smoothly to MAX_SINGULARITY_DAMPING, so x stays finite.
        """
        u, singular_values, v_transposed = self.numpy.linalg.svd(jacobian, full_matrices=False)
        smallest = singular_values[-1]
        damping_squared = 0.0
        if smallest < SINGULARITY_THRESHOLD:
            damping_squared = ((1.0 - (smallest / SINGULARITY_THRESHOLD) ** 2)
                               * MAX_SINGULARITY_DAMPING ** 2)
        filtered = singular_values / (singular_values ** 2 + damping_squared)
        return v_transposed.T @ (filtered * (u.T @ right_hand_side))

    def forward_velocity(self, joint_angles, joint_velocities):
        """Forward velocity kinematics: end-effector [v; w] = J(q) q'."""
        joint_velocities = self.numpy.asarray(joint_velocities, dtype=float)
        return self.jacobian(joint_angles) @ joint_velocities

    def inverse_velocity(self, joint_angles, end_effector_velocity):
        """Inverse velocity kinematics: joint velocities q' = J(q)^-1 [v; w]."""
        end_effector_velocity = self.numpy.asarray(end_effector_velocity, dtype=float)
        return self._solve_with_singularity_damping(self.jacobian(joint_angles),
                                                    end_effector_velocity)

    def jacobian_derivative(self, joint_angles, joint_velocities, step=1e-6):
        """Time derivative of the Jacobian, J' = dJ/dt, while moving with q'.

        How: J changes only through q, so dJ/dt = (dJ/dq) q'. A central difference
        along the direction of motion gives it:  J' = [J(q + h q') - J(q - h q')] / 2h
        """
        joint_angles = self.numpy.asarray(joint_angles, dtype=float)
        joint_velocities = self.numpy.asarray(joint_velocities, dtype=float)
        ahead = self.jacobian(joint_angles + step * joint_velocities)
        behind = self.jacobian(joint_angles - step * joint_velocities)
        return (ahead - behind) / (2.0 * step)

    def forward_acceleration(self, joint_angles, joint_velocities, joint_accelerations):
        """Forward acceleration kinematics: [a; alpha] = J q'' + J' q'.

        Comes from differentiating [v; w] = J q' with the product rule.
        """
        joint_velocities = self.numpy.asarray(joint_velocities, dtype=float)
        joint_accelerations = self.numpy.asarray(joint_accelerations, dtype=float)
        return (self.jacobian(joint_angles) @ joint_accelerations
                + self.jacobian_derivative(joint_angles, joint_velocities) @ joint_velocities)

    def inverse_acceleration(self, joint_angles, joint_velocities, end_effector_acceleration):
        """Inverse acceleration kinematics: q'' = J^-1 ([a; alpha] - J' q')."""
        joint_velocities = self.numpy.asarray(joint_velocities, dtype=float)
        end_effector_acceleration = self.numpy.asarray(end_effector_acceleration, dtype=float)
        right_hand_side = (end_effector_acceleration
                           - self.jacobian_derivative(joint_angles, joint_velocities)
                           @ joint_velocities)
        return self._solve_with_singularity_damping(self.jacobian(joint_angles),
                                                    right_hand_side)


# ===========================================================================
# ROS2 node: target angles -> smooth commands
# ===========================================================================
class JointController(Node):
    """Turns target joint angles into smooth commands published to the simulator."""

    def __init__(self):
        """Create the node: declare settings, set up state, connect to ROS topics."""
        super().__init__("joint_slider_gui")
        self._declare_parameters()
        self._read_parameters()
        self._create_state_variables()
        self._create_ros_connections()

    # ----- setup helpers ---------------------------------------------------
    def _declare_parameters(self):
        """Declare every ROS parameter (with defaults) so they can be set at launch."""
        self.declare_parameter("command_topic", "/joint_commands")
        self.declare_parameter("state_topic", "/joint_state")
        self.declare_parameter("mode", "velocity")       # velocity | position
        self.declare_parameter("rate", 100.0)            # command rate in Hz
        self.declare_parameter("velocity_gain", 6.0)     # velocity mode: speed per radian of error
        self.declare_parameter("max_acceleration", 20.0) # velocity mode: rad/s^2
        self.declare_parameter("max_speed", 5.0)         # rad/s (also adjustable in the GUI)
        self.declare_parameter("model_path", "")         # scene.xml for limits + kinematics
        self.declare_parameter("ee_name", "")            # end-effector site/body name
        # adds command / simulation_frequency to the joint target for every message).
        self.declare_parameter("simulation_frequency", 1000.0)

    def _read_parameters(self):
        """Copy the parameter values into plain attributes for quick access."""
        get = self.get_parameter
        self.control_mode = str(get("mode").value).lower()
        self.command_rate_hz = float(get("rate").value)
        self.velocity_gain = float(get("velocity_gain").value)
        self.max_acceleration = float(get("max_acceleration").value)
        self.max_speed = float(get("max_speed").value)
        self.state_topic = str(get("state_topic").value)
        self.command_topic = str(get("command_topic").value)
        self.simulation_frequency = float(get("simulation_frequency").value)

    def _create_state_variables(self):
        """Create the shared data. The lock protects it because ROS and the GUI
        run in different threads."""
        self.lock = threading.Lock()
        self.current_angles = None                    # filled by the first joint state
        self.target_angles = [0.0] * NUMBER_OF_JOINTS
        self.position_setpoint = [0.0] * NUMBER_OF_JOINTS   # position mode: gliding setpoint
        self.velocity_command = [0.0] * NUMBER_OF_JOINTS    # velocity mode: ramped command
        self.has_received_state = False

        self.joint_velocity_setpoint = [0.0] * NUMBER_OF_JOINTS
        self.joint_acceleration_setpoint = [0.0] * NUMBER_OF_JOINTS
        self.kinematics = None          # own ArmKinematics copy, attached in main()
        self.kinematics_error_logged = False
        self.latest_end_effector_motion = None   # (velocity, acceleration) for the GUI readouts
        self.tip_velocity_command = None         # [vx vy vz wx wy wz] while moving the tip
        self.tip_motion_time_left = 0.0

    def _create_ros_connections(self):
        """Create the publisher, the joint-state subscriber, and the command timer."""
        self.command_publisher = self.create_publisher(
            Float64MultiArray, self.command_topic, 10)
        self.create_subscription(JointState, self.state_topic, self._on_joint_state, 10)
        self.create_timer(1.0 / self.command_rate_hz, self._send_commands)
        self.end_effector_velocity_publisher = self.create_publisher(
            Float64MultiArray, END_EFFECTOR_VELOCITY_TOPIC, 10)
        self.end_effector_acceleration_publisher = self.create_publisher(
            Float64MultiArray, END_EFFECTOR_ACCELERATION_TOPIC, 10)
        self.get_logger().info(
            f"mode={self.control_mode}, listening on {self.state_topic}, "
            f"publishing on {self.command_topic}")

    # ----- incoming data -----------------------------------------------------
    def _on_joint_state(self, message):
        """Store the arm's latest joint angles (first 6 positions of the message).

        On the very first message the targets are set to the arm's pose, so the
        arm doesn't jump when the program starts.
        """
        if len(message.position) < NUMBER_OF_JOINTS:
            return
        with self.lock:
            self.current_angles = list(message.position[:NUMBER_OF_JOINTS])
            if not self.has_received_state:
                self.target_angles = list(self.current_angles)
                self.position_setpoint = list(self.current_angles)
                self.has_received_state = True

    # ----- called by the GUI ---------------------------------------------------
    def get_current_angles(self):
        """Return a copy of the arm's latest joint angles, or None if none yet."""
        with self.lock:
            return list(self.current_angles) if self.current_angles else None

    def resync_to_current_pose(self):
        """Make the targets equal the arm's actual pose (so it stops/holds there).

        Returns the pose, or None if no joint state has arrived yet.
        """
        with self.lock:
            if self.current_angles is None:
                return None
            self.target_angles = list(self.current_angles)
            self.position_setpoint = list(self.current_angles)
            self.velocity_command = [0.0] * NUMBER_OF_JOINTS
            self.joint_velocity_setpoint = [0.0] * NUMBER_OF_JOINTS
            self.joint_acceleration_setpoint = [0.0] * NUMBER_OF_JOINTS
            self.tip_velocity_command = None
            return list(self.current_angles)

    def set_target_angle(self, joint_index, angle):
        """Set the target angle (radians) for one joint."""
        with self.lock:
            self.tip_velocity_command = None   # a joint command ends any tip-velocity motion
            self.target_angles[joint_index] = float(angle)

    def set_max_speed(self, speed):
        """Set the maximum joint speed (rad/s) used when moving to targets."""
        with self.lock:
            self.max_speed = float(speed)

    def set_max_acceleration(self, acceleration):
        """Set the maximum joint acceleration (rad/s^2) used to speed up and slow down."""
        with self.lock:
            self.max_acceleration = float(acceleration)

    def get_end_effector_motion(self):
        """Latest (velocity, acceleration) of the end effector, or None if not computed yet."""
        with self.lock:
            return self.latest_end_effector_motion

    def get_target_angles(self):
        """Copy of the current target joint angles."""
        with self.lock:
            return list(self.target_angles)

    def start_tip_velocity(self, tip_velocity, duration):
        """Move the end effector with [vx vy vz wx wy wz] for `duration` seconds.

        Uses inverse velocity kinematics every tick. The joints still speed up
        and slow down at max_acceleration. Returns False if it can't start.
        """
        with self.lock:
            if not self.has_received_state or self.kinematics is None:
                return False
            self.tip_velocity_command = self.kinematics.numpy.array(tip_velocity, dtype=float)
            self.tip_motion_time_left = float(duration)
            return True

    def stop_tip_velocity(self):
        """Brake the tip-velocity motion to a stop (at max_acceleration)."""
        with self.lock:
            self.tip_motion_time_left = 0.0

    def is_tip_moving(self):
        with self.lock:
            return self.tip_velocity_command is not None

    # ----- outgoing commands ---------------------------------------------------
    def _send_commands(self):
        """Timer callback: compute the next command for every joint and publish it.

        Does nothing until the first joint state has arrived.
        """
        time_step = 1.0 / self.command_rate_hz
        commanded_motion = None
        with self.lock:
            if not self.has_received_state:
                return
            if self.control_mode == "position":
                if USE_VELOCITY_ACCELERATION_COMMANDS:
                    commands = self._next_commands_from_velocity_and_acceleration(time_step)
                    commanded_motion = (list(self.position_setpoint),
                                        list(self.joint_velocity_setpoint),
                                        list(self.joint_acceleration_setpoint))
                else:
                    commands = self._next_position_commands(time_step)
            else:
                if USE_VELOCITY_ACCELERATION_COMMANDS:
                    commands = self._next_velocity_mode_commands(time_step)
                    commanded_motion = (list(self.position_setpoint),
                                        list(self.joint_velocity_setpoint),
                                        list(self.joint_acceleration_setpoint))
                else:
                    commands = self._next_velocity_commands(time_step)

        message = Float64MultiArray()
        message.data = commands
        self.command_publisher.publish(message)
        if commanded_motion is not None:
            self._publish_end_effector_motion(*commanded_motion)

    def _next_commands_from_velocity_and_acceleration(self, time_step):
        if self.tip_velocity_command is not None and self.kinematics is not None:
            return self._next_tip_velocity_angles(time_step)
        commands = []
        largest_speed_change = self.max_acceleration * time_step
        for joint in range(NUMBER_OF_JOINTS):
            target = self.target_angles[joint]
            angle = self.position_setpoint[joint]
            speed = self.joint_velocity_setpoint[joint]
            remaining = target - angle

            # Braking one speed step (max_acceleration * dt) per tick from speed n steps
            # covers n(n+1)/2 * max_acceleration * dt^2, so the largest speed that can
            # still stop within `remaining` is:
            stopping_speed = largest_speed_change * (
                -0.5 + math.sqrt(0.25 + 2.0 * abs(remaining)
                                 / (self.max_acceleration * time_step ** 2)))
            wanted_speed = math.copysign(min(self.max_speed, stopping_speed), remaining)
            new_speed = speed + clamp_value(wanted_speed - speed,
                                            -largest_speed_change, largest_speed_change)
            new_angle = angle + new_speed * time_step

            # Reached (or about to pass) the target and slow enough to stop this tick:
            # finish exactly on it. If it is too fast to stop, it overshoots slightly
            # and comes back, so the acceleration limit is never broken.
            reaches_target = (target - new_angle) * remaining <= 0.0
            if reaches_target and abs(speed) <= largest_speed_change:
                new_angle = target
                new_speed = 0.0

            self.joint_acceleration_setpoint[joint] = (new_speed - speed) / time_step
            self.joint_velocity_setpoint[joint] = new_speed
            self.position_setpoint[joint] = new_angle
            commands.append(new_angle)
        return commands

    def _next_tip_velocity_angles(self, time_step):
        numpy = self.kinematics.numpy
        angles = numpy.array(self.position_setpoint)
        speeds = numpy.array(self.joint_velocity_setpoint)
        still_moving = self.tip_motion_time_left > 0.0

        wanted_speeds = numpy.zeros(NUMBER_OF_JOINTS)
        if still_moving:
            wanted_speeds = self.kinematics.inverse_velocity(angles, self.tip_velocity_command)
            fastest = numpy.max(numpy.abs(wanted_speeds))
            if fastest > self.max_speed:
                wanted_speeds = wanted_speeds * (self.max_speed / fastest)

        largest_speed_change = self.max_acceleration * time_step
        new_speeds = speeds + numpy.clip(wanted_speeds - speeds,
                                         -largest_speed_change, largest_speed_change)
        new_angles = angles + new_speeds * time_step
        at_limit = ((new_angles < self.kinematics.lowest_angles)
                    | (new_angles > self.kinematics.highest_angles))
        new_angles = numpy.clip(new_angles, self.kinematics.lowest_angles,
                                self.kinematics.highest_angles)
        new_speeds[at_limit] = 0.0                       # a joint at its limit stops there

        self.joint_acceleration_setpoint = list((new_speeds - speeds) / time_step)
        self.joint_velocity_setpoint = list(new_speeds)
        self.position_setpoint = list(new_angles)
        self.target_angles = list(new_angles)            # hold here when the motion ends
        self.tip_motion_time_left -= time_step
        if not still_moving and not numpy.any(new_speeds):
            self.tip_velocity_command = None             # fully stopped
        return list(new_angles)

    def _next_velocity_mode_commands(self, time_step):
        previous_angles = list(self.position_setpoint)
        new_angles = self._next_commands_from_velocity_and_acceleration(time_step)
        return [(new - old) * self.simulation_frequency
                for new, old in zip(new_angles, previous_angles)]

    def _publish_end_effector_motion(self, joint_angles, joint_velocities, joint_accelerations):
        """Publish the end effector's velocity and acceleration for the commanded motion.

        Uses forward velocity kinematics ([v; w] = J q') and forward acceleration
        kinematics ([a; alpha] = J q'' + J' q'). Skipped if the model isn't loaded.
        """
        if self.kinematics is None:
            return
        try:
            velocity = self.kinematics.forward_velocity(joint_angles, joint_velocities)
            acceleration = self.kinematics.forward_acceleration(
                joint_angles, joint_velocities, joint_accelerations)
        except Exception as error:          # never let this stop the command timer
            if not self.kinematics_error_logged:
                self.get_logger().error(f"End-effector velocity/acceleration failed: {error}")
                self.kinematics_error_logged = True
            return
        with self.lock:
            self.latest_end_effector_motion = (list(velocity), list(acceleration))
        for publisher, values in ((self.end_effector_velocity_publisher, velocity),
                                  (self.end_effector_acceleration_publisher, acceleration)):
            message = Float64MultiArray()
            message.data = [float(value) for value in values]
            publisher.publish(message)

    def _next_position_commands(self, time_step):
        """Position mode: move each setpoint toward its target, at most max_speed.

        How: each tick, the setpoint advances by the remaining distance, capped
        at max_speed * time_step, so the arm glides instead of snapping.
        Call with the lock held.
        """
        commands = []
        largest_step = self.max_speed * time_step
        for joint in range(NUMBER_OF_JOINTS):
            remaining = self.target_angles[joint] - self.position_setpoint[joint]
            self.position_setpoint[joint] += clamp_value(remaining, -largest_step, largest_step)
            commands.append(self.position_setpoint[joint])
        return commands

    def _next_velocity_commands(self, time_step):
        """Velocity mode: speed toward each target, ramped by an acceleration limit.

        How: wanted speed = gain * angle error, capped at max_speed (and zero
        when within 0.002 rad). The command then moves toward the wanted speed
        by at most max_acceleration * time_step per tick, so it ramps smoothly.
        Call with the lock held.
        """
        commands = []
        largest_speed_change = self.max_acceleration * time_step
        for joint in range(NUMBER_OF_JOINTS):
            angle_error = self.target_angles[joint] - self.current_angles[joint]
            wanted_speed = clamp_value(self.velocity_gain * angle_error,
                                       -self.max_speed, self.max_speed)
            if abs(angle_error) < 0.002:
                wanted_speed = 0.0
            speed_change = wanted_speed - self.velocity_command[joint]
            self.velocity_command[joint] += clamp_value(
                speed_change, -largest_speed_change, largest_speed_change)
            commands.append(self.velocity_command[joint])
        return commands


# ===========================================================================
# GUI window
# ===========================================================================
class ControllerWindow:
    """The Tkinter window: joint sliders, speed slider, and the kinematics panel."""

    def __init__(self, root, controller, kinematics, joint_limits):
        """Build every part of the window, then start the periodic refresh."""
        self.root = root
        self.controller = controller
        self.kinematics = kinematics          # None when no model was loaded
        self.joint_limits = joint_limits
        self.is_synced_to_arm = False
        self.refresh_count = 0
        self.tip_was_moving = False

        self.slider_variables = []
        self.sliders = []
        self.target_labels = []
        self.actual_labels = []

        self._build_main_frame()
        self._build_status_label()
        self._build_joint_rows()
        self._build_speed_row()
        self._build_button_row()
        self._build_kinematics_panel()
        self._build_motion_panel()
        self._show_kinematics_availability()
        self._refresh()

    # ----- window construction -----------------------------------------------
    def _build_main_frame(self):
        """Set the window title/size and create the padded frame everything sits in."""
        self.root.title("Lite6 Controller")
        self.root.minsize(640, 0)
        self.frame = ttk.Frame(self.root, padding=12)
        self.frame.pack(fill="both", expand=True)
        self.frame.columnconfigure(1, weight=1)   # let the sliders stretch

    def _build_status_label(self):
        """Top line showing whether joint states are being received."""
        self.status_label = ttk.Label(
            self.frame, text="Waiting for joint state...", foreground=COLOR_WARNING)
        self.status_label.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

    def _build_joint_rows(self):
        """One row per joint: name, slider, target-angle label, actual-angle label.

        Sliders start disabled; they unlock once the arm's real pose is known.
        """
        for joint, (name, lowest, highest) in enumerate(self.joint_limits):
            row = joint + 1
            ttk.Label(self.frame, text=name, width=8).grid(
                row=row, column=0, sticky="w", pady=5)

            variable = tk.DoubleVar(value=clamp_value(0.0, lowest, highest))
            slider = ttk.Scale(
                self.frame, from_=lowest, to=highest, variable=variable,
                orient="horizontal",
                command=lambda value, index=joint: self._on_slider_moved(index, value))
            slider.grid(row=row, column=1, sticky="ew", padx=8)
            slider.state(["disabled"])

            target_label = ttk.Label(self.frame, text="target    0.0°", width=14, anchor="e")
            target_label.grid(row=row, column=2)
            actual_label = ttk.Label(self.frame, text="actual    0.0°", width=14, anchor="e")
            actual_label.grid(row=row, column=3)

            self.slider_variables.append(variable)
            self.sliders.append(slider)
            self.target_labels.append(target_label)
            self.actual_labels.append(actual_label)

    def _build_speed_row(self):
        """Separator line plus the max-speed slider (0.05 to 10 rad/s)."""
        row = NUMBER_OF_JOINTS + 1
        ttk.Separator(self.frame).grid(
            row=row, column=0, columnspan=4, sticky="ew", pady=10)
        ttk.Label(self.frame, text="Max speed").grid(row=row + 1, column=0, sticky="w")

        self.speed_variable = tk.DoubleVar(value=self.controller.max_speed)
        ttk.Scale(self.frame, from_=0.05, to=10.0, variable=self.speed_variable,
                  orient="horizontal", command=self._on_speed_changed
                  ).grid(row=row + 1, column=1, sticky="ew", padx=8)
        self.speed_label = ttk.Label(
            self.frame, text=f"{self.controller.max_speed:.2f} rad/s", width=14, anchor="e")
        self.speed_label.grid(row=row + 1, column=2)

    def _build_button_row(self):
        """The 'Sync to arm' and 'Go to zero' buttons."""
        row = NUMBER_OF_JOINTS + 3
        button_frame = ttk.Frame(self.frame)
        button_frame.grid(row=row, column=0, columnspan=4, pady=(12, 0))
        ttk.Button(button_frame, text="Sync to arm",
                   command=self._sync_sliders_to_arm).pack(side="left", padx=4)
        ttk.Button(button_frame, text="Go to zero",
                   command=self._go_to_zero).pack(side="left", padx=4)

    def _build_kinematics_panel(self):
        """The end-effector box: live position, X/Y/Z inputs, and the Go buttons."""
        row = NUMBER_OF_JOINTS + 4
        box = ttk.LabelFrame(self.frame, text="End effector (kinematics)", padding=10)
        box.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(14, 0))

        self.position_label = ttk.Label(
            box, text="Current position:  --", font=("TkDefaultFont", 10, "bold"))
        self.position_label.grid(row=0, column=0, columnspan=8, sticky="w", pady=(0, 8))

        self.position_entries = {}
        for column, axis in enumerate("XYZ"):
            ttk.Label(box, text=f"{axis} (m)").grid(row=1, column=column * 2, padx=(0, 4))
            entry = ttk.Entry(box, width=9)
            entry.insert(0, "0.000")
            entry.grid(row=1, column=column * 2 + 1, padx=(0, 10))
            self.position_entries[axis] = entry

        self.keep_orientation = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Keep orientation", variable=self.keep_orientation
                        ).grid(row=1, column=6, padx=6)
        self.go_button = ttk.Button(box, text="Go to point", command=self._go_to_point)
        self.go_button.grid(row=1, column=7, padx=4)
        self.use_current_button = ttk.Button(
            box, text="Use current", command=self._fill_entries_from_arm)
        self.use_current_button.grid(row=2, column=7, padx=4, pady=(6, 0))

        self.message_label = ttk.Label(box, text="", wraplength=560, justify="left")
        self.message_label.grid(row=2, column=0, columnspan=7, sticky="w", pady=(6, 0))

    def _show_kinematics_availability(self):
        """Disable the kinematics buttons (and explain why) if no model was loaded;
        otherwise tell the user which end effector is being controlled."""
        if self.kinematics is None:
            self.go_button.state(["disabled"])
            self.use_current_button.state(["disabled"])
            self._show_message(
                "Kinematics unavailable: pass -p model_path:=<scene.xml> "
                "and make sure `pip install mujoco numpy` is done.", COLOR_ERROR)
        else:
            self._show_message(
                f"End effector = {self.kinematics.end_effector_description}", COLOR_OK)

    # ----- slider / button callbacks ---------------------------------------------
    def _on_slider_moved(self, joint_index, slider_value):
        """A joint slider moved: update its label and send the new target to the node."""
        angle = float(slider_value)
        self.target_labels[joint_index].config(text=f"target {math.degrees(angle):6.1f}°")
        self.controller.set_target_angle(joint_index, angle)

    def _on_speed_changed(self, slider_value):
        """The max-speed slider moved: update its label and tell the node."""
        speed = float(slider_value)
        self.speed_label.config(text=f"{speed:.2f} rad/s")
        self.controller.set_max_speed(speed)

    def _go_to_zero(self):
        """Set every joint slider (and so every target) to 0, within each joint's limits."""
        for joint, (_, lowest, highest) in enumerate(self.joint_limits):
            zero = clamp_value(0.0, lowest, highest)
            self.slider_variables[joint].set(zero)
            self._on_slider_moved(joint, zero)

    def _sync_sliders_to_arm(self):
        """Move the sliders to where the arm really is, and hold the arm there.

        How: ask the node to reset its targets to the current pose, then copy
        that pose onto the sliders and labels. The arm itself doesn't move.
        """
        current_angles = self.controller.resync_to_current_pose()
        if current_angles is None:
            return
        for joint, (_, lowest, highest) in enumerate(self.joint_limits):
            self.slider_variables[joint].set(clamp_value(current_angles[joint], lowest, highest))
            self.target_labels[joint].config(
                text=f"target {math.degrees(current_angles[joint]):6.1f}°")

    # ----- kinematics panel actions -------------------------------------------------
    def _fill_entries_from_arm(self):
        """Put the arm's current end-effector X, Y, Z into the three input boxes.

        How: run forward kinematics on the latest joint angles.
        """
        current_angles = self.controller.get_current_angles()
        if current_angles is None or self.kinematics is None:
            return
        position, _ = self.kinematics.forward_kinematics(current_angles)
        for axis, value in zip("XYZ", position):
            self.position_entries[axis].delete(0, "end")
            self.position_entries[axis].insert(0, f"{value:.3f}")

    def _read_target_position(self):
        """Read X, Y, Z from the input boxes as floats; None (and an error) if invalid."""
        try:
            return [float(self.position_entries[axis].get()) for axis in "XYZ"]
        except ValueError:
            self._show_message("X, Y, Z must be numbers (metres).", COLOR_ERROR)
            return None

    def _solve_inverse_kinematics(self, target_position, current_angles):
        """Run IK for the target, honouring the 'Keep orientation' checkbox.

        How: if ticked, first try to also hold the current orientation; if that
        fails, retry with position only and add a note. Returns
        (angles, position_error, reached, note).
        """
        note = ""
        if self.keep_orientation.get():
            _, current_rotation = self.kinematics.forward_kinematics(current_angles)
            angles, position_error, _, reached = self.kinematics.inverse_kinematics(
                target_position, current_angles, current_rotation)
            if reached:
                return angles, position_error, True, note
            note = " (orientation could not be held, so it will change)"
        angles, position_error, _, reached = self.kinematics.inverse_kinematics(
            target_position, current_angles, None)
        return angles, position_error, reached, note

    def _apply_angles_to_sliders(self, joint_angles):
        """Set every slider (and so every target) to the IK solution, within limits.

        The node then drives the arm there smoothly.
        """
        for joint, (_, lowest, highest) in enumerate(self.joint_limits):
            angle = clamp_value(float(joint_angles[joint]), lowest, highest)
            self.slider_variables[joint].set(angle)
            self._on_slider_moved(joint, angle)

    def _go_to_point(self):
        """'Go to point' button: read the target, solve IK, and move the arm.

        If the point is unreachable the arm stays put and the message shows
        how close the solver got.
        """
        current_angles = self.controller.get_current_angles()
        if current_angles is None:
            self._show_message("No joint state yet.", COLOR_ERROR)
            return
        target_position = self._read_target_position()
        if target_position is None:
            return

        angles, position_error, reached, note = self._solve_inverse_kinematics(
            target_position, current_angles)
        if not reached:
            self._show_message(
                f"Unreachable: closest solution is {position_error * 1000:.1f} mm away. "
                f"Not moving.", COLOR_ERROR)
            return

        self._apply_angles_to_sliders(angles)
        angle_text = ", ".join(f"{math.degrees(a):.0f}°" for a in angles)
        self._show_message(
            f"IK solved (error {position_error * 1000:.2f} mm){note}. Joints: {angle_text}",
            COLOR_OK)

    def _show_message(self, text, color):
        """Show a coloured message line under the kinematics inputs."""
        self.message_label.config(text=text, foreground=color)

    # ----- periodic refresh (runs every 50 ms) -----------------------------------------
    def _refresh(self):
        """Update the display from the latest joint state, then schedule the next update.

        Handles three cases: first data arriving, normal updates, and a
        warning if no data shows up after ~3 seconds.
        """
        self.refresh_count += 1
        current_angles = self.controller.get_current_angles()
        if current_angles:
            if not self.is_synced_to_arm:
                self._handle_first_joint_state()
            self._update_actual_angle_labels(current_angles)
            self._update_current_position_label(current_angles)
            self._update_motion_panel(current_angles)
        elif self.refresh_count > 60:
            self._show_no_data_warning()
        self.root.after(50, self._refresh)

    def _handle_first_joint_state(self):
        """First data arrived: put the sliders on the arm's pose, unlock them,
        mark the status green and prefill the X/Y/Z boxes."""
        self._sync_sliders_to_arm()
        for slider in self.sliders:
            slider.state(["!disabled"])
        self.is_synced_to_arm = True
        self.status_label.config(
            text=f"Connected to {self.controller.state_topic} ({self.controller.control_mode} mode)",
            foreground=COLOR_OK)
        self._fill_entries_from_arm()

    def _update_actual_angle_labels(self, current_angles):
        """Show each joint's real angle (degrees) next to its slider."""
        for joint, angle in enumerate(current_angles):
            self.actual_labels[joint].config(text=f"actual {math.degrees(angle):6.1f}°")

    def _update_current_position_label(self, current_angles):
        """Show the live end-effector X, Y, Z, computed with forward kinematics."""
        if self.kinematics is None:
            return
        position, _ = self.kinematics.forward_kinematics(current_angles)
        self.position_label.config(
            text=f"Current position:  X {position[0]:.3f}   "
                 f"Y {position[1]:.3f}   Z {position[2]:.3f}  m")

    TIP_VELOCITY_NAMES = ["vx", "vy", "vz", "wx", "wy", "wz"]

    def _build_motion_panel(self):
        """Box under the kinematics panel: max acceleration, live end-effector velocity /
        acceleration, and a tip-velocity input (inverse velocity kinematics)."""
        row = NUMBER_OF_JOINTS + 5
        box = ttk.LabelFrame(self.frame, padding=10,
                             text="End effector velocity & acceleration")
        box.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(14, 0))
        box.columnconfigure(0, weight=1)

        # --- max acceleration slider (0.5 to 50 rad/s^2) ---
        acceleration_row = ttk.Frame(box)
        acceleration_row.grid(row=0, column=0, sticky="ew")
        acceleration_row.columnconfigure(1, weight=1)
        ttk.Label(acceleration_row, text="Max acceleration").grid(row=0, column=0, sticky="w")
        self.acceleration_variable = tk.DoubleVar(value=self.controller.max_acceleration)
        ttk.Scale(acceleration_row, from_=0.5, to=50.0, variable=self.acceleration_variable,
                  orient="horizontal", command=self._on_acceleration_changed
                  ).grid(row=0, column=1, sticky="ew", padx=8)
        self.acceleration_value_label = ttk.Label(
            acceleration_row, text=f"{self.controller.max_acceleration:.1f} rad/s²",
            width=14, anchor="e")
        self.acceleration_value_label.grid(row=0, column=2)

        # --- live readouts (forward velocity / acceleration kinematics) ---
        mono = ("TkFixedFont", 9)
        self.velocity_readout = ttk.Label(box, text="Velocity      --", font=mono)
        self.velocity_readout.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.acceleration_readout = ttk.Label(box, text="Acceleration  --", font=mono)
        self.acceleration_readout.grid(row=2, column=0, sticky="w")
        self.singularity_readout = ttk.Label(box, text="Singularity   --", font=mono)
        self.singularity_readout.grid(row=3, column=0, sticky="w")

        # --- tip velocity input (inverse velocity kinematics) ---
        input_row = ttk.Frame(box)
        input_row.grid(row=4, column=0, sticky="w", pady=(10, 0))
        for column, name in enumerate(self.TIP_VELOCITY_NAMES):
            ttk.Label(input_row, text=name).grid(row=0, column=column + 1)
        ttk.Label(input_row, text="Tip velocity\n(m/s, rad/s)").grid(row=1, column=0, sticky="w")
        self.tip_velocity_entries = []
        for column in range(6):
            entry = ttk.Entry(input_row, width=7)
            entry.insert(0, "0.0")
            entry.grid(row=1, column=column + 1, padx=2)
            self.tip_velocity_entries.append(entry)

        button_row = ttk.Frame(box)
        button_row.grid(row=5, column=0, sticky="w", pady=(6, 0))
        ttk.Label(button_row, text="Duration (s)").pack(side="left")
        self.duration_entry = ttk.Entry(button_row, width=6)
        self.duration_entry.insert(0, "2.0")
        self.duration_entry.pack(side="left", padx=(4, 10))
        self.move_tip_button = ttk.Button(button_row, text="Move tip",
                                          command=self._start_tip_velocity)
        self.move_tip_button.pack(side="left", padx=4)
        self.stop_tip_button = ttk.Button(button_row, text="Stop",
                                          command=self._stop_tip_velocity)
        self.stop_tip_button.pack(side="left", padx=4)

        self.motion_message_label = ttk.Label(box, text="", wraplength=560, justify="left")
        self.motion_message_label.grid(row=6, column=0, sticky="w", pady=(6, 0))
        if self.kinematics is None:
            self.move_tip_button.state(["disabled"])
            self.stop_tip_button.state(["disabled"])

    def _on_acceleration_changed(self, slider_value):
        """The max-acceleration slider moved: update its label and tell the node."""
        acceleration = float(slider_value)
        self.acceleration_value_label.config(text=f"{acceleration:.1f} rad/s²")
        self.controller.set_max_acceleration(acceleration)

    @staticmethod
    def _format_six(values, linear_unit, angular_unit):
        return (f"[{values[0]:+7.3f} {values[1]:+7.3f} {values[2]:+7.3f}] {linear_unit}   "
                f"[{values[3]:+7.3f} {values[4]:+7.3f} {values[5]:+7.3f}] {angular_unit}")

    def _update_motion_panel(self, current_angles):
        """Refresh the readouts; keep the sliders on the arm while the tip is moving."""
        if self.kinematics is None:
            return
        motion = self.controller.get_end_effector_motion()
        if motion is not None:
            velocity, acceleration = motion
            self.velocity_readout.config(
                text="Velocity      " + self._format_six(velocity, "m/s", "rad/s"))
            self.acceleration_readout.config(
                text="Acceleration  " + self._format_six(acceleration, "m/s²", "rad/s²"))
        smallest = self.kinematics.smallest_singular_value(current_angles)
        near = smallest < SINGULARITY_THRESHOLD
        self.singularity_readout.config(
            text=f"Singularity   smallest singular value {smallest:.4f}"
                 + ("   NEAR SINGULARITY" if near else ""),
            foreground=COLOR_ERROR if near else "")

        tip_moving = self.controller.is_tip_moving()
        if tip_moving or self.tip_was_moving:
            self._show_targets_on_sliders(self.controller.get_target_angles())
        if self.tip_was_moving and not tip_moving:
            self._show_motion_message("Tip motion finished.", COLOR_OK)
        self.tip_was_moving = tip_moving

    def _show_targets_on_sliders(self, angles):
        """Move the sliders and target labels to `angles` without sending new targets."""
        for joint, (_, lowest, highest) in enumerate(self.joint_limits):
            self.slider_variables[joint].set(clamp_value(angles[joint], lowest, highest))
            self.target_labels[joint].config(text=f"target {math.degrees(angles[joint]):6.1f}°")

    def _show_motion_message(self, text, color):
        self.motion_message_label.config(text=text, foreground=color)

    def _start_tip_velocity(self):
        """'Move tip' button: read [v; w] and the duration, then start the motion."""
        try:
            tip_velocity = [float(entry.get()) for entry in self.tip_velocity_entries]
            duration = float(self.duration_entry.get())
        except ValueError:
            self._show_motion_message("Tip velocity and duration must be numbers.", COLOR_ERROR)
            return
        if duration <= 0:
            self._show_motion_message("Duration must be more than 0 s.", COLOR_ERROR)
            return
        current_angles = self.controller.get_current_angles()
        if current_angles is None or not self.controller.start_tip_velocity(tip_velocity,
                                                                            duration):
            self._show_motion_message("No joint state yet.", COLOR_ERROR)
            return
        joint_speeds = self.kinematics.inverse_velocity(current_angles, tip_velocity)
        self._show_motion_message(
            "Inverse velocity kinematics: joint velocities = ["
            + ", ".join(f"{speed:+.3f}" for speed in joint_speeds) + "] rad/s", COLOR_OK)

    def _stop_tip_velocity(self):
        """'Stop' button: brake the tip motion to a stop."""
        self.controller.stop_tip_velocity()
        self._show_motion_message("Stopping.", COLOR_WARNING)

    def _show_no_data_warning(self):
        """Turn the status line red and say which topic we're waiting on."""
        self.status_label.config(
            text=f"No data on {self.controller.state_topic} - check `ros2 topic list` "
                 f"and pass -p state_topic:=<name>",
            foreground=COLOR_ERROR)


# ===========================================================================
# Program entry point
# ===========================================================================
def load_kinematics(controller):
    """Try to build ArmKinematics from the model_path parameter.

    Returns (kinematics, joint_limits). If there is no model path, or loading
    fails, returns (None, FALLBACK_JOINT_LIMITS) and logs why, so sliders still work.
    """
    logger = controller.get_logger()
    model_path = str(controller.get_parameter("model_path").value)
    if not model_path:
        logger.warn("No model_path given: IK disabled, fallback joint limits used.")
        return None, FALLBACK_JOINT_LIMITS
    try:
        kinematics = ArmKinematics(
            model_path, NUMBER_OF_JOINTS, str(controller.get_parameter("ee_name").value))
    except Exception as error:
        logger.error(f"Could not load model for kinematics ({error}). "
                     f"Using fallback limits; IK disabled.")
        return None, FALLBACK_JOINT_LIMITS

    limits_text = ", ".join(f"{name} [{math.degrees(low):.0f}, {math.degrees(high):.0f}]"
                            for name, low, high in kinematics.joint_limits)
    logger.info(f"Joint limits from model: {limits_text}")
    logger.info(f"End effector: {kinematics.end_effector_description}")
    return kinematics, kinematics.joint_limits


def main():
    """Start ROS, load kinematics, run ROS in a background thread, show the window.

    Tkinter must run in the main thread, so ROS spins in a daemon thread.
    """
    rclpy.init()
    controller = JointController()
    kinematics, joint_limits = load_kinematics(controller)
    if kinematics is not None:
        controller.kinematics = ArmKinematics(
            str(controller.get_parameter("model_path").value), NUMBER_OF_JOINTS,
            str(controller.get_parameter("ee_name").value))

    threading.Thread(target=rclpy.spin, args=(controller,), daemon=True).start()

    root = tk.Tk()
    ControllerWindow(root, controller, kinematics, joint_limits)
    try:
        root.mainloop()
    finally:
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()