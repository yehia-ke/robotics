"""
The industrial application (Milestone 4): laboratory sample sorting with the Lite6.

Four capped sample vials stand in the pick tray. The robot sorts each vial into the bin
whose colour matches its cap (red / green / blue). Positions come from the scene model
(pre-programmed coordinates, no camera), as proposed in Milestone 1.

Each vial is handled with BOTH trajectory types:
    1. joint-space quintic  -> above the vial (transit height)       [large motion]
    2. open the gripper
    3. task-space line      -> straight down to the grasp height     [precise approach]
    4. close the gripper, wait for the grip
    5. task-space line      -> straight up to transit height          [lift]
    6. joint-space quintic  -> above the matching bin
    7. task-space line      -> down to the release height
    8. open the gripper (the vial drops into the bin)
    9. task-space line      -> back up
and finally a joint-space move back home.

Lifting demo: grasp the Erlenmeyer flask (150 g, ~4x a vial) by its neck on the hot plate,
lift it 5 cm, hold it, put it back.

This module only PLANS (a list of Step objects). Execution (sending the trajectories to the
controller and waiting for them) is done by the GUI (sliders_node.py) or by the offline
validation (validation/offline_validation.py).
"""
from dataclasses import dataclass, field

import numpy as np

import kinematics as kin

HOME = np.array([0.0, 0.0, 1.57, 0.0, 1.57, 0.0])

TRANSIT_HEIGHT = 0.20        # TCP height for moves between tray and bins (clears the tube rack)
VIAL_GRASP_HEIGHT = 0.060    # TCP (fingertips) height when grasping: fingers on the cap
BIN_RELEASE_HEIGHT = 0.13    # TCP height when releasing over a bin (vial bottom above the walls)
FLASK_GRASP_HEIGHT = 0.168   # TCP height on the flask neck
FLASK_LIFT = 0.05

GRIPPER_CLOSE_FORCE = 5.0    # N on the finger motor (+ closes)
GRIPPER_OPEN_FORCE = -5.0
GRIP_SETTLE_TIME = 0.6       # s to wait after closing / opening

VIAL_MASS = 0.034            # kg (glass + liquid + cap), used as payload for gravity compensation
FLASK_MASS = 0.151

COLOURS = ("red", "green", "blue")


@dataclass
class Step:
    """One action of the task.

    kind:
      "joint"   : joint-space trajectory to the pose (position, yaw) [or to `joints`]
      "line"    : task-space straight line to the pose (position, yaw)
      "gripper" : gripper force (N) + payload mass (kg) for gravity compensation, then wait
      "wait"    : hold still for `duration` s
    """
    kind: str
    description: str
    position: np.ndarray = None
    yaw: float = 0.0
    joints: np.ndarray = None
    profile: str = "quintic"
    duration: float = None
    force: float = 0.0
    payload: float = 0.0
    extra: dict = field(default_factory=dict)

    def target_rotation(self):
        return kin.top_down_rotation(self.yaw)


# ============================================================================
# Scene layout
# ============================================================================
@dataclass
class Layout:
    vials: list          # [(name, colour, position xyz of the vial body centre)]
    bins: dict           # colour -> bin centre xyz (bench level)
    flask: np.ndarray    # flask body origin


def load_layout(model_path):
    """Read vial / bin / flask positions and cap colours from the scene model."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "lab_home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    def body_position(name):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            raise ValueError(f"body '{name}' not found in {model_path}")
        return data.xpos[body].copy()

    vials = []
    for index in range(1, 10):
        name = f"lab_sample_vial_{index}"
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            break
        cap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_cap")
        material = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, model.geom_matid[cap]) or ""
        colour = next((c for c in COLOURS if c in material), None)
        vials.append((name, colour, body_position(name)))
    bins = {c: body_position(f"lab_sort_bin_{c}") for c in COLOURS}
    return Layout(vials=vials, bins=bins, flask=body_position("lab_erlenmeyer_flask"))


def radial_yaw(position):
    """Gripper yaw pointing away from the robot: keeps joints 4 and 6 near zero."""
    return float(np.arctan2(position[1], position[0]))


# ============================================================================
# Task plans
# ============================================================================
def pick_and_place_steps(name, pick_xy, grasp_height, place_xy, release_height, payload):
    pick_yaw = radial_yaw(pick_xy)
    place_yaw = radial_yaw(place_xy)
    above_pick = np.array([pick_xy[0], pick_xy[1], TRANSIT_HEIGHT])
    grasp = np.array([pick_xy[0], pick_xy[1], grasp_height])
    above_place = np.array([place_xy[0], place_xy[1], TRANSIT_HEIGHT])
    release = np.array([place_xy[0], place_xy[1], release_height])
    return [
        Step("joint", f"{name}: joint-space move above the vial", above_pick, pick_yaw),
        Step("gripper", f"{name}: open gripper", force=GRIPPER_OPEN_FORCE, duration=0.3),
        Step("line", f"{name}: straight line down to grasp", grasp, pick_yaw),
        Step("gripper", f"{name}: close gripper", force=GRIPPER_CLOSE_FORCE, payload=payload,
             duration=GRIP_SETTLE_TIME),
        Step("line", f"{name}: straight line up (lift)", above_pick, pick_yaw),
        Step("joint", f"{name}: joint-space move above the bin", above_place, place_yaw),
        Step("line", f"{name}: straight line down into the bin", release, place_yaw),
        Step("gripper", f"{name}: release", force=GRIPPER_OPEN_FORCE, payload=0.0,
             duration=GRIP_SETTLE_TIME),
        Step("line", f"{name}: straight line up", above_place, place_yaw),
    ]


BIN_SLOT_SPACING = 0.05      # m between vials released into the same bin (bin inside: 12.6 cm)


def bin_slots(layout):
    """Release point of every vial: bin centre, or side-by-side slots if a bin gets several."""
    per_colour = {}
    for name, colour, _ in layout.vials:
        per_colour.setdefault(colour, []).append(name)
    slots = {}
    for colour, names in per_colour.items():
        if colour not in layout.bins:
            continue
        for k, name in enumerate(names):
            offset = (k - (len(names) - 1) / 2.0) * BIN_SLOT_SPACING
            slots[name] = layout.bins[colour][:2] + np.array([offset, 0.0])
    return slots


def plan_sorting(layout, vial_names=None):
    """Sort the vials (all, or only `vial_names`) into the bins matching their cap colour."""
    steps = []
    slots = bin_slots(layout)
    for name, colour, position in layout.vials:
        if vial_names is not None and name not in vial_names:
            continue
        if name not in slots:
            continue
        label = f"{name.replace('lab_sample_', '')} ({colour})"
        steps += pick_and_place_steps(label, position[:2], VIAL_GRASP_HEIGHT,
                                      slots[name], BIN_RELEASE_HEIGHT, VIAL_MASS)
    steps.append(Step("joint", "return home (joint space)", joints=HOME.copy()))
    return steps


def plan_lift_demo(layout):
    """Lift the Erlenmeyer flask from the hot plate by its neck and put it back."""
    xy = layout.flask[:2]
    yaw = radial_yaw(xy)
    above = np.array([xy[0], xy[1], FLASK_GRASP_HEIGHT + 0.05])
    grasp = np.array([xy[0], xy[1], FLASK_GRASP_HEIGHT])
    lifted = grasp + np.array([0.0, 0.0, FLASK_LIFT])
    return [
        Step("joint", "flask: joint-space move above the neck", above, yaw),
        Step("gripper", "flask: open gripper", force=GRIPPER_OPEN_FORCE, duration=0.3),
        Step("line", "flask: straight line down to the neck", grasp, yaw),
        Step("gripper", "flask: close gripper", force=GRIPPER_CLOSE_FORCE, payload=FLASK_MASS,
             duration=GRIP_SETTLE_TIME),
        Step("line", "flask: lift 5 cm (task-space line)", lifted, yaw, duration=2.0),
        Step("wait", "flask: hold it up", duration=2.0),
        Step("line", "flask: put it back down", grasp, yaw, duration=2.0),
        Step("gripper", "flask: release", force=GRIPPER_OPEN_FORCE, payload=0.0,
             duration=GRIP_SETTLE_TIME),
        Step("line", "flask: straight line up", above, yaw),
        Step("joint", "return home (joint space)", joints=HOME.copy()),
    ]


# ============================================================================
# Turning a motion step into a joint trajectory
# ============================================================================
def solve_pose(position, yaw, q_seed):
    """IK for a top-down TCP pose, branch nearest q_seed. Raises ValueError if unreachable."""
    R = kin.top_down_rotation(yaw)
    seeds = [np.asarray(q_seed, float),
             HOME + np.array([radial_yaw(position), 0, 0, 0, 0, 0]),
             HOME]
    q, ok = kin.ik_pose(position, R, seeds)
    if not ok:
        raise ValueError(f"pose {np.round(position, 3)} (yaw {np.degrees(yaw):.0f} deg) is unreachable")
    # joints 4 and 6 have +/-360 deg range: take the equivalent angle closest to the seed
    for joint in (0, 3, 5):
        q[joint] = _closest_equivalent(q[joint], q_seed[joint], kin.JOINT_LOWER[joint],
                                       kin.JOINT_UPPER[joint])
    return q


def _closest_equivalent(angle, reference, lower, upper):
    candidates = [angle + k * 2 * np.pi for k in (-2, -1, 0, 1, 2)]
    candidates = [c for c in candidates if lower <= c <= upper]
    return min(candidates, key=lambda c: abs(c - reference))


def plan_motion(step, q_start, max_joint_speed=1.5, max_tcp_speed=0.12):
    """Joint trajectory (trajectory.SampledTrajectory) for a 'joint' / 'line' / 'wait' step.

    Speeds are deliberately moderate for a lab with glassware."""
    import trajectory as traj
    q_start = np.asarray(q_start, float)
    if step.kind == "joint":
        goal = step.joints if step.joints is not None else solve_pose(step.position, step.yaw, q_start)
        return traj.joint_space(q_start, goal, step.duration, step.profile, max_speed=max_joint_speed)
    if step.kind == "line":
        return traj.task_space_line(q_start, step.position, step.target_rotation(), step.duration,
                                    step.profile, max_tcp_speed=max_tcp_speed)
    if step.kind == "wait":
        return traj.hold(q_start, step.duration or 1.0, label="hold")
    raise ValueError(f"step kind '{step.kind}' has no motion")
