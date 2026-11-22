"""
Collision check of planned trajectories against the lab scene (Milestone 4).

Before a trajectory is sent to the controller, every few samples the arm is placed in a
private copy of the MuJoCo scene (no simulation, kinematics only) and MuJoCo's collision
detection is asked for contacts. A trajectory is rejected if the arm would
    * hit itself (two robot bodies that are not neighbours), or
    * hit a fixed part of the cell (bench top, tray, bins, rack, hot plate, ...).
Contacts with loose labware (vials, flask, beakers) are ignored: touching them is the job.

This is how the visualisation model is used to test motions before the closed loop runs them.
"""
import numpy as np

ROBOT_ROOT_BODY = "link_base"
PENETRATION_TOLERANCE = 0.001      # m: contacts shallower than this are ignored (touching)


class CollisionChecker:

    def __init__(self, model_path):
        import mujoco
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "lab_home")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        model = self.model
        self.arm_qpos = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")])
                         for i in range(1, 7)]
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_ROOT_BODY)
        self.robot_bodies = {b for b in range(model.nbody) if self._is_descendant(b, root)}
        # bodies that move freely (free joint somewhere up their chain): loose labware
        self.loose_bodies = {b for b in range(model.nbody) if self._has_free_joint(b)}
        fingers = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                   for n in ("gripper_left_finger", "gripper_right_finger")}
        self.ignored_pairs = {frozenset(fingers)}

    def _is_descendant(self, body, root):
        while body > 0:
            if body == root:
                return True
            body = int(self.model.body_parentid[body])
        return body == root

    def _has_free_joint(self, body):
        while body > 0:
            start, count = self.model.body_jntadr[body], self.model.body_jntnum[body]
            for joint in range(start, start + count):
                if self.model.jnt_type[joint] == self.mujoco.mjtJoint.mjJNT_FREE:
                    return True
            body = int(self.model.body_parentid[body])
        return False

    def _name(self, body):
        return self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body) or "world"

    def collisions(self, q):
        """List of (body A, body B, penetration depth m) for the arm at joint angles q."""
        self.data.qpos[self.arm_qpos] = q
        self.mujoco.mj_kinematics(self.model, self.data)
        self.mujoco.mj_collision(self.model, self.data)
        found = []
        for contact in self.data.contact[:self.data.ncon]:
            if contact.dist > -PENETRATION_TOLERANCE:
                continue
            a = int(self.model.geom_bodyid[contact.geom1])
            b = int(self.model.geom_bodyid[contact.geom2])
            robot_a, robot_b = a in self.robot_bodies, b in self.robot_bodies
            if not (robot_a or robot_b):
                continue                                   # labware vs scene: not our business
            if a in self.loose_bodies or b in self.loose_bodies:
                continue                                   # grasping / pushing labware is allowed
            if frozenset((a, b)) in self.ignored_pairs:
                continue
            found.append((self._name(a), self._name(b), float(-contact.dist)))
        return found

    def check_trajectory(self, trajectory, every=3):
        """None if collision-free, else a description of the first collision."""
        indices = list(range(0, len(trajectory.q), every)) + [len(trajectory.q) - 1]
        for k in indices:
            hits = self.collisions(trajectory.q[k])
            if hits:
                a, b, depth = max(hits, key=lambda h: h[2])
                return (f"collision at t = {trajectory.t[k]:.2f} s: {a} hits {b} "
                        f"({depth * 1000:.1f} mm) - trajectory not executed")
        return None


if __name__ == "__main__":
    import os
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "model", "ufactory_lite6", "lab-scene.xml")
    checker = CollisionChecker(path)
    print("home:", checker.collisions(np.array([0, 0, 1.57, 0, 1.57, 0])) or "free")
    print("arm into the bench:", checker.collisions(np.array([0, 1.9, 1.2, 0, 1.0, 0])))
