import numpy as np

N_JOINTS = 6

# ----------------------------------------------------------------------------
# Lite6 standard-DH table (Milestone 2 notation), in metres / radians
#   i    theta_i          d_i    a_i    alpha_i
#   1    q1               L1     0      -pi/2
#   2    q2 + OFFSET2     0      L3      pi
#   3    q3 + OFFSET3     0      L5      pi/2
#   4    q4               L6     0       pi/2
#   5    q5               L7     0      -pi/2
#   6    q6               L8     0       0
#
# OFFSET2 = OFFSET3 = -pi/2 and L7 = 0 are what makes this table reproduce the
# MuJoCo model (lite6.xml) exactly (verified in self_test).  CHECK these two
# against your notebook: with OFFSET2 = +pi/2 or OFFSET3 = 0 the pose does not
# match the simulator.
# ----------------------------------------------------------------------------
L1 = 0.2435
L3 = 0.2002
L5 = 0.087
L6 = 0.22761
L7 = 0.0
L8 = 0.0625

DH_THETA_OFFSET = np.array([0.0, -np.pi / 2, -np.pi / 2, 0.0, 0.0, 0.0])
DH_D = np.array([L1, 0.0, 0.0, L6, L7, L8])
DH_A = np.array([0.0, L3, L5, 0.0, 0.0, 0.0])
DH_ALPHA = np.array([-np.pi / 2, np.pi, np.pi / 2, np.pi / 2, -np.pi / 2, 0.0])

JOINT_LOWER = np.array([-2 * np.pi, -2.61799, -0.061087, -2 * np.pi, -2.1642, -2 * np.pi])
JOINT_UPPER = np.array([2 * np.pi, 2.61799, 5.23599, 2 * np.pi, 2.1642, 2 * np.pi])

SINGULARITY_EPS = 0.05      # s_min below this -> damping starts
MAX_DAMPING = 0.05          # lam at s_min = 0


# ============================================================================
# Helpers
# ============================================================================
def mujoco_order(x):
    """[linear; angular] -> [angular; linear] (MuJoCo's mj_objectVelocity order)."""
    x = np.asarray(x, float)
    return np.concatenate([x[3:], x[:3]])


def from_mujoco_order(x):
    """[angular; linear] -> [linear; angular]."""
    return mujoco_order(x)


def dh_transform(theta, d, a, alpha):
    """Standard-DH transform {i-1}T_{i} = Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def frames(q):
    """Return [T_0_0 ... T_0_6]: base-to-frame-i transforms (7 matrices)."""
    q = np.asarray(q, float)
    T = np.eye(4)
    out = [T.copy()]
    for i in range(N_JOINTS):
        T = T @ dh_transform(q[i] + DH_THETA_OFFSET[i], DH_D[i], DH_A[i], DH_ALPHA[i])
        out.append(T.copy())
    return out


def _axes_and_origins(q):
    """Joint axes z[i] and axis points p[i] (i = 0..5), the end-effector origin p_e,
    and T_0_6, all in the base frame.

    Standard DH: joint i+1 rotates about the z axis of frame i, so its axis is the
    z column of T_0_i and p[i] is the origin of frame i (i = 0..5)."""
    T = frames(q)
    z = [T[i][:3, 2] for i in range(N_JOINTS)]
    p = [T[i][:3, 3] for i in range(N_JOINTS)]
    return z, p, T[N_JOINTS][:3, 3], T[N_JOINTS]


# ============================================================================
# Position kinematics (needed by the Jacobian and by the validation)
# ============================================================================
def fk(q):
    """Forward position kinematics. Returns (position[3], rotation[3x3])."""
    T = frames(q)[N_JOINTS]
    return T[:3, 3].copy(), T[:3, :3].copy()


# ============================================================================
# Jacobian and its time derivative
# ============================================================================
def jacobian(q):
    """6x6 geometric Jacobian, rows [linear; angular], base frame."""
    z, p, p_e, _ = _axes_and_origins(q)
    J = np.zeros((6, N_JOINTS))
    for i in range(N_JOINTS):
        J[:3, i] = np.cross(z[i], p_e - p[i])
        J[3:, i] = z[i]
    return J


def jacobian_dot(q, qd):
    """Analytic time derivative of the Jacobian along (q, qd). 6x6."""
    qd = np.asarray(qd, float)
    z, p, p_e, _ = _axes_and_origins(q)
    J = jacobian(q)
    v_e = J[:3] @ qd                                   # pdot_e

    Jd = np.zeros((6, N_JOINTS))
    omega_prev = np.zeros(3)                           # w_{i-1}
    for i in range(N_JOINTS):
        z_dot = np.cross(omega_prev, z[i])
        p_dot_i = np.zeros(3)                          # pdot_i = sum_{k<i} z_k x (p_i-p_k) qd_k
        for k in range(i):
            p_dot_i += np.cross(z[k], p[i] - p[k]) * qd[k]
        Jd[:3, i] = np.cross(z_dot, p_e - p[i]) + np.cross(z[i], v_e - p_dot_i)
        Jd[3:, i] = z_dot
        omega_prev = omega_prev + z[i] * qd[i]
    return Jd


def jacobian_dot_numeric(q, qd, h=1e-6):
    """Central finite-difference Jdot (cross-check for the analytic version)."""
    q = np.asarray(q, float)
    qd = np.asarray(qd, float)
    return (jacobian(q + h * qd) - jacobian(q - h * qd)) / (2 * h)


# ============================================================================
# Singularity handling
# ============================================================================
def singularity_measure(q):
    """Return dict: smallest singular value, condition number, |det J|."""
    s = np.linalg.svd(jacobian(q), compute_uv=False)
    return {
        "sigma_min": float(s[-1]),
        "condition": float(s[0] / s[-1]) if s[-1] > 1e-12 else float("inf"),
        "det": float(abs(np.linalg.det(jacobian(q)))),
    }


def is_near_singularity(q, eps=SINGULARITY_EPS):
    return singularity_measure(q)["sigma_min"] < eps


def damped_solve(J, b, eps=SINGULARITY_EPS, max_damping=MAX_DAMPING):
    """Solve J x = b with singularity-adaptive damped least squares (SVD form)."""
    U, s, Vt = np.linalg.svd(J)
    s_min = s[-1]
    lam2 = 0.0 if s_min >= eps else (1.0 - (s_min / eps) ** 2) * max_damping ** 2
    filt = s / (s ** 2 + lam2)
    return Vt.T @ (filt * (U.T @ b))


# ============================================================================
# VELOCITY kinematics
# ============================================================================
def fk_velocity(q, qd):
    """Forward velocity kinematics:  [v; w] = J(q) qd.  Returns 6-vector."""
    return jacobian(q) @ np.asarray(qd, float)


def ik_velocity(q, twist, damped=True):
    """Inverse velocity kinematics:  qd = J^-1 [v; w]  (damped near singularities)."""
    J = jacobian(q)
    twist = np.asarray(twist, float)
    if damped:
        return damped_solve(J, twist)
    return np.linalg.solve(J, twist)


# ============================================================================
# ACCELERATION kinematics
# ============================================================================
def fk_acceleration(q, qd, qdd):
    """Forward acceleration kinematics:  [a; alpha] = J qdd + Jdot qd."""
    qd = np.asarray(qd, float)
    qdd = np.asarray(qdd, float)
    return jacobian(q) @ qdd + jacobian_dot(q, qd) @ qd


def ik_acceleration(q, qd, accel, damped=True):
    """Inverse acceleration kinematics:  qdd = J^-1 ([a; alpha] - Jdot qd)."""
    qd = np.asarray(qd, float)
    rhs = np.asarray(accel, float) - jacobian_dot(q, qd) @ qd
    J = jacobian(q)
    if damped:
        return damped_solve(J, rhs)
    return np.linalg.solve(J, rhs)


# ============================================================================
# Numerical position IK (damped least squares) -- used to find a start pose
# ============================================================================
def _rotation_error(R_cur, R_tgt):
    """Small-angle axis*angle vector that rotates R_cur toward R_tgt (base frame)."""
    return 0.5 * sum(np.cross(R_cur[:, k], R_tgt[:, k]) for k in range(3))


def ik_position(target_p, q0, target_R=None, max_iter=300, tol=1e-6, max_step=0.3):
    """Iterative IK. Returns (q, converged). Orientation optional."""
    q = np.clip(np.asarray(q0, float), JOINT_LOWER, JOINT_UPPER)
    target_p = np.asarray(target_p, float)
    for _ in range(max_iter):
        p, R = fk(q)
        e = target_p - p
        if target_R is not None:
            e = np.concatenate([e, _rotation_error(R, target_R)])
        if np.linalg.norm(e) < tol:
            return q, True
        J = jacobian(q) if target_R is not None else jacobian(q)[:3]
        lam2 = 0.05 ** 2
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(J.shape[0]), e)
        m = np.max(np.abs(dq))
        if m > max_step:
            dq *= max_step / m
        q = np.clip(q + dq, JOINT_LOWER, JOINT_UPPER)
    return q, False


# ============================================================================
# MUJOCO SIDE: set joint angles / velocities / accelerations, SENSE the end
# effector.  (mujoco is imported lazily, so everything above works without it.)
# ============================================================================
class MujocoArm:
    """The simulator side of the validation: input joint values, sense the end effector.

    Everything returned uses THIS FILE's conventions: [linear; angular], base frame.
    Gravity is disabled by default so that sensed accelerations are pure kinematics.
    """

    def __init__(self, xml_path, site="attachment_site", joints=None, gravity=False):
        import mujoco
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        names = joints or [f"joint{i}" for i in range(1, N_JOINTS + 1)]
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        if min(ids) < 0:
            raise ValueError(f"joint(s) not found in {xml_path}: {names}")
        self.qadr = [int(self.model.jnt_qposadr[j]) for j in ids]   # index into qpos
        self.dadr = [int(self.model.jnt_dofadr[j]) for j in ids]    # index into qvel / qacc

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site_id < 0:
            raise ValueError(f"site '{site}' not found in {xml_path}")

        if not gravity:
            self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)

    # ----- input -----------------------------------------------------------
    def _set_state(self, q, qd=None):
        d = self.data
        d.qvel[:] = 0.0
        d.qpos[self.qadr] = np.asarray(q, float)
        if qd is not None:
            d.qvel[self.dadr] = np.asarray(qd, float)
        self.mujoco.mj_forward(self.model, d)          # propagates positions + velocities

    # ----- sensing -----------------------------------------------------------
    def position(self, q):
        """Sensed end-effector (position[3], rotation[3x3]) for joint angles q."""
        self._set_state(q)
        d = self.data
        return d.site_xpos[self.site_id].copy(), d.site_xmat[self.site_id].reshape(3, 3).copy()

    def twist(self, q, qd):
        """Sensed [v; w] of the end effector (world axes, at the site origin)."""
        self._set_state(q, qd)
        res = np.zeros(6)                              # MuJoCo order: [angular; linear]
        self.mujoco.mj_objectVelocity(self.model, self.data, self.mujoco.mjtObj.mjOBJ_SITE,
                                      self.site_id, res, 0)
        return from_mujoco_order(res)

    def accel_fd(self, q, qd, qdd, h=1e-5):
        """Sensed [a; alpha]: central difference of the SENSED twist along
        q(t) = q + qd t + 0.5 qdd t^2.  Most robust way to sense acceleration."""
        q, qd, qdd = (np.asarray(x, float) for x in (q, qd, qdd))
        plus = self.twist(q + qd * h + 0.5 * qdd * h * h, qd + qdd * h)
        minus = self.twist(q - qd * h + 0.5 * qdd * h * h, qd - qdd * h)
        return (plus - minus) / (2 * h)

    def accel_rne(self, q, qd, qdd):
        """Sensed [a; alpha] straight from MuJoCo (mj_objectAcceleration). Optional
        cross-check: needs gravity disabled, otherwise -g is added to the result."""
        self._set_state(q, qd)
        self.data.qacc[:] = 0.0
        self.data.qacc[self.dadr] = np.asarray(qdd, float)
        self.mujoco.mj_rnePostConstraint(self.model, self.data)
        res = np.zeros(6)
        self.mujoco.mj_objectAcceleration(self.model, self.data, self.mujoco.mjtObj.mjOBJ_SITE,
                                          self.site_id, res, 0)
        return from_mujoco_order(res)

    def jacobian(self, q):
        """MuJoCo's own 6x6 Jacobian [linear; angular] (cross-check of jacobian(q))."""
        self._set_state(q)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        return np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])


def compare_with_mujoco(xml_path, n=100, tol=1e-3, seed=0, verbose=True):
    """Equations (this file) vs MuJoCo-sensed end effector, forward AND inverse.
    Returns True if every worst-case error is below `tol`."""
    arm = MujocoArm(xml_path)
    rng = np.random.default_rng(seed)
    names = ["position", "orientation", "jacobian", "velocity", "acceleration (FD)",
             "acceleration (RNE)", "inv position", "inv velocity", "inv acceleration"]
    worst = dict.fromkeys(names, 0.0)

    for _ in range(n):
        q = rng.uniform(JOINT_LOWER, JOINT_UPPER) * 0.8
        q[4] = rng.choice([-1, 1]) * rng.uniform(0.3, 2.0)      # stay off the wrist singularity
        qd = rng.uniform(-1, 1, N_JOINTS)
        qdd = rng.uniform(-2, 2, N_JOINTS)

        # ---- forward: input joint values, compare sensed vs equations --------
        p_s, R_s = arm.position(q)
        p_e, R_e = fk(q)
        worst["position"] = max(worst["position"], np.abs(p_s - p_e).max())
        worst["orientation"] = max(worst["orientation"], np.abs(R_s - R_e).max())
        worst["jacobian"] = max(worst["jacobian"], np.abs(arm.jacobian(q) - jacobian(q)).max())

        tw_s = arm.twist(q, qd)
        worst["velocity"] = max(worst["velocity"], np.abs(tw_s - fk_velocity(q, qd)).max())

        acc_e = fk_acceleration(q, qd, qdd)
        acc_fd = arm.accel_fd(q, qd, qdd)
        worst["acceleration (FD)"] = max(worst["acceleration (FD)"], np.abs(acc_fd - acc_e).max())
        worst["acceleration (RNE)"] = max(worst["acceleration (RNE)"],
                                          np.abs(arm.accel_rne(q, qd, qdd) - acc_e).max())

        # ---- inverse: feed the SENSED end-effector values to the inverse equations
        q_ik, ok = ik_position(p_s, q + rng.normal(0, 0.1, N_JOINTS), R_s)
        if ok:   # judge the IK by the pose MuJoCo senses for the solution it returns
            worst["inv position"] = max(worst["inv position"],
                                        np.abs(arm.position(q_ik)[0] - p_s).max())
        worst["inv velocity"] = max(worst["inv velocity"],
                                    np.abs(ik_velocity(q, tw_s, damped=False) - qd).max())
        worst["inv acceleration"] = max(worst["inv acceleration"],
                                        np.abs(ik_acceleration(q, qd, acc_fd, damped=False) - qdd).max())

    ok_all = True
    for k, v in worst.items():
        ok = v < tol
        if k != "acceleration (RNE)":          # RNE is an optional extra check
            ok_all &= ok
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {k:20s} worst error = {v:.2e}  (tol {tol:.0e})")
    return ok_all


# ============================================================================
# SELF-TEST  (independent of the DH code: uses the pos/quat chain from lite6.xml)
# ============================================================================
def _quat_to_R(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# (pos, quat) of link1..link6 relative to their parents, copied from lite6.xml
_XML_CHAIN = [
    ((0, 0, 0.2435), (1, 0, 0, 0)),
    ((0, 0, 0), (-1, 1, 1, 1)),
    ((0.2002, 0, 0), (-2.59734e-06, -0.707105, -0.707108, -2.59735e-06)),
    ((0.087, -0.22761, 0), (0.707105, 0.707108, 0, 0)),
    ((0, 0, 0), (1, 1, 0, 0)),
    ((0, 0.0625, 0), (1, -1, 0, 0)),
]


def _fk_from_xml_chain(q):
    """Reference FK built straight from the MJCF body tree (hinge about local z)."""
    T = np.eye(4)
    for (pos, quat), qi in zip(_XML_CHAIN, q):
        Tl = np.eye(4)
        Tl[:3, :3] = _quat_to_R(*quat)
        Tl[:3, 3] = pos
        c, s = np.cos(qi), np.sin(qi)
        Rz = np.eye(4)
        Rz[:2, :2] = [[c, -s], [s, c]]
        T = T @ Tl @ Rz
    return T[:3, 3], T[:3, :3]


def _angular_velocity_fd(q, qd, h=1e-6):
    """w from R(t): skew(w) = Rdot R^T, by central difference of FK."""
    _, Rp = fk(q + h * qd)
    _, Rm = fk(q - h * qd)
    _, R0 = fk(q)
    W = (Rp - Rm) / (2 * h) @ R0.T
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def self_test(verbose=True):
    rng = np.random.default_rng(1)
    worst = {k: 0.0 for k in ["fk_vs_xml", "J_vs_fd", "Jdot_vs_fd", "acc_vs_fd",
                              "vel_roundtrip", "acc_roundtrip", "ik_position"]}
    for _ in range(200):
        q = rng.uniform(JOINT_LOWER, JOINT_UPPER) * 0.8
        q[4] = rng.choice([-1, 1]) * rng.uniform(0.3, 2.0)        # keep the wrist away from q5=0
        qd = rng.uniform(-1, 1, 6)
        qdd = rng.uniform(-2, 2, 6)

        # 1) DH FK == FK from the XML body tree
        p, R = fk(q)
        p_x, R_x = _fk_from_xml_chain(q)
        worst["fk_vs_xml"] = max(worst["fk_vs_xml"], np.abs(p - p_x).max(), np.abs(R - R_x).max())

        # 2) J qd == d/dt FK (linear: finite difference, angular: Rdot R^T)
        h = 1e-6
        v_fd = (fk(q + h * qd)[0] - fk(q - h * qd)[0]) / (2 * h)
        w_fd = _angular_velocity_fd(q, qd)
        twist = fk_velocity(q, qd)
        worst["J_vs_fd"] = max(worst["J_vs_fd"], np.abs(twist - np.concatenate([v_fd, w_fd])).max())

        # 3) analytic Jdot == numeric Jdot
        worst["Jdot_vs_fd"] = max(worst["Jdot_vs_fd"],
                                  np.abs(jacobian_dot(q, qd) - jacobian_dot_numeric(q, qd)).max())

        # 4) forward acceleration == d/dt (J(q(t)) qd(t)) along a quadratic trajectory
        def twist_at(t):
            return fk_velocity(q + qd * t + 0.5 * qdd * t * t, qd + qdd * t)
        acc_fd = (twist_at(h) - twist_at(-h)) / (2 * h)
        worst["acc_vs_fd"] = max(worst["acc_vs_fd"],
                                 np.abs(fk_acceleration(q, qd, qdd) - acc_fd).max())

        # 5) round trips (inverse(forward(x)) == x)
        worst["vel_roundtrip"] = max(worst["vel_roundtrip"],
                                     np.abs(ik_velocity(q, twist, damped=False) - qd).max())
        acc = fk_acceleration(q, qd, qdd)
        worst["acc_roundtrip"] = max(worst["acc_roundtrip"],
                                     np.abs(ik_acceleration(q, qd, acc, damped=False) - qdd).max())

        # 6) numerical position IK reproduces the pose
        q_sol, ok = ik_position(p, q + rng.normal(0, 0.1, 6), R)
        if ok:
            worst["ik_position"] = max(worst["ik_position"], np.abs(fk(q_sol)[0] - p).max())

    # fk_vs_xml: lite6.xml stores some quaternions rounded to ~6 digits (0.707105 vs
    # 0.7071068), so MuJoCo's arm differs from the ideal DH model by ~1e-5 m.
    # With exact quaternions the two agree to 1e-15 (checked separately).
    tol = {"fk_vs_xml": 5e-5, "J_vs_fd": 1e-6, "Jdot_vs_fd": 1e-6, "acc_vs_fd": 1e-5,
           "vel_roundtrip": 1e-6, "acc_roundtrip": 1e-5, "ik_position": 1e-5}
    ok_all = True
    for k, v in worst.items():
        ok = v < tol[k]
        ok_all &= ok
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {k:14s} worst error = {v:.2e}  (tol {tol[k]:.0e})")

    # singularity demo: wrist singular at q5 = 0
    q_sing = np.array([0.3, -0.5, 1.0, 0.2, 0.0, 0.1])
    m = singularity_measure(q_sing)
    qd_damped = ik_velocity(q_sing, np.array([0.1, 0, 0, 0, 0.5, 0]))
    if verbose:
        print(f"\nAt q5 = 0 (wrist singularity): sigma_min = {m['sigma_min']:.2e}, "
              f"|det J| = {m['det']:.2e}, damped qd max = {np.abs(qd_damped).max():.2f} rad/s (finite)")
    ok_all &= bool(np.isfinite(qd_damped).all()) and m["sigma_min"] < 1e-6
    return ok_all


if __name__ == "__main__":
    import sys
    print("== Self-test (equations vs independent numeric checks, no simulator) ==")
    good = self_test()
    if len(sys.argv) > 1:                      # python3 kinematics.py path/to/scene.xml
        print(f"\n== Equations vs MuJoCo-sensed end effector: {sys.argv[1]} ==")
        good &= compare_with_mujoco(sys.argv[1])
    raise SystemExit(0 if good else 1)