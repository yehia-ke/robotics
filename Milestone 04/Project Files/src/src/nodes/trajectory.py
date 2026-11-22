"""
Trajectory generation for the Lite6 (Milestone 4).

JOINT-SPACE trajectories (each joint independently, rest-to-rest by default)
    Cubic   q(t) = a0 + a1 t + a2 t^2 + a3 t^3
            4 conditions: q(0) = q0, q(T) = qf, q'(0) = v0, q'(T) = vf
            a0 = q0, a1 = v0,
            a2 = (3(qf - q0) - (2 v0 + vf) T) / T^2,
            a3 = (-2(qf - q0) + (v0 + vf) T) / T^3
            -> position and velocity continuous, acceleration JUMPS at t = 0 and t = T.
    Quintic q(t) = a0 + ... + a5 t^5
            6 conditions: + q''(0) = acc0, q''(T) = accf  -> acceleration is continuous too
            (a 6x6 linear system, solved below in closed form).

TASK-SPACE trajectories (the TCP moves on a STRAIGHT LINE)
    p(t) = p0 + s(t) (p1 - p0)
    R(t) = R0 Exp(s(t) r),  r = Log(R0^T R1)        (orientation turns about one fixed axis)
    s(t) goes 0 -> 1 with a cubic or quintic time scaling (the same polynomials as above).
    Velocity  : v = s' (p1 - p0),   w = s' R0 r
    Accel.    : a = s''(p1 - p0),   alpha = s'' R0 r
    Joint values for every sample (Milestones 2-3 kinematics):
        q     = IK(p, R)                     (numerical IK, warm-started from the previous sample)
        q'    = J(q)^-1 [v; w]               (inverse velocity kinematics)
        q''   = J(q)^-1 ([a; alpha] - J' q') (inverse acceleration kinematics)

Every trajectory is finally SAMPLED (SampledTrajectory, default dt = 4 ms = control period)
and sent to the controller as (t, q, q', q'') points.
"""
import numpy as np

import kinematics as kin

DEFAULT_SAMPLE_TIME = 0.004
# task-space samples each need an IK solve + inverse velocity/acceleration kinematics, so
# they are 10 ms apart; the controller interpolates linearly between samples
TASK_SPACE_SAMPLE_TIME = 0.01
# peak speed of a rest-to-rest profile = factor * distance / T
PEAK_SPEED_FACTOR = {"cubic": 1.5, "quintic": 1.875}
# Lite6 joint speed limit used to size / check trajectories (rad/s); the real arm allows ~3.1
MAX_JOINT_SPEED = 2.5
SINGULARITY_WARNING = 0.02     # smallest singular value of J below which a path is rejected


# ============================================================================
# Polynomials
# ============================================================================
def cubic_coefficients(q0, qf, T, v0=0.0, vf=0.0):
    """Cubic coefficients [a0, a1, a2, a3] (each an array over joints)."""
    q0, qf, v0, vf = (np.asarray(x, float) for x in (q0, qf, v0, vf))
    q0, qf, v0, vf = np.broadcast_arrays(q0, qf, v0, vf)
    delta = qf - q0
    a0 = q0
    a1 = v0
    a2 = (3.0 * delta - (2.0 * v0 + vf) * T) / T ** 2
    a3 = (-2.0 * delta + (v0 + vf) * T) / T ** 3
    return [a0, a1, a2, a3]


def quintic_coefficients(q0, qf, T, v0=0.0, vf=0.0, acc0=0.0, accf=0.0):
    """Quintic coefficients [a0 .. a5] (each an array over joints)."""
    q0, qf, v0, vf, acc0, accf = (np.asarray(x, float) for x in (q0, qf, v0, vf, acc0, accf))
    q0, qf, v0, vf, acc0, accf = np.broadcast_arrays(q0, qf, v0, vf, acc0, accf)
    delta = qf - q0
    a0 = q0
    a1 = v0
    a2 = acc0 / 2.0
    a3 = (20.0 * delta - (8.0 * vf + 12.0 * v0) * T - (3.0 * acc0 - accf) * T ** 2) / (2.0 * T ** 3)
    a4 = (-30.0 * delta + (14.0 * vf + 16.0 * v0) * T + (3.0 * acc0 - 2.0 * accf) * T ** 2) / (2.0 * T ** 4)
    a5 = (12.0 * delta - 6.0 * (vf + v0) * T + (accf - acc0) * T ** 2) / (2.0 * T ** 5)
    return [a0, a1, a2, a3, a4, a5]


class Polynomial:
    """q(t) = sum a_k t^k for 0 <= t <= T (held constant outside)."""

    def __init__(self, coefficients, duration):
        self.coefficients = [np.asarray(c, float) for c in coefficients]
        self.duration = float(duration)

    def evaluate(self, t):
        """Return (position, velocity, acceleration) at time t."""
        t = min(max(float(t), 0.0), self.duration)
        position = np.zeros_like(self.coefficients[0])
        velocity = np.zeros_like(position)
        acceleration = np.zeros_like(position)
        for k, a in enumerate(self.coefficients):
            position = position + a * t ** k
            if k >= 1:
                velocity = velocity + k * a * t ** (k - 1)
            if k >= 2:
                acceleration = acceleration + k * (k - 1) * a * t ** (k - 2)
        return position, velocity, acceleration


def make_polynomial(profile, start, goal, duration):
    """Rest-to-rest cubic or quintic from start to goal (scalars or arrays)."""
    if profile == "cubic":
        return Polynomial(cubic_coefficients(start, goal, duration), duration)
    if profile == "quintic":
        return Polynomial(quintic_coefficients(start, goal, duration), duration)
    raise ValueError(f"unknown profile '{profile}' (use 'cubic' or 'quintic')")


def minimum_duration(distance, max_speed, profile, shortest=0.5):
    """Shortest T for which the profile's peak speed stays below max_speed."""
    return max(shortest, PEAK_SPEED_FACTOR[profile] * abs(distance) / max_speed)


# ============================================================================
# Sampled trajectory (what is sent to the controller)
# ============================================================================
class SampledTrajectory:
    """Joint trajectory as samples: t[k], q[k], qd[k], qdd[k], plus the desired TCP pose."""

    def __init__(self, label, t, q, qd, qdd, tcp_position=None, kind="joint"):
        self.label = label
        self.kind = kind                    # "joint" or "task"
        self.t = np.asarray(t, float)
        self.q = np.asarray(q, float)
        self.qd = np.asarray(qd, float)
        self.qdd = np.asarray(qdd, float)
        if tcp_position is None:            # joint-space: the TCP path is FK of the joints
            tcp_position = np.array([kin.fk(qk)[0] for qk in self.q])
        self.tcp_position = np.asarray(tcp_position, float)

    @property
    def duration(self):
        return float(self.t[-1])

    @property
    def final_position(self):
        return self.q[-1].copy()

    def sample(self, t):
        """Linear interpolation of (q, qd, qdd) at time t (held at the ends)."""
        if t <= self.t[0]:
            return self.q[0], self.qd[0] * 0.0 if t < self.t[0] else self.qd[0], self.qdd[0]
        if t >= self.t[-1]:
            return self.q[-1], self.qd[-1] * 0.0, self.qdd[-1] * 0.0
        k = int(np.searchsorted(self.t, t) - 1)
        w = (t - self.t[k]) / (self.t[k + 1] - self.t[k])
        blend = lambda x: (1.0 - w) * x[k] + w * x[k + 1]
        return blend(self.q), blend(self.qd), blend(self.qdd)

    def check(self, max_speed=MAX_JOINT_SPEED):
        """Feasibility report: joint limits, peak joint speed/acceleration, singularity margin."""
        below = self.q < kin.JOINT_LOWER - 1e-6
        above = self.q > kin.JOINT_UPPER + 1e-6
        sigma = [kin.singularity_measure(qk)["sigma_min"] for qk in self.q[::max(1, len(self.q) // 50)]]
        report = {
            "duration_s": self.duration,
            "peak_joint_speed": float(np.abs(self.qd).max()),
            "peak_joint_acceleration": float(np.abs(self.qdd).max()),
            "within_joint_limits": not (below.any() or above.any()),
            "min_singular_value": float(min(sigma)),
        }
        report["ok"] = (report["within_joint_limits"]
                        and report["peak_joint_speed"] <= max_speed * 1.001)
        return report


def concatenate(label, trajectories):
    """Join trajectories end to end (each must start where the previous one ends)."""
    t, q, qd, qdd, tcp = [], [], [], [], []
    offset = 0.0
    for k, traj in enumerate(trajectories):
        first = 0 if k == 0 else 1           # drop the duplicated joint sample
        t.append(traj.t[first:] + offset)
        q.append(traj.q[first:])
        qd.append(traj.qd[first:])
        qdd.append(traj.qdd[first:])
        tcp.append(traj.tcp_position[first:])
        offset += traj.duration
    kinds = {traj.kind for traj in trajectories}
    return SampledTrajectory(label, np.concatenate(t), np.concatenate(q), np.concatenate(qd),
                             np.concatenate(qdd), np.concatenate(tcp),
                             kind=kinds.pop() if len(kinds) == 1 else "mixed")


def _time_grid(duration, dt):
    n = max(2, int(np.ceil(duration / dt)) + 1)
    return np.linspace(0.0, duration, n)


# ============================================================================
# Joint-space trajectory
# ============================================================================
def joint_space(q_start, q_goal, duration=None, profile="quintic", dt=DEFAULT_SAMPLE_TIME,
                max_speed=MAX_JOINT_SPEED):
    """Cubic / quintic rest-to-rest trajectory of all joints, synchronised to one duration.

    If duration is None the shortest duration that respects max_speed is used (the joint
    with the largest motion sets it)."""
    q_start = np.asarray(q_start, float)
    q_goal = np.clip(np.asarray(q_goal, float), kin.JOINT_LOWER, kin.JOINT_UPPER)
    if duration is None:
        duration = minimum_duration(np.abs(q_goal - q_start).max(), max_speed, profile)
    poly = make_polynomial(profile, q_start, q_goal, duration)
    t = _time_grid(duration, dt)
    samples = [poly.evaluate(tk) for tk in t]
    q = np.array([s[0] for s in samples])
    qd = np.array([s[1] for s in samples])
    qdd = np.array([s[2] for s in samples])
    return SampledTrajectory(f"joint-space {profile} ({duration:.2f} s)", t, q, qd, qdd, kind="joint")


# ============================================================================
# Task-space straight line
# ============================================================================
class StraightLine:
    """Straight-line TCP motion p0 -> p1 with orientation R0 -> R1 and a cubic/quintic s(t)."""

    def __init__(self, p0, R0, p1, R1, duration, profile="quintic"):
        self.p0 = np.asarray(p0, float)
        self.p1 = np.asarray(p1, float)
        self.R0 = np.asarray(R0, float)
        self.R1 = np.asarray(R1, float)
        self.duration = float(duration)
        self.profile = profile
        self.rotation = kin.axis_angle_from_rotation(self.R0.T @ self.R1)   # r = Log(R0^T R1)
        self.scaling = make_polynomial(profile, 0.0, 1.0, duration)        # s(t): 0 -> 1

    @property
    def length(self):
        return float(np.linalg.norm(self.p1 - self.p0))

    def evaluate(self, t):
        """TCP (p, R, twist [v; w], acceleration [a; alpha]) at time t."""
        s, s_dot, s_ddot = (float(x) for x in self.scaling.evaluate(t))
        delta = self.p1 - self.p0
        p = self.p0 + s * delta
        R = self.R0 @ kin.rotation_from_axis_angle(s * self.rotation)
        axis_world = self.R0 @ self.rotation
        twist = np.concatenate([s_dot * delta, s_dot * axis_world])
        acceleration = np.concatenate([s_ddot * delta, s_ddot * axis_world])
        return p, R, twist, acceleration


class TrajectoryError(RuntimeError):
    pass


def task_space_line(q_start, p_goal, R_goal=None, duration=None, profile="quintic",
                    dt=TASK_SPACE_SAMPLE_TIME, max_tcp_speed=0.15, max_speed=MAX_JOINT_SPEED):
    """Straight line of the TCP from FK(q_start) to (p_goal, R_goal), as a joint trajectory.

    R_goal None keeps the start orientation. If duration is None it is chosen from
    max_tcp_speed (m/s). Raises TrajectoryError if a sample is unreachable, too close to a
    singularity, or (when duration was given) needs more than max_speed at a joint."""
    q_start = np.asarray(q_start, float)
    p0, R0 = kin.fk(q_start)
    R_goal = R0 if R_goal is None else np.asarray(R_goal, float)
    p_goal = np.asarray(p_goal, float)
    if duration is None:
        angle = np.linalg.norm(kin.axis_angle_from_rotation(R0.T @ R_goal))
        duration = max(minimum_duration(np.linalg.norm(p_goal - p0), max_tcp_speed, profile),
                       minimum_duration(angle, 1.0, profile))        # <= 1 rad/s tool rotation
    line = StraightLine(p0, R0, p_goal, R_goal, duration, profile)

    t = _time_grid(duration, dt)
    q = np.zeros((len(t), kin.N_JOINTS))
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    tcp = np.zeros((len(t), 3))
    q_previous = q_start
    for k, tk in enumerate(t):
        p, R, twist, acceleration = line.evaluate(tk)
        q_k, converged = kin.ik_position(p, q_previous, R, tol=1e-7)
        if not converged:
            raise TrajectoryError(f"IK failed at t = {tk:.3f} s, p = {np.round(p, 3)} (unreachable?)")
        if np.abs(q_k - q_previous).max() > 0.2:
            raise TrajectoryError(f"IK jumped to another branch at t = {tk:.3f} s")
        sigma = kin.singularity_measure(q_k)["sigma_min"]
        if sigma < SINGULARITY_WARNING:
            raise TrajectoryError(f"path passes near a singularity at t = {tk:.3f} s "
                                  f"(sigma_min = {sigma:.3f})")
        q[k] = q_k
        qd[k] = kin.ik_velocity(q_k, twist)                       # q'  = J^-1 [v; w]
        qdd[k] = kin.ik_acceleration(q_k, qd[k], acceleration)   # q'' = J^-1 ([a; alpha] - J' q')
        tcp[k] = p
        q_previous = q_k
    qd[-1] = 0.0
    qdd[-1] = 0.0
    trajectory = SampledTrajectory(
        f"task-space line {profile} ({line.length * 100:.1f} cm, {duration:.2f} s)",
        t, q, qd, qdd, tcp, kind="task")
    peak = float(np.abs(qd).max())
    if peak > max_speed:
        raise TrajectoryError(f"line needs {peak:.2f} rad/s at a joint (> {max_speed}); "
                              f"use a longer duration")
    return trajectory


def hold(q, duration, dt=DEFAULT_SAMPLE_TIME, label="hold"):
    """Stay still at q for `duration` seconds."""
    t = _time_grid(duration, dt)
    q = np.tile(np.asarray(q, float), (len(t), 1))
    zeros = np.zeros_like(q)
    return SampledTrajectory(label, t, q, zeros, zeros, kind="joint")


# ============================================================================
# Self test
# ============================================================================
def self_test(verbose=True):
    ok = True
    # boundary conditions of the polynomials
    for profile in ("cubic", "quintic"):
        poly = make_polynomial(profile, np.array([0.1, -1.0]), np.array([1.2, 0.5]), 2.0)
        p0, v0, a0 = poly.evaluate(0.0)
        p1, v1, a1 = poly.evaluate(2.0)
        good = (np.allclose(p0, [0.1, -1.0]) and np.allclose(p1, [1.2, 0.5])
                and np.allclose(v0, 0) and np.allclose(v1, 0))
        if profile == "quintic":
            good &= bool(np.allclose(a0, 0) and np.allclose(a1, 0))
        ok &= good
        if verbose:
            print(f"{'PASS' if good else 'FAIL'}  {profile} boundary conditions "
                  f"(start acc {a0.round(3)}, end acc {a1.round(3)})")
    # numeric derivative of the polynomial == analytic velocity / acceleration
    poly = make_polynomial("quintic", 0.0, 1.0, 1.5)
    h = 1e-6
    for t in (0.3, 0.75, 1.2):
        v_fd = (poly.evaluate(t + h)[0] - poly.evaluate(t - h)[0]) / (2 * h)
        a_fd = (poly.evaluate(t + h)[1] - poly.evaluate(t - h)[1]) / (2 * h)
        ok &= bool(abs(v_fd - poly.evaluate(t)[1]) < 1e-6 and abs(a_fd - poly.evaluate(t)[2]) < 1e-5)
    # task-space line: FK of the joint samples lies on the line, and J q' == [v; w]
    q_start = np.array([0.2, 0.6, 1.5, 0.0, 0.9, 0.2])
    p0, R0 = kin.fk(q_start)
    goal = p0 + np.array([0.05, -0.08, -0.06])
    traj = task_space_line(q_start, goal, kin.top_down_rotation(0.3), duration=2.0)
    direction = (goal - p0) / np.linalg.norm(goal - p0)
    worst_off_line = max(np.linalg.norm(np.cross(kin.fk(qk)[0] - p0, direction)) for qk in traj.q)
    worst_end = np.linalg.norm(kin.fk(traj.q[-1])[0] - goal)
    line = StraightLine(p0, R0, goal, kin.top_down_rotation(0.3), 2.0)
    worst_twist = max(np.abs(kin.fk_velocity(traj.q[k], traj.qd[k]) - line.evaluate(traj.t[k])[2]).max()
                      for k in range(0, len(traj.t), 25))
    good = worst_off_line < 1e-6 and worst_end < 1e-6 and worst_twist < 1e-6
    ok &= good
    if verbose:
        print(f"{'PASS' if good else 'FAIL'}  straight line: max distance from line {worst_off_line:.1e} m, "
              f"end error {worst_end:.1e} m, |J q' - [v;w]| {worst_twist:.1e}")
        print("      check:", traj.check())
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)
