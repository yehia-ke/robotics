"""
Motor-level joint TORQUE controller for the Lite6 (Milestone 4).

Control law (per joint, computed every control period dt):

    e      = q_ref  - q                       position error
    e'     = q'_ref - q'                      velocity error (q' measured by the joint encoders)
    I_e   += e dt                             integral of the error (with anti-windup)

    tau = Kp e + Ki I_e + Kd e'               PID feedback
        + G(q)                                gravity compensation      (use_gravity)
        + M(q) q''_ref + B q'_ref             inertia + friction feed-forward (use_feedforward)

    tau is clipped to the motor torque limits.

* The PID part makes the loop CLOSED: it reacts to the measured joint angle/velocity.
* G(q) (dynamics.py) cancels the gravity load, so the PID does not have to hold the arm
  up with a steady error; the integral term then only removes what the model misses
  (e.g. the mass of a grasped vial).
* M(q) q''_ref (dynamics.py) supplies the torque needed to ACCELERATE along the trajectory
  so the tracking error stays small during fast motions. B q'_ref cancels the viscous joint
  damping (B = 1 N*m*s/rad). Switch it off to see pure PID + gravity compensation.
* Anti-windup: the integral only grows while the joint's torque is not saturated, and it is
  clamped to +/- integral_limit.

The derivative uses the measured joint velocity (not a differentiated angle), so there is
no derivative noise / kick.
"""
import numpy as np

import dynamics

N_JOINTS = 6

# Motor torque limits (N*m) = ctrlrange of the motors in lite6_gripper_torque.xml
TORQUE_LIMITS = np.array([50.0, 50.0, 32.0, 32.0, 32.0, 20.0])

# Gains tuned in simulation (validation/tune_controller.py) for a 250 Hz loop.
# Effective joint inertia is ~0.1 - 0.5 kg m^2 (rotor armature 0.1 included), so
# Kp = 400 gives a closed-loop bandwidth of roughly 30 - 60 rad/s.
DEFAULT_KP = np.array([400.0, 400.0, 400.0, 200.0, 200.0, 150.0])
DEFAULT_KI = np.array([300.0, 300.0, 300.0, 150.0, 150.0, 100.0])
DEFAULT_KD = np.array([25.0, 25.0, 20.0, 10.0, 10.0, 8.0])
DEFAULT_INTEGRAL_LIMIT = 0.05          # rad*s: caps Ki * I_e at a few N*m


class JointTorqueController:
    """PID + gravity compensation (+ inertia feed-forward) joint torque controller."""

    def __init__(self, kp=DEFAULT_KP, ki=DEFAULT_KI, kd=DEFAULT_KD,
                 torque_limits=TORQUE_LIMITS, integral_limit=DEFAULT_INTEGRAL_LIMIT,
                 use_gravity=True, use_feedforward=True):
        self.kp = np.array(kp, float)
        self.ki = np.array(ki, float)
        self.kd = np.array(kd, float)
        self.torque_limits = np.array(torque_limits, float)
        self.integral_limit = float(integral_limit)
        self.use_gravity = bool(use_gravity)
        self.use_feedforward = bool(use_feedforward)
        self.payload_mass = 0.0
        self.integral = np.zeros(N_JOINTS)
        self.last_terms = None

    def reset(self):
        """Forget the integral (call when the reference jumps or control restarts)."""
        self.integral[:] = 0.0

    def set_gains(self, kp=None, ki=None, kd=None):
        if kp is not None:
            self.kp = np.array(kp, float)
        if ki is not None:
            self.ki = np.array(ki, float)
        if kd is not None:
            self.kd = np.array(kd, float)

    def compute(self, q, qd, q_ref, qd_ref, qdd_ref, dt):
        """Return the 6 joint torques for the measured (q, qd) and the reference.

        Also stores the individual terms in self.last_terms (for logging / the GUI)."""
        q, qd, q_ref, qd_ref, qdd_ref = (np.asarray(x, float) for x in (q, qd, q_ref, qd_ref, qdd_ref))
        error = q_ref - q
        error_rate = qd_ref - qd

        proportional = self.kp * error
        derivative = self.kd * error_rate
        integral = self.ki * self.integral

        gravity = np.zeros(N_JOINTS)
        feedforward = np.zeros(N_JOINTS)
        if self.use_gravity or self.use_feedforward:
            G, M = dynamics.gravity_and_mass(q, self.payload_mass)
            if self.use_gravity:
                gravity = G
            if self.use_feedforward:
                feedforward = M @ qdd_ref + dynamics.JOINT_DAMPING * qd_ref

        unsaturated = proportional + integral + derivative + gravity + feedforward
        torque = np.clip(unsaturated, -self.torque_limits, self.torque_limits)
        saturated = np.abs(unsaturated) > self.torque_limits

        # anti-windup: integrate only on joints that are not saturated
        self.integral = np.where(saturated, self.integral, self.integral + error * dt)
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        self.last_terms = {
            "error": error, "error_rate": error_rate, "proportional": proportional,
            "integral": integral, "derivative": derivative, "gravity": gravity,
            "feedforward": feedforward, "torque": torque, "saturated": saturated,
        }
        return torque


class ReferenceGenerator:
    """Smooth set-point for jogging with the GUI sliders (no planned trajectory running).

    Each joint moves toward its target with a trapezoidal velocity profile: it accelerates at
    max_acceleration up to max_speed and brakes so that it stops exactly on the target.
    Produces (q_ref, qd_ref, qdd_ref) for the controller. (Same profile as Milestone 3.)"""

    def __init__(self, q, max_speed=1.0, max_acceleration=4.0):
        self.position = np.array(q, float)
        self.velocity = np.zeros(N_JOINTS)
        self.acceleration = np.zeros(N_JOINTS)
        self.target = np.array(q, float)
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)

    def reset(self, q):
        self.position = np.array(q, float)
        self.velocity[:] = 0.0
        self.acceleration[:] = 0.0
        self.target = np.array(q, float)

    def is_moving(self):
        return bool(np.any(self.velocity != 0.0) or np.any(self.target != self.position))

    def step(self, dt):
        largest_speed_change = self.max_acceleration * dt
        for joint in range(N_JOINTS):
            remaining = self.target[joint] - self.position[joint]
            speed = self.velocity[joint]
            # largest speed from which it can still brake to a stop within `remaining`
            stopping_speed = largest_speed_change * (
                -0.5 + np.sqrt(0.25 + 2.0 * abs(remaining) / (self.max_acceleration * dt ** 2)))
            wanted = np.copysign(min(self.max_speed, stopping_speed), remaining)
            new_speed = speed + np.clip(wanted - speed, -largest_speed_change, largest_speed_change)
            new_position = self.position[joint] + new_speed * dt
            if (self.target[joint] - new_position) * remaining <= 0.0 and abs(speed) <= largest_speed_change:
                new_position = self.target[joint]
                new_speed = 0.0
            self.acceleration[joint] = (new_speed - speed) / dt
            self.velocity[joint] = new_speed
            self.position[joint] = new_position
        return self.position.copy(), self.velocity.copy(), self.acceleration.copy()
