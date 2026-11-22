"""
Lite6 rigid-body dynamics terms used by the motor-level controller (Milestone 4).

Equation of motion of the arm (n = 6 joints):

    tau = M(q) q'' + C(q, q') q' + G(q) + B q'

The controller (controller.py) needs two of these terms, derived here from the DH
kinematics (kinematics.py) and the link inertial data of the CAD model:

* Gravity vector, from the potential energy P(q) = sum_i m_i g0 z_ci(q):

      G(q) = dP/dq = sum_i m_i g0 * (J_v,ci(q))[z row]^T

  where J_v,ci is the linear Jacobian of the centre of mass c_i of link i
  (column k <= i: z_{k-1} x (c_i - p_{k-1}), zero for k > i).

* Mass (inertia) matrix, from the kinetic energy:

      M(q) = sum_i [ m_i J_v,ci^T J_v,ci + J_w,i^T R_i I_i R_i^T J_w,i ] + diag(armature)

  J_w,i = angular Jacobian of link i (column k <= i: z_{k-1}), R_i = orientation of the
  link's principal inertia frame, I_i = its diagonal inertia. "armature" is the reflected
  rotor inertia of each joint motor (0.1 kg m^2 in the model).

The link data below is copied from lite6_gripper_torque.xml (<inertial> elements).
Link 6 carries the gripper: link6 + gripper body + both fingers are lumped into one rigid
body. The body frames are tied to the DH frames by constant transforms computed from the
XML body tree (kinematics._XML_CHAIN), so no simulator is needed at run time.

Validation:  python3 dynamics.py path/to/lab-scene.xml
    compares G(q) with MuJoCo's qfrc_bias (q' = 0) and M(q) with mj_fullM.
"""
import numpy as np

import kinematics as kin

GRAVITY = 9.81
ARMATURE = np.full(kin.N_JOINTS, 0.1)          # <joint armature="0.1"> (class lite6)
JOINT_DAMPING = np.full(kin.N_JOINTS, 1.0)     # <joint damping="1">     (class lite6)

# (mass, centre of mass in the BODY frame, principal-axes quaternion (w x y z), diagonal inertia)
_LINK_INERTIALS = [
    (1.411, (-0.00036, 0.04195, -0.0025), (0.608059, 0.792349, -0.0438707, 0.0228718),
     (0.00145276, 0.00135275, 0.000853355)),
    (1.34, (0.179, 0.0, 0.0584), (0.417561, 0.571619, 0.569585, 0.417693),
     (0.00560971, 0.0052152, 0.00122018)),
    (0.953, (0.072, -0.0357, -0.001), (0.128259, 0.662963, -0.167256, 0.71837),
     (0.0018521, 0.00175546, 0.000703807)),
    (1.284, (-0.002, -0.0285, -0.0813), (0.975248, 0.22109, 0.00203498, -0.00262178),
     (0.00370503, 0.00349091, 0.00109586)),
    (0.804, (0.0, 0.01, 0.0019), (0.71423, 0.696388, -0.0531933, 0.0456997),
     (0.000567553, 0.000529266, 0.000507681)),
    (0.13, (0.0, -0.00194, -0.0102), (-0.0376023, 0.704057, 0.0446838, 0.707738),
     (0.000148148, 8.57757e-05, 7.71412e-05)),
]

# Bodies rigidly attached to link 6 (gripper), expressed in the link6 body frame.
# Fingers are taken at their nominal (closed) position; their 8 mm stroke is negligible.
_GRIPPER_INERTIALS = [
    (0.26, (0.0, 0.0, 0.026), (1, 0, 0, 0), (0.00016117, 0.000118, 0.00014455)),        # body
    (0.024, (0.0, -0.01, 0.064), (1, 0, 0, 0), (1.961e-6, 2.274e-6, 1.238e-6)),          # left finger
    (0.024, (0.0, 0.01, 0.064), (1, 0, 0, 0), (1.961e-6, 2.274e-6, 1.238e-6)),           # right finger
]


# ============================================================================
# Build the per-link constants (done once at import)
# ============================================================================
def _inertia_in_body_frame(quat, diagonal):
    """Inertia tensor about the COM expressed in the body frame: R diag(I) R^T."""
    R = kin._quat_to_R(*quat)
    return R @ np.diag(diagonal) @ R.T


def _lump(parts):
    """Combine rigid parts (mass, com, quat, diag) given in one frame into a single body:
    total mass, overall COM, and inertia about that COM (parallel-axis theorem)."""
    total_mass = sum(p[0] for p in parts)
    com = sum(p[0] * np.asarray(p[1], float) for p in parts) / total_mass
    inertia = np.zeros((3, 3))
    for mass, c, quat, diagonal in parts:
        r = np.asarray(c, float) - com
        inertia += _inertia_in_body_frame(quat, diagonal) + mass * (r @ r * np.eye(3) - np.outer(r, r))
    return total_mass, com, inertia


def _body_frame_offsets():
    """Constant transform DH frame i -> MuJoCo body frame of link i (i = 1..6).

    Both frames are fixed to link i, so DH_i(q)^-1 * Body_i(q) does not depend on q;
    it is evaluated at q = 0 with the XML body tree."""
    q = np.zeros(kin.N_JOINTS)
    dh_frames = kin.frames(q)
    offsets = []
    T = np.eye(4)
    for i, ((pos, quat), qi) in enumerate(zip(kin._XML_CHAIN, q)):
        Tl = np.eye(4)
        Tl[:3, :3] = kin._quat_to_R(*quat)
        Tl[:3, 3] = pos
        T = T @ Tl                                  # q = 0: no joint rotation
        offsets.append(np.linalg.inv(dh_frames[i + 1]) @ T)
    return offsets


def _build_links():
    links = []
    offsets = _body_frame_offsets()
    for i, (mass, com, quat, diagonal) in enumerate(_LINK_INERTIALS):
        parts = [(mass, com, quat, diagonal)]
        if i == kin.N_JOINTS - 1:
            parts += _GRIPPER_INERTIALS
        mass_i, com_body, inertia_body = _lump(parts)
        offset = offsets[i]
        # express everything in DH frame i once, so run-time code only uses DH frames
        com_dh = offset[:3, :3] @ com_body + offset[:3, 3]
        inertia_dh = offset[:3, :3] @ inertia_body @ offset[:3, :3].T
        links.append((mass_i, com_dh, inertia_dh))
    return links


LINKS = _build_links()          # [(mass, com in DH frame i, inertia about com in DH frame i)]
PAYLOAD_MASS = 0.0              # optional extra mass at the TCP (e.g. a grasped object), kg

# the same data as stacked arrays, for the vectorised code below
_MASSES = np.array([link[0] for link in LINKS])                  # (6,)
_COMS = np.array([link[1] for link in LINKS])                    # (6, 3)  in DH frame i
_INERTIAS = np.array([link[2] for link in LINKS])                # (6, 3, 3) in DH frame i
# _LOWER[i, k] = 1 when joint k moves link i (k <= i)
_LOWER = np.tril(np.ones((kin.N_JOINTS, kin.N_JOINTS)))


# ============================================================================
# Dynamics terms
# ============================================================================
def link_jacobians(q):
    """Jacobian columns of every link, all at once (vectorised, base frame).

    Returns (Jv, z, R):
      Jv[i, k] = z_k x (c_i - p_k) for k <= i, else 0   -> linear Jacobian of COM i, column k
      z[k]     = axis of joint k+1                      -> angular Jacobian column k (k <= i)
      R[i]     = orientation of DH frame i+1 (link i)
    """
    T = np.array(kin.frames(q))                       # (7, 4, 4)
    z = T[:kin.N_JOINTS, :3, 2]                       # (6, 3)  joint axes
    p = T[:kin.N_JOINTS, :3, 3]                       # (6, 3)  points on the axes
    R = T[1:, :3, :3]                                 # (6, 3, 3) link orientations
    c = T[1:, :3, 3] + np.einsum("iab,ib->ia", R, _COMS)          # (6, 3) COM positions
    Jv = np.cross(z[None, :, :], c[:, None, :] - p[None, :, :])  # (6 links, 6 joints, 3)
    Jv *= _LOWER[:, :, None]
    return Jv, z, R


def gravity_torque(q, payload_mass=None):
    """G(q): joint torques that exactly balance gravity (N*m).

    G_k = g0 * sum_i m_i * (J_v,ci)_z,k   (z component of the COM Jacobians)."""
    payload_mass = PAYLOAD_MASS if payload_mass is None else payload_mass
    Jv, _, _ = link_jacobians(q)
    G = GRAVITY * (_MASSES @ Jv[:, :, 2])
    if payload_mass > 0.0:
        G += payload_mass * GRAVITY * kin.jacobian(q)[2]
    return G


def mass_matrix(q, include_armature=True):
    """M(q): 6x6 joint-space inertia matrix (kg m^2).

    M_kl = sum_i m_i Jv_ik . Jv_il  +  sum_{i >= max(k,l)} z_k^T (R_i I_i R_i^T) z_l  (+ armature)."""
    Jv, z, R = link_jacobians(q)
    M = np.einsum("i,ika,ila->kl", _MASSES, Jv, Jv)
    inertia_world = np.einsum("iab,ibc,idc->iad", R, _INERTIAS, R)      # R I R^T per link
    rotational = np.einsum("ka,iab,lb->ikl", z, inertia_world, z)       # (link, k, l)
    mask = _LOWER[:, :, None] * _LOWER[:, None, :]                       # k <= i and l <= i
    M += (rotational * mask).sum(axis=0)
    if include_armature:
        M += np.diag(ARMATURE)
    return M


def gravity_and_mass(q, payload_mass=None):
    """(G(q), M(q)) sharing one kinematics pass: what the controller calls every tick."""
    payload_mass = PAYLOAD_MASS if payload_mass is None else payload_mass
    Jv, z, R = link_jacobians(q)
    G = GRAVITY * (_MASSES @ Jv[:, :, 2])
    M = np.einsum("i,ika,ila->kl", _MASSES, Jv, Jv)
    inertia_world = np.einsum("iab,ibc,idc->iad", R, _INERTIAS, R)
    rotational = np.einsum("ka,iab,lb->ikl", z, inertia_world, z)
    M += (rotational * (_LOWER[:, :, None] * _LOWER[:, None, :])).sum(axis=0)
    M += np.diag(ARMATURE)
    if payload_mass > 0.0:
        J = kin.jacobian(q)
        G += payload_mass * GRAVITY * J[2]
        M += payload_mass * J[:3].T @ J[:3]
    return G, M


# ============================================================================
# Validation against MuJoCo
# ============================================================================
def compare_with_mujoco(xml_path, n=50, seed=0, verbose=True):
    """G(q) vs MuJoCo qfrc_bias (zero velocity) and M(q) vs mj_fullM, random poses."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 7)]
    qadr = [int(model.jnt_qposadr[j]) for j in ids]
    dadr = [int(model.jnt_dofadr[j]) for j in ids]
    rng = np.random.default_rng(seed)
    worst_g, worst_m, largest_g = 0.0, 0.0, 0.0
    full = np.zeros((model.nv, model.nv))
    for _ in range(n):
        q = rng.uniform(kin.JOINT_LOWER, kin.JOINT_UPPER) * 0.8
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        g_sim = data.qfrc_bias[dadr]
        mujoco.mj_fullM(model, data, full)          # MuJoCo 3.13 signature: (model, data, dst)
        m_sim = full[np.ix_(dadr, dadr)]
        worst_g = max(worst_g, np.abs(gravity_torque(q) - g_sim).max())
        worst_m = max(worst_m, np.abs(mass_matrix(q) - m_sim).max())
        largest_g = max(largest_g, np.abs(g_sim).max())
    ok = worst_g < 0.02 and worst_m < 1e-3
    if verbose:
        print(f"{'PASS' if worst_g < 0.02 else 'FAIL'}  gravity G(q)    worst error = {worst_g:.2e} N*m "
              f"(largest |G| = {largest_g:.2f} N*m)")
        print(f"{'PASS' if worst_m < 1e-3 else 'FAIL'}  mass matrix M(q) worst error = {worst_m:.2e} kg m^2")
    return ok


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python3 dynamics.py path/to/lab-scene.xml")
        raise SystemExit(2)
    raise SystemExit(0 if compare_with_mujoco(sys.argv[1]) else 1)
