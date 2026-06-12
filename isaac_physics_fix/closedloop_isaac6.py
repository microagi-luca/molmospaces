"""LIVE closed-loop policy in Isaac Sim 6.0 (native, no Isaac Lab): the trained PPO policy
runs frame-by-frame on real GPU-PhysX, the Franka physically grabs and pulls the MolmoSpaces
drawer. snapshot mode renders check frames + measures the drawer joint; stream mode serves
the live view over WebRTC.

Policy = numpy MLP (weights from policy_weights.npz). Articulations driven via PhysX tensors.
Replicates molmo_drawer env obs(23)/action(9) exactly.

Usage: ~/isaacsim6/sim/python.sh closedloop_isaac6.py [snapshot|stream]
"""
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
MODE = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
PUBLIC_IP = "35.246.20.66"

from isaacsim import SimulationApp

if MODE == "stream":
    exp = os.path.expanduser("~/isaacsim6/sim/apps/isaacsim.exp.full.streaming.kit")
    app = SimulationApp(
        {
            "headless": True,
            "extra_args": [
                f"--/exts/omni.kit.livestream.app/primaryStream/publicIp={PUBLIC_IP}",
                "--/exts/omni.kit.livestream.app/primaryStream/signalPort=49100",
                "--/exts/omni.kit.livestream.app/primaryStream/streamPort=47998",
            ],
        },
        experience=exp,
    )
else:
    app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from isaacsim.core.api import SimulationContext
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

DRAWER_USD = os.path.expanduser("~/assets_rl/molmo_drawer.usda")
FRANKA_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
)
WEIGHTS = os.path.expanduser("~/rl_logs/molmo_drawer/policy_weights.npz")
ROLLOUT = os.path.expanduser("~/rl_logs/molmo_drawer/rollout.npz")  # for training joint order
SLIDER_LINK = "drawer_372d9ee41d70550432c30a66a6e5b331_1_1_0"
GRASP_LOCAL = np.array([0.0, 0.015, 0.037])
DRAWER_TRAVEL = 0.319
ACTION_SCALE = 7.5
DOF_VEL_SCALE = 0.1
DECIMATION = 2
PHYS_DT = 1.0 / 120.0
DT = PHYS_DT * DECIMATION

# Franka init joint targets (env init_state), by joint name
INIT_Q = {
    "panda_joint1": 1.157, "panda_joint2": -1.066, "panda_joint3": -0.155, "panda_joint4": -2.239,
    "panda_joint5": -1.841, "panda_joint6": 1.003, "panda_joint7": 0.469,
    "panda_finger_joint1": 0.035, "panda_finger_joint2": 0.035,
}


def log(*a):
    print(">>>", *a, flush=True)


# ---------- quaternion helpers, (w,x,y,z) ----------
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qrot(q, v):
    return qmul(qmul(q, np.array([0.0, v[0], v[1], v[2]])), qconj(q))[1:]


def tf_combine(q1, t1, q2, t2):
    return qmul(q1, q2), t1 + qrot(q1, t2)


def tf_inverse(q, t):
    qi = qconj(q)
    return qi, -qrot(qi, t)


def xform7_to_pq(x):
    # PhysX tensors link transform: [px,py,pz, qx,qy,qz,qw] -> pos, quat(wxyz)
    return np.array(x[0:3]), np.array([x[6], x[3], x[4], x[5]])


# ---------- build scene (training frame) ----------
ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

ground = UsdGeom.Cube.Define(stage, "/World/Ground")
ground.GetPrim().GetAttribute("size").Set(20.0)
UsdGeom.Xformable(ground).AddTranslateOp().Set(Gf.Vec3d(0, 0, -10.0))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1200.0)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(3000.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40, 15, 0))

drawer = stage.DefinePrim("/World/Drawer")
drawer.GetReferences().AddReference(DRAWER_USD)
dx = UsdGeom.Xformable(drawer)
dx.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
dx.AddOrientOp().Set(Gf.Quatf(0.7071068, 0.0, 0.0, -0.7071068))

franka = stage.DefinePrim("/World/Franka")
franka.GetReferences().AddReference(FRANKA_USD)
fx = UsdGeom.Xformable(franka)
fx.AddTranslateOp().Set(Gf.Vec3d(0.62, 0, 0))
fx.AddOrientOp().Set(Gf.Quatf(0.0, 0.0, 0.0, 1.0))  # Rz(180)

# occluder plate over the drawer top (matches training; forces front-handle approach)
blk = UsdGeom.Cube.Define(stage, "/World/Blocker")
blk.GetPrim().GetAttribute("size").Set(2.0)
bx = UsdGeom.Xformable(blk)
bx.AddTranslateOp().Set(Gf.Vec3d(-0.20, 0.0, 0.633))
bx.AddScaleOp().Set(Gf.Vec3f(0.215, 0.28, 0.015))
UsdPhysics.CollisionAPI.Apply(blk.GetPrim())

# ---------- physics ----------
sim = SimulationContext(physics_dt=PHYS_DT, rendering_dt=DT, backend="numpy")
sim.reset()
log("SimulationContext ready, device:", sim.get_physics_dt())

import omni.physics.tensors as tensors

_stage_id = omni.usd.get_context().get_stage_id()
sv = tensors.create_simulation_view("numpy", stage_id=_stage_id)
sv.set_subspace_roots("/")
robot_av = sv.create_articulation_view("/World/Franka")
drawer_av = sv.create_articulation_view("/World/Drawer/*")

# discover dof / link ordering
def names_of(av, kind):
    for attr in ((".shared_metatype." + kind), (".metatype." + kind), ("." + kind)):
        try:
            obj = av
            for p in attr.strip(".").split("."):
                obj = getattr(obj, p)
            return list(obj)
        except Exception:
            continue
    return None

robot_dof_names = names_of(robot_av, "dof_names")
robot_link_names = names_of(robot_av, "link_names")
drawer_dof_names = names_of(drawer_av, "dof_names")
drawer_link_names = names_of(drawer_av, "link_names")
log("robot dof_names (physx order):", robot_dof_names)
log("robot link_names:", robot_link_names)
log("drawer dof_names:", drawer_dof_names)
log("drawer link_names:", drawer_link_names)

# training joint order (what the policy expects)
train_names = [str(n) for n in np.load(ROLLOUT, allow_pickle=True)["names"]]
log("training joint order:", train_names)

# map physx dof index -> training index, and back
p2t = np.array([robot_dof_names.index(n) for n in train_names])  # train[i] = physx[p2t[i]]
t2p = np.array([train_names.index(n) for n in robot_dof_names])  # physx[i] = train[t2p[i]]

li7 = robot_link_names.index("panda_link7")
lil = robot_link_names.index("panda_leftfinger")
lir = robot_link_names.index("panda_rightfinger")
slider_idx = [i for i, n in enumerate(drawer_link_names) if n.endswith("_1_1_0")][0]

# dof limits in training order
lim = robot_av.get_dof_limits()[0]  # [physx_dof, 2]
lower = lim[p2t, 0]
upper = lim[p2t, 1]
speed_scales = np.ones(9)
for i, n in enumerate(train_names):
    if "finger" in n:
        speed_scales[i] = 0.1

# ---------- match Isaac Lab actuator gains (the policy trained with THESE, not the
# Franka USD's default drives) ----------
def _maxf(n):
    if "finger" in n:
        return 200.0
    return 87.0 if int(n[-1]) <= 4 else 12.0

stiff = np.array([2000.0 if "finger" in n else 80.0 for n in robot_dof_names])[None, :]
damp = np.array([100.0 if "finger" in n else 4.0 for n in robot_dof_names])[None, :]
mforce = np.array([_maxf(n) for n in robot_dof_names])[None, :]
robot_av.set_dof_stiffnesses(stiff, np.array([0]))
robot_av.set_dof_dampings(damp, np.array([0]))
robot_av.set_dof_max_forces(mforce, np.array([0]))
log("set drive gains: stiff", stiff[0].tolist(), "damp", damp[0].tolist())

# ---------- reset robot to init pose ----------
init_q_physx = np.array([INIT_Q[n] for n in robot_dof_names])[None, :]
# set joint state via physx tensors (positions); fall back to targets
try:
    robot_av.set_dof_positions(init_q_physx, np.array([0]))
except Exception as e:  # noqa: BLE001
    log("set_dof_positions unavailable:", e)
robot_av.set_dof_position_targets(init_q_physx.copy(), np.array([0]))
for _ in range(30):
    sim.step(render=False)

# robot_dof_targets accumulator (env starts at zeros, in training order)
robot_dof_targets_t = np.zeros(9)

# ---------- policy (numpy) ----------
W = np.load(WEIGHTS)
w0, b0, w2, b2, w4, b4, w6, b6 = (W["w0"], W["b0"], W["w2"], W["b2"], W["w4"], W["b4"], W["w6"], W["b6"])
nmean, nstd = W["norm_mean"][0], W["norm_std"][0]


def elu(x):
    return np.where(x > 0, x, np.exp(np.clip(x, -20, 0)) - 1)


def policy(obs):
    o = (obs - nmean) / nstd
    h = elu(w0 @ o + b0)
    h = elu(w2 @ h + b2)
    h = elu(w4 @ h + b4)
    return w6 @ h + b6


# ---------- grasp frame init (replicate env __init__) ----------
def link_pq(av, idx):
    return xform7_to_pq(av.get_link_transforms()[0, idx])


hand_p, hand_q = link_pq(robot_av, li7)
lf_p, lf_q = link_pq(robot_av, lil)
rf_p, rf_q = link_pq(robot_av, lir)
finger_p = (lf_p + rf_p) / 2.0
finger_q = lf_q
hinv_q, hinv_t = tf_inverse(hand_q, hand_p)
rlg_q, rlg_t = tf_combine(hinv_q, hinv_t, finger_q, finger_p)
rlg_t = rlg_t + np.array([0, 0.04, 0])
log("robot_local_grasp_pos:", np.round(rlg_t, 4).tolist())

drawer_local_grasp_q = np.array([1.0, 0.0, 0.0, 0.0])


def compute_obs():
    jp_p = robot_av.get_dof_positions()[0]
    jv_p = robot_av.get_dof_velocities()[0]
    jp = jp_p[p2t]
    jv = jv_p[p2t]
    djp = drawer_av.get_dof_positions()[0, 0]
    djv = drawer_av.get_dof_velocities()[0, 0]
    h_p, h_q = link_pq(robot_av, li7)
    s_p, s_q = link_pq(drawer_av, slider_idx)
    _, rg = tf_combine(h_q, h_p, rlg_q, rlg_t)
    _, dg = tf_combine(s_q, s_p, drawer_local_grasp_q, GRASP_LOCAL)
    to_target = dg - rg
    dof_scaled = 2.0 * (jp - lower) / (upper - lower) - 1.0
    obs = np.concatenate([dof_scaled, jv * DOF_VEL_SCALE, to_target, [djp], [djv]])
    return np.clip(obs, -5.0, 5.0), djp


# ---------- snapshot camera ----------
if MODE == "snapshot":
    import imageio.v2 as imageio
    import omni.replicator.core as rep

    os.makedirs("/tmp/molmo_isaac6", exist_ok=True)
    cam = rep.create.camera(position=(1.7, -1.25, 1.05), look_at=(0.1, 0.0, 0.45), focal_length=18.0)
    rp = rep.create.render_product(cam, (1280, 720))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])

if MODE == "stream":
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=[1.7, -1.25, 1.05], target=[0.1, 0.0, 0.45])
    except Exception as e:  # noqa: BLE001
        log("camera set failed:", e)
    log("STREAM READY — connect WebRTC client to " + PUBLIC_IP)


def reset_episode():
    global robot_dof_targets_t
    try:
        robot_av.set_dof_positions(init_q_physx.copy(), np.array([0]))
        robot_av.set_dof_velocities(np.zeros((1, 9)), np.array([0]))
        drawer_av.set_dof_positions(np.zeros((1, 1)), np.array([0]))
    except Exception:
        pass
    robot_av.set_dof_position_targets(init_q_physx.copy(), np.array([0]))
    robot_dof_targets_t = np.zeros(9)
    for _ in range(5):
        sim.step(render=False)


def run_episode(capture=False):
    global robot_dof_targets_t
    frames = {}
    maxd = 0.0
    n = 500
    for s in range(n):
        obs, djp = compute_obs()
        maxd = max(maxd, djp)
        a = np.clip(policy(obs), -1.0, 1.0)
        robot_dof_targets_t = np.clip(
            robot_dof_targets_t + speed_scales * DT * a * ACTION_SCALE, lower, upper
        )
        tgt_physx = robot_dof_targets_t[t2p][None, :]
        robot_av.set_dof_position_targets(tgt_physx, np.array([0]))
        sim.step(render=False)
        sim.step(render=True)
        if s in (0, 30, 80, 150, 260, 399, 499):
            tt = obs[18:21]
            log(f"step {s}: drawer={djp:.4f} ({djp/DRAWER_TRAVEL*100:.0f}%) "
                f"|to_target|={np.linalg.norm(tt):.3f} |a|={np.linalg.norm(a):.2f}")
        if capture and s in (5, 150, 260, n - 2):
            arr = np.asarray(rgb.get_data())
            frames[s] = arr[..., :3].astype(np.uint8)
    return frames, maxd


if MODE == "snapshot":
    reset_episode()
    frames, maxd = run_episode(capture=True)
    for s, f in frames.items():
        imageio.imwrite(f"/tmp/molmo_isaac6/cl_{s:03d}.png", f)
        log(f"wrote cl_{s:03d}.png")
    log(f"CLOSED-LOOP RESULT: max drawer {maxd:.4f} m = {maxd / DRAWER_TRAVEL * 100:.1f}% of travel")
    log("SNAPSHOT_OK")
    app.close()
    os._exit(0)
else:
    loop = 0
    while app.is_running():
        loop += 1
        log(f"episode {loop}")
        reset_episode()
        run_episode(capture=False)
