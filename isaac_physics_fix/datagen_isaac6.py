"""Teacher->student data generation in Isaac Sim 6.0: roll out the robust privileged RL
teacher and log VISION demonstrations — wrist (eye-in-hand) + scene RGB, proprioception,
and the teacher's action — so a vision policy / VLA can be trained to open the drawer from
pixels (no privileged state, no hardcoded base pose at inference).

Output per successful episode -> /tmp/molmo_demos/ep_XXXX.npz
  wrist_rgb [T,H,W,3] u8, scene_rgb [T,H,W,3] u8, joint_pos [T,9], joint_vel [T,9],
  action [T,9], ee_pose [T,7] (pos+quat wxyz), drawer_pos [T], success bool

Usage: ~/isaacsim6/sim/python.sh datagen_isaac6.py [N_EPISODES]
"""
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 5
RES = 224
OUTDIR = "/tmp/molmo_demos"

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
import omni.replicator.core as rep
import imageio.v2 as imageio
from isaacsim.core.api import SimulationContext
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

DRAWER_USD = os.path.expanduser("~/assets_rl/molmo_drawer.usda")
FRANKA_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
)
WEIGHTS = os.path.expanduser("~/rl_logs/molmo_drawer/policy_weights.npz")
GRASP_LOCAL = np.array([0.0, 0.015, 0.037])
DRAWER_TRAVEL = 0.319
ACTION_SCALE, DOF_VEL_SCALE, DECIMATION, PHYS_DT = 7.5, 0.1, 2, 1.0 / 120.0
DT = PHYS_DT * DECIMATION
INIT_Q = {"panda_joint1": 1.157, "panda_joint2": -1.066, "panda_joint3": -0.155,
          "panda_joint4": -2.239, "panda_joint5": -1.841, "panda_joint6": 1.003,
          "panda_joint7": 0.469, "panda_finger_joint1": 0.035, "panda_finger_joint2": 0.035}


def log(*a):
    print(">>>", *a, flush=True)


# ---- quat helpers (w,x,y,z) ----
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qrot(q, v):
    return qmul(qmul(q, np.array([0.0, v[0], v[1], v[2]])), qconj(q))[1:]


def tf_combine(q1, t1, q2, t2):
    return qmul(q1, q2), t1 + qrot(q1, t2)


def tf_inverse(q, t):
    qi = qconj(q)
    return qi, -qrot(qi, t)


def x7(x):
    return np.array(x[0:3]), np.array([x[6], x[3], x[4], x[5]])


def mat_to_quat(M):
    t = np.trace(M)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (M[2, 1] - M[1, 2]) / s
        y = (M[0, 2] - M[2, 0]) / s
        z = (M[1, 0] - M[0, 1]) / s
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        w = (M[2, 1] - M[1, 2]) / s
        x = 0.25 * s
        y = (M[0, 1] + M[1, 0]) / s
        z = (M[0, 2] + M[2, 0]) / s
    elif M[1, 1] > M[2, 2]:
        s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        w = (M[0, 2] - M[2, 0]) / s
        x = (M[0, 1] + M[1, 0]) / s
        y = 0.25 * s
        z = (M[1, 2] + M[2, 1]) / s
    else:
        s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
        w = (M[1, 0] - M[0, 1]) / s
        x = (M[0, 2] + M[2, 0]) / s
        y = (M[1, 2] + M[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def look_at_quat(eye, target, up=np.array([0.0, 0.0, 1.0])):
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    r = r / (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, f)
    M = np.stack([r, u, -f], axis=1)  # camera looks down -z
    return mat_to_quat(M)


# ---- scene ----
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
fx.AddOrientOp().Set(Gf.Quatf(0.0, 0.0, 0.0, 1.0))
blk = UsdGeom.Cube.Define(stage, "/World/Blocker")
blk.GetPrim().GetAttribute("size").Set(2.0)
bx = UsdGeom.Xformable(blk)
bx.AddTranslateOp().Set(Gf.Vec3d(-0.20, 0.0, 0.633))
bx.AddScaleOp().Set(Gf.Vec3f(0.215, 0.28, 0.015))
UsdPhysics.CollisionAPI.Apply(blk.GetPrim())

# robust physics config (solver iters + friction, matching training)
_pa = PhysxSchema.PhysxArticulationAPI.Apply(franka)
_pa.CreateSolverPositionIterationCountAttr(12)
_pa.CreateSolverVelocityIterationCountAttr(1)
_pm = UsdShade.Material.Define(stage, "/World/PhysMat")
_pmapi = UsdPhysics.MaterialAPI.Apply(_pm.GetPrim())
_pmapi.CreateStaticFrictionAttr(1.0)
_pmapi.CreateDynamicFrictionAttr(1.0)
_pmapi.CreateRestitutionAttr(0.0)
for _root in ("/World/Drawer", "/World/Ground", "/World/Blocker"):
    for _p in Usd.PrimRange(stage.GetPrimAtPath(_root)):
        if _p.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(_p).Bind(
                _pm, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics")

# cameras: fixed scene cam + eye-in-hand wrist cam (pose updated each step)
scene_cam = UsdGeom.Camera.Define(stage, "/World/SceneCam")
UsdGeom.Xformable(scene_cam).AddTranslateOp().Set(Gf.Vec3d(1.7, -1.25, 1.05))
_sq = look_at_quat(np.array([1.7, -1.25, 1.05]), np.array([0.1, 0.0, 0.45]))
UsdGeom.Xformable(scene_cam).AddOrientOp().Set(Gf.Quatf(*[float(v) for v in _sq]))
wrist_cam = UsdGeom.Camera.Define(stage, "/World/WristCam")
wrist_cam.CreateFocalLengthAttr(12.0)
_wt = UsdGeom.Xformable(wrist_cam)
_w_pos = _wt.AddTranslateOp()
_w_rot = _wt.AddOrientOp()

sim = SimulationContext(physics_dt=PHYS_DT, rendering_dt=DT, backend="numpy")
sim.reset()
for _ in range(40):
    sim.step(render=False)

import omni.physics.tensors as tensors

sv = tensors.create_simulation_view("numpy", stage_id=omni.usd.get_context().get_stage_id())
sv.set_subspace_roots("/")
robot_av = sv.create_articulation_view("/World/Franka")
drawer_av = sv.create_articulation_view("/World/Drawer/*")
rdn = list(robot_av.shared_metatype.dof_names) if hasattr(robot_av, "shared_metatype") else None
rln = list(robot_av.shared_metatype.link_names)
dln = list(drawer_av.shared_metatype.link_names)
stiff = np.array([2000.0 if "finger" in n else 80.0 for n in rdn])[None, :]
damp = np.array([100.0 if "finger" in n else 4.0 for n in rdn])[None, :]
mforce = np.array([200.0 if "finger" in n else (87.0 if int(n[-1]) <= 4 else 12.0) for n in rdn])[None, :]
robot_av.set_dof_stiffnesses(stiff, np.array([0]))
robot_av.set_dof_dampings(damp, np.array([0]))
robot_av.set_dof_max_forces(mforce, np.array([0]))
li7, lih = rln.index("panda_link7"), rln.index("panda_hand")
lil, lir = rln.index("panda_leftfinger"), rln.index("panda_rightfinger")
slider_idx = [i for i, n in enumerate(dln) if n.endswith("_1_1_0")][0]
lim = robot_av.get_dof_limits()[0]
lower, upper = lim[:, 0], lim[:, 1]
speed_scales = np.array([0.1 if "finger" in n else 1.0 for n in rdn])
init_q = np.array([INIT_Q[n] for n in rdn])[None, :]

# render products
rp_scene = rep.create.render_product("/World/SceneCam", (RES, RES))
rp_wrist = rep.create.render_product("/World/WristCam", (RES, RES))
ann_s = rep.AnnotatorRegistry.get_annotator("rgb")
ann_w = rep.AnnotatorRegistry.get_annotator("rgb")
ann_s.attach([rp_scene])
ann_w.attach([rp_wrist])

W = np.load(WEIGHTS)
w0, b0, w2, b2, w4, b4, w6, b6 = (W["w0"], W["b0"], W["w2"], W["b2"], W["w4"], W["b4"], W["w6"], W["b6"])
nmean, nstd = W["norm_mean"][0], W["norm_std"][0]


def elu(x):
    return np.where(x > 0, x, np.exp(np.clip(x, -20, 0)) - 1)


def policy(o):
    o = (o - nmean) / nstd
    h = elu(w0 @ o + b0)
    h = elu(w2 @ h + b2)
    h = elu(w4 @ h + b4)
    return w6 @ h + b6


def link_pq(av, idx):
    return x7(av.get_link_transforms()[0, idx])


hand_p, hand_q = link_pq(robot_av, li7)
lf_p, _ = link_pq(robot_av, lil)
rf_p, lf_q = (link_pq(robot_av, lir)[0], link_pq(robot_av, lil)[1])
finger_p = (link_pq(robot_av, lil)[0] + link_pq(robot_av, lir)[0]) / 2.0
hinv_q, hinv_t = tf_inverse(hand_q, hand_p)
rlg_q, rlg_t = tf_combine(hinv_q, hinv_t, link_pq(robot_av, lil)[1], finger_p)
rlg_t = rlg_t + np.array([0, 0.04, 0])
drawer_lg_q = np.array([1.0, 0.0, 0.0, 0.0])

# eye-in-hand offset in the panda_hand frame (behind + above, looking toward the grasp)
WRIST_EYE_LOCAL = np.array([0.0, -0.04, -0.12])
WRIST_TGT_LOCAL = np.array([0.0, 0.02, 0.10])

robot_dof_targets = np.zeros(9)


def compute_obs():
    jp = robot_av.get_dof_positions()[0]
    jv = robot_av.get_dof_velocities()[0]
    djp = drawer_av.get_dof_positions()[0, 0]
    djv = drawer_av.get_dof_velocities()[0, 0]
    h_p, h_q = link_pq(robot_av, li7)
    s_p, s_q = link_pq(drawer_av, slider_idx)
    _, rg = tf_combine(h_q, h_p, rlg_q, rlg_t)
    _, dg = tf_combine(s_q, s_p, drawer_lg_q, GRASP_LOCAL)
    obs = np.concatenate([2.0 * (jp - lower) / (upper - lower) - 1.0, jv * DOF_VEL_SCALE,
                          dg - rg, [djp], [djv]])
    return np.clip(obs, -5.0, 5.0), jp, jv, djp, rg


def update_wrist_cam(grasp_p):
    # eye-in-hand: mount just behind+above the gripper, look at the grasp point (fingertips)
    hp, _ = link_pq(robot_av, lih)
    fdir = grasp_p - hp
    fdir = fdir / (np.linalg.norm(fdir) + 1e-9)
    eye = hp - 0.09 * fdir + np.array([0.0, 0.0, 0.07])
    tgt = grasp_p + 0.03 * fdir
    q = look_at_quat(eye, tgt)
    _w_pos.Set(Gf.Vec3d(*[float(v) for v in eye]))
    _w_rot.Set(Gf.Quatf(*[float(v) for v in q]))


def grab(ann):
    arr = np.asarray(ann.get_data())
    if arr.ndim == 3 and arr.shape[0] == RES and arr.shape[1] == RES and arr.shape[2] >= 3:
        return arr[..., :3].astype(np.uint8)
    return np.zeros((RES, RES, 3), np.uint8)


def reset_ep():
    global robot_dof_targets
    robot_av.set_dof_positions(init_q.copy(), np.array([0]))
    robot_av.set_dof_velocities(np.zeros((1, 9)), np.array([0]))
    drawer_av.set_dof_positions(np.zeros((1, 1)), np.array([0]))
    robot_av.set_dof_position_targets(init_q.copy(), np.array([0]))
    robot_dof_targets = np.zeros(9)
    for _ in range(5):
        sim.step(render=False)


os.makedirs(OUTDIR, exist_ok=True)
# prime the renderer/annotators so the first frames are valid
reset_ep()
for _ in range(8):
    update_wrist_cam(compute_obs()[4])
    sim.step(render=False)
    sim.step(render=True)
n_ok = 0
for ep in range(N_EP):
    reset_ep()
    W_rgb, S_rgb, JP, JV, AC, EE, DR = [], [], [], [], [], [], []
    maxd = 0.0
    for s in range(420):
        obs, jp, jv, djp, rg = compute_obs()
        maxd = max(maxd, djp)
        a = np.clip(policy(obs), -1.0, 1.0)
        robot_dof_targets = np.clip(robot_dof_targets + speed_scales * DT * a * ACTION_SCALE, lower, upper)
        robot_av.set_dof_position_targets(robot_dof_targets[None, :], np.array([0]))
        update_wrist_cam(rg)
        sim.step(render=False)
        sim.step(render=True)
        hp, hq = link_pq(robot_av, lih)
        wf, sf = grab(ann_w), grab(ann_s)
        if ep == 0 and s in (100, 250, 380):
            imageio.imwrite(f"/tmp/molmo_demos/wrist_{s}.png", wf)
        W_rgb.append(wf)
        S_rgb.append(sf)
        JP.append(jp.copy()); JV.append(jv.copy()); AC.append(a.copy())
        EE.append(np.concatenate([hp, hq])); DR.append(djp)
    success = maxd > 0.20
    if ep == 0:
        imageio.imwrite("/tmp/molmo_demos/sample_wrist.png", W_rgb[len(W_rgb) // 2])
        imageio.imwrite("/tmp/molmo_demos/sample_scene.png", S_rgb[len(S_rgb) // 2])
    if success:
        np.savez_compressed(
            f"{OUTDIR}/ep_{n_ok:04d}.npz",
            wrist_rgb=np.array(W_rgb), scene_rgb=np.array(S_rgb),
            joint_pos=np.array(JP), joint_vel=np.array(JV), action=np.array(AC),
            ee_pose=np.array(EE), drawer_pos=np.array(DR), success=success,
        )
        n_ok += 1
    log(f"episode {ep}: max drawer {maxd:.3f} ({maxd/DRAWER_TRAVEL*100:.0f}%) success={success} T={len(W_rgb)}")

log(f"DATAGEN DONE: {n_ok}/{N_EP} successful demos saved to {OUTDIR}")
app.close()
os._exit(0)
