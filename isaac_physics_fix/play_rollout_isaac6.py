"""Replay a recorded RL rollout (record_molmo_rollout.py -> rollout.npz) in Isaac Sim 6.0:
Franka + MolmoSpaces drawer, driving the robot joints through the recorded trajectory so
the gripper physically hooks and pulls the drawer open. Two modes:

  snapshot : default kit, render check frames to /tmp/molmo_isaac6/replay_check_*.png, exit
  stream   : WebRTC livestream (full UI) — connect with the streaming client and watch; loops forever

Usage: ~/isaacsim6/sim/python.sh play_rollout_isaac6.py [snapshot|stream] [--assist]
"""
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
MODE = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
ASSIST = "--assist" in sys.argv
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
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

DRAWER_USD = os.path.expanduser("~/assets_rl/molmo_drawer.usda")
FRANKA_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
)
ROLLOUT = os.path.expanduser("~/rl_logs/molmo_drawer/rollout.npz")

rec = np.load(ROLLOUT, allow_pickle=True)
rob_traj, drawer_traj, names = rec["robot"], rec["drawer"], [str(n) for n in rec["names"]]
print(f">>> rollout: {rob_traj.shape[0]} steps, joints: {names}", flush=True)

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

# ground (top surface at z=0)
ground = UsdGeom.Cube.Define(stage, "/World/Ground")
ground.GetPrim().GetAttribute("size").Set(20.0)
UsdGeom.Xformable(ground).AddTranslateOp().Set(Gf.Vec3d(0, 0, -10.0))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1500.0)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(3000.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40, 20, 0))

# drawer at origin, rotated -90deg about Z (same pose as the RL env)
drawer = stage.DefinePrim("/World/Drawer")
drawer.GetReferences().AddReference(DRAWER_USD)
dx = UsdGeom.Xformable(drawer)
dx.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
dx.AddOrientOp().Set(Gf.Quatf(0.7071068, 0.0, 0.0, -0.7071068))

# Franka at (0.62, 0, 0), rotated 180deg about Z (same pose as the RL env)
franka = stage.DefinePrim("/World/Franka")
franka.GetReferences().AddReference(FRANKA_USD)
fx = UsdGeom.Xformable(franka)
fx.AddTranslateOp().Set(Gf.Vec3d(0.62, 0, 0))
fx.AddOrientOp().Set(Gf.Quatf(0.0, 0.0, 0.0, 1.0))

# map joint name -> (prim, is_angular); author stiff drives + initial joint state
joints = {}
for p in stage.Traverse():
    if not str(p.GetPath()).startswith("/World/Franka"):
        continue
    if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in names:
        joints[p.GetName()] = (p, True)
    elif p.IsA(UsdPhysics.PrismaticJoint) and p.GetName() in names:
        joints[p.GetName()] = (p, False)
print(f">>> matched {len(joints)}/{len(names)} robot joints in USD", flush=True)

q0 = rob_traj[0]
for i, nm in enumerate(names):
    if nm not in joints:
        continue
    prim, ang = joints[nm]
    tok = "angular" if ang else "linear"
    d = UsdPhysics.DriveAPI.Apply(prim, tok)
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(2000.0 if ang else 50000.0)
    d.CreateDampingAttr(200.0 if ang else 2000.0)
    d.CreateMaxForceAttr(1.0e5)
    val = float(np.degrees(q0[i])) if ang else float(q0[i])
    d.CreateTargetPositionAttr(val)
    try:
        from pxr import PhysxSchema
        js = PhysxSchema.JointStateAPI.Apply(prim, tok)
        js.CreatePositionAttr(val)
    except Exception:
        pass  # warmup at q0 drive targets settles the arm anyway

# optional drawer assist (drive the drawer joint along its recorded trajectory)
drawer_drive = None
if ASSIST:
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.PrismaticJoint) and "drawer" in p.GetName().lower():
            drawer_drive = UsdPhysics.DriveAPI.Apply(p, "linear")
            drawer_drive.CreateStiffnessAttr(500.0)
            drawer_drive.CreateDampingAttr(50.0)
            drawer_drive.CreateTargetPositionAttr(0.0)
            print(">>> drawer assist drive enabled", flush=True)

if MODE == "snapshot":
    import imageio.v2 as imageio
    import omni.replicator.core as rep

    cam = rep.create.camera(position=(1.7, -1.3, 1.1), look_at=(0.15, 0.0, 0.5), focal_length=18.0)
    rp = rep.create.render_product(cam, (1280, 720))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])

timeline = omni.timeline.get_timeline_interface()


def run_once(capture=False):
    frames = {}
    timeline.play()
    for _ in range(40):  # settle at initial pose
        app.update()
    n = rob_traj.shape[0]
    for s in range(n):
        for i, nm in enumerate(names):
            if nm not in joints:
                continue
            prim, ang = joints[nm]
            tok = "angular" if ang else "linear"
            val = float(np.degrees(rob_traj[s][i])) if ang else float(rob_traj[s][i])
            UsdPhysics.DriveAPI(prim, tok).GetTargetPositionAttr().Set(val)
        if drawer_drive is not None:
            drawer_drive.GetTargetPositionAttr().Set(float(drawer_traj[s]))
        app.update()
        if capture and s in (5, int(n * 0.55), n - 2):
            arr = np.asarray(rgb.get_data())
            frames[s] = arr[..., :3].astype(np.uint8)
            print(f">>> captured frame at step {s}", flush=True)
    for _ in range(90):  # hold final
        app.update()
    timeline.stop()
    for _ in range(10):
        app.update()
    return frames


if MODE == "snapshot":
    os.makedirs("/tmp/molmo_isaac6", exist_ok=True)
    frames = run_once(capture=True)
    for s, f in frames.items():
        path = f"/tmp/molmo_isaac6/replay_check_{s:03d}.png"
        imageio.imwrite(path, f)
        print(f">>> wrote {path}", flush=True)
    print(">>> SNAPSHOT_OK", flush=True)
    app.close()
    os._exit(0)
else:
    # frame the viewport camera on the robot+drawer (a fresh stage's default camera may
    # point at nothing -> black/empty view for the streaming client)
    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(eye=[1.9, -1.5, 1.3], target=[0.15, 0.0, 0.5])
        print(">>> viewport camera framed", flush=True)
    except Exception as e:  # noqa: BLE001
        print(">>> camera set failed (navigate manually with mouse):", e, flush=True)
    print(">>> STREAM READY — connect the WebRTC client to " + PUBLIC_IP, flush=True)
    loop = 0
    while app.is_running():
        loop += 1
        print(f">>> replay loop {loop}", flush=True)
        run_once(capture=False)
