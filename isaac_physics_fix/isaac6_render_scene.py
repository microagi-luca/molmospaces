"""Isaac Sim 6.0 (standalone) headless RTX render: load a converted MolmoSpaces scene,
drive drawer/cabinet joints open via authored DriveAPI targets, capture RGB frames with
Replicator, write an MP4. Native isaacsim API only (no Isaac Lab).

Usage: python.sh isaac6_render_scene.py <scene.usda> <out.mp4> [eye_x eye_y eye_z look_x look_y look_z]
"""
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import numpy as np
import omni.timeline
import omni.usd
import omni.replicator.core as rep
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

SCENE = sys.argv[1]
OUT = sys.argv[2]
CAM = [float(x) for x in sys.argv[3:9]] if len(sys.argv) >= 9 else [2.4, 0.2, 1.8, 0.6, -2.0, 0.6]
N_FRAMES, WARMUP = 240, 20


def log(*a):
    print(">>>", *a, flush=True)


ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())
scene_prim = stage.DefinePrim("/World/Scene")
scene_prim.GetReferences().AddReference(SCENE)
UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1200.0)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(3000.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35, 15, 0))

# author OPEN drive targets on drawer + cabinet joints (maximal-coordinate; no Isaac Lab)
n_driven = 0
for prim in stage.Traverse():
    name = prim.GetName().lower()
    if prim.IsA(UsdPhysics.PrismaticJoint) and "drawer" in name:
        j = UsdPhysics.PrismaticJoint(prim)
        hi = j.GetUpperLimitAttr().Get() or 0.3
        d = UsdPhysics.DriveAPI.Apply(prim, "linear")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(3000.0)
        d.CreateDampingAttr(200.0)
        d.CreateMaxForceAttr(1.0e6)
        d.CreateTargetPositionAttr(float(hi))
        n_driven += 1
    elif prim.IsA(UsdPhysics.RevoluteJoint) and ("cabinet" in name or "fridge" in name.lower()):
        j = UsdPhysics.RevoluteJoint(prim)
        lo, hi = j.GetLowerLimitAttr().Get() or 0.0, j.GetUpperLimitAttr().Get() or 0.0
        tgt = hi if abs(hi) >= abs(lo) else lo
        d = UsdPhysics.DriveAPI.Apply(prim, "angular")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(3000.0)
        d.CreateDampingAttr(200.0)
        d.CreateMaxForceAttr(1.0e6)
        d.CreateTargetPositionAttr(float(tgt))
        n_driven += 1
log(f"authored OPEN drives on {n_driven} joints")

cam = rep.create.camera(position=tuple(CAM[:3]), look_at=tuple(CAM[3:]), focal_length=16.0)
rp = rep.create.render_product(cam, (1280, 720))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])

timeline = omni.timeline.get_timeline_interface()
timeline.play()
log("timeline playing; warmup...")
for _ in range(WARMUP):
    app.update()

import imageio.v2 as imageio

frames = []
for i in range(N_FRAMES):
    app.update()
    arr = np.asarray(rgb.get_data())
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        frames.append(arr[..., :3].astype(np.uint8).copy())
    if i % 60 == 0:
        m = float(frames[-1].mean()) if frames else -1
        log(f"frame {i}: captured={len(frames)} mean={m:.1f}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
imageio.imwrite(OUT.replace(".mp4", "_preview.png"), frames[int(len(frames) * 0.8)])
imageio.mimwrite(OUT, frames, fps=30, quality=8)
log(f"WROTE {OUT} ({len(frames)} frames) — RENDER6_SCENE_OK")
app.close()
os._exit(0)
