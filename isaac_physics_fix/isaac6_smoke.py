"""Isaac Sim 6.0 headless RTX render smoke test on Blackwell — does it render a frame
without the 5.1 viewport/hydra segfault?  Native isaacsim API (no Isaac Lab)."""
import os

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp

_cfg = {"headless": True}
_exp = os.environ.get("ISAAC_EXPERIENCE")
if _exp:
    _cfg["experience"] = _exp
app = SimulationApp(_cfg)

import numpy as np
import imageio.v2 as imageio
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, UsdGeom, UsdLux

os.makedirs("/tmp/molmo_isaac6", exist_ok=True)
stage = omni.usd.get_context().get_stage()

UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1000.0)
UsdLux.DistantLight.Define(stage, "/World/Sun").CreateIntensityAttr(2500.0)

cube = UsdGeom.Cube.Define(stage, "/World/Cube")
cube.GetPrim().GetAttribute("size").Set(1.0)
UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5))
ground = UsdGeom.Cube.Define(stage, "/World/Ground")
ground.GetPrim().GetAttribute("size").Set(20.0)
UsdGeom.Xformable(ground).AddTranslateOp().Set(Gf.Vec3d(0, 0, -10.0))

print(">>> scene built, creating camera/render product", flush=True)
cam = rep.create.camera(position=(4, 4, 3), look_at=(0, 0, 0.5))
rp = rep.create.render_product(cam, (1280, 720))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])

print(">>> stepping renderer", flush=True)
for i in range(30):
    rep.orchestrator.step(rt_subframes=2)

arr = np.asarray(rgb.get_data())
print(">>> rgb shape", arr.shape, "mean", float(arr[..., :3].mean()), flush=True)
imageio.imwrite("/tmp/molmo_isaac6/smoke.png", arr[..., :3].astype(np.uint8))
print(">>> RENDER6_OK", flush=True)
app.close()
os._exit(0)
