"""Render a few candidate camera angles of a scene to PNGs (headless RTX) to pick framing."""
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

_kw = dict(headless=True, enable_cameras=True)
_exp = os.environ.get("ISAAC_EXPERIENCE")
if _exp:
    _kw["experience"] = _exp
app_launcher = AppLauncher(**_kw)
simulation_app = app_launcher.app

import numpy as np
import torch
import imageio.v2 as imageio
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext

SCENE = sys.argv[1]
OUTDIR = sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)

# (eye_xyz, target_xyz) candidates for the iTHOR kitchen (room ~ x[-2.5,2.5] y[-2.9,2.56] z<2.6)
POSES = {
    "A_corner_high": ((3.2, 3.2, 2.4), (0.0, -0.6, 0.8)),
    "B_over_drawers": ((0.9, 1.2, 1.9), (0.85, -2.2, 0.55)),
    "C_opp_corner": ((-2.9, 3.0, 2.2), (0.6, -1.6, 0.7)),
    "D_close_cluster": ((2.4, -0.3, 1.5), (0.6, -2.2, 0.6)),
}

sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cpu"))
sim_utils.UsdFileCfg(usd_path=SCENE).func("/World/Scene", sim_utils.UsdFileCfg(usd_path=SCENE))
sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))

cam = Camera(
    CameraCfg(
        prim_path="/World/cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, clipping_range=(0.05, 60.0)),
    )
)

sim.reset()
dt = sim.get_physics_dt()
# warmup so the RTX/MDL shaders compile on first render
for _ in range(40):
    sim.step()
    cam.update(dt)
print(">>> warmup done", flush=True)

for name, (eye, tgt) in POSES.items():
    cam.set_world_poses_from_view(torch.tensor([eye]), torch.tensor([tgt]))
    for _ in range(6):
        sim.step()
        cam.update(dt)
    rgb = cam.data.output["rgb"][0].cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    path = os.path.join(OUTDIR, f"frame_{name}.png")
    imageio.imwrite(path, rgb.astype(np.uint8))
    print(f">>> wrote {path} {rgb.shape} mean={rgb.mean():.1f}", flush=True)

os._exit(0)
