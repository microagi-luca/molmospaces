"""Drive a THOR drawer's prismatic joint as a maximal-coordinate joint (author a linear
drive at runtime), then measure the slider rigid body's world displacement to see whether
it slides open freely or jams on contacts."""
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from pxr import UsdPhysics

SCENE = sys.argv[1]


def log(*a):
    print(">>>", *a, flush=True)


sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cpu"))
sim_utils.UsdFileCfg(usd_path=SCENE).func("/World/Scene", sim_utils.UsdFileCfg(usd_path=SCENE))
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))
stage = omni.usd.get_context().get_stage()

slider = joint = upper = None
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.PrismaticJoint) and "drawer" in prim.GetName().lower():
        j = UsdPhysics.PrismaticJoint(prim)
        joint = prim
        upper = j.GetUpperLimitAttr().Get()
        slider = str(j.GetBody0Rel().GetTargets()[0])
        break
log("joint:", joint.GetPath(), "slider body:", slider, "upper(m):", upper)

# author a linear drive targeting the open position (the missing drive in the USD)
d = UsdPhysics.DriveAPI.Apply(joint, "linear")
d.CreateTypeAttr("force")
d.CreateStiffnessAttr(2000.0)
d.CreateDampingAttr(100.0)
d.CreateMaxForceAttr(1.0e6)
d.CreateTargetPositionAttr(float(upper))

body = RigidObject(RigidObjectCfg(prim_path=slider, spawn=None))
sim.reset()
body.update(sim.get_physics_dt())
p0 = body.data.root_pos_w[0].clone()
log("slider world pos t0:", [round(x, 4) for x in p0.cpu().numpy().tolist()])

for i in range(400):
    sim.step()
    body.update(sim.get_physics_dt())
    if i in (100, 250, 399):
        pi = body.data.root_pos_w[0]
        log(f"step {i}: pos = {[round(x, 4) for x in pi.cpu().numpy().tolist()]}")

p1 = body.data.root_pos_w[0].clone()
disp = (p1 - p0).norm().item()
log(f"RESULT: slider moved {disp:.4f} m   (commanded travel = {float(upper):.4f} m)")
frac = disp / float(upper) if upper else 0.0
log("VERDICT:", "SLIDES_OK" if frac > 0.8 else ("PARTIAL" if frac > 0.1 else "JAMMED/NO-MOVE"))
os._exit(0)
