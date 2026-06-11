"""Load a MolmoSpaces iTHOR USD scene headless, find a drawer articulation, add an
actuator, command it OPEN, and report whether it actually moves (drive + contacts test)."""
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from pxr import Usd, UsdPhysics

SCENE = sys.argv[1]


def log(*a):
    print(">>>", *a, flush=True)


sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cpu"))

# spawn the whole scene as a payload/reference under /World/Scene
scene_cfg = sim_utils.UsdFileCfg(usd_path=SCENE)
scene_cfg.func("/World/Scene", scene_cfg)
light = sim_utils.DomeLightCfg(intensity=2000.0)
light.func("/World/Light", light)

stage = omni.usd.get_context().get_stage()

# discover articulation roots + a drawer's prismatic joint
roots, drawer_roots = [], []
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        p = prim.GetPath().pathString
        roots.append(p)
        if "drawer" in p.lower():
            drawer_roots.append(p)
log("total articulation roots:", len(roots))
log("drawer roots found:", len(drawer_roots))
for r in drawer_roots[:5]:
    log("  ", r)

if not drawer_roots:
    log("NO DRAWER ROOTS — dumping first roots:", roots[:8])
    os._exit(2)

target_root = drawer_roots[0]
# find the prismatic joint + its upper limit under that root
joint_name, upper = None, None
for prim in stage.Traverse():
    p = prim.GetPath().pathString
    if p.startswith(target_root) and prim.IsA(UsdPhysics.PrismaticJoint):
        joint_name = prim.GetName()
        j = UsdPhysics.PrismaticJoint(prim)
        upper = j.GetUpperLimitAttr().Get()
        break
log("TARGET:", target_root, "joint:", joint_name, "upper_limit(m):", upper)

art = Articulation(
    ArticulationCfg(
        prim_path=target_root,
        spawn=None,
        actuators={
            "drawer": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=2000.0, damping=100.0
            )
        },
    )
)

sim.reset()
art.update(sim.get_physics_dt())
log("joint_names:", art.joint_names)
q0 = art.data.joint_pos.clone()
log("q0 (closed):", q0.cpu().numpy().tolist())

# passive settle: no command — does it stay put or drift/jitter?
for _ in range(120):
    sim.step()
    art.update(sim.get_physics_dt())
log("q after 1s passive settle:", art.data.joint_pos.cpu().numpy().tolist())

# command OPEN to the upper limit
import torch

tgt = torch.full_like(art.data.joint_pos, float(upper or 0.3))
art.set_joint_position_target(tgt)
for i in range(400):
    art.write_data_to_sim()
    sim.step()
    art.update(sim.get_physics_dt())
    if i in (50, 150, 399):
        log(f"step {i}: q = {art.data.joint_pos.cpu().numpy().tolist()}")

q_open = art.data.joint_pos.cpu().numpy().reshape(-1)[0]
frac = q_open / float(upper) if upper else 0.0
log(f"RESULT: commanded={upper:.4f}m reached={q_open:.4f}m  -> {frac*100:.1f}% open")
log("VERDICT:", "OPENS_OK" if frac > 0.8 else ("PARTIAL" if frac > 0.1 else "JAMMED"))
os._exit(0)
