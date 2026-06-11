"""Generalized articulation check: for each name-substring, find its `_art` root, bind an
Isaac Lab Articulation, drive every joint to its open limit, and report % opened + stability.
Handles revolute (rad) and prismatic (m) uniformly via joint_pos_limits (SI)."""
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from pxr import UsdPhysics

SCENE = sys.argv[1]
SUBS = sys.argv[2:] or ["cabinet", "drawer", "dishwasher", "oven"]


def log(*a):
    print(">>>", *a, flush=True)


sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cpu"))
sim_utils.UsdFileCfg(usd_path=SCENE).func("/World/Scene", sim_utils.UsdFileCfg(usd_path=SCENE))
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))
stage = omni.usd.get_context().get_stage()

# one art-root per requested substring
roots = {}
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        path = prim.GetPath().pathString
        for sub in SUBS:
            if sub.lower() in path.lower() and sub not in roots:
                roots[sub] = path
log("matched roots:", {k: v.split("/")[-1] for k, v in roots.items()})

arts = {}
for sub, path in roots.items():
    try:
        a = Articulation(
            ArticulationCfg(
                prim_path=path,
                spawn=None,
                actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=2000.0, damping=150.0)},
            )
        )
        arts[sub] = a
    except Exception as e:  # noqa: BLE001
        log(f"BIND FAILED for {sub}: {e}")

sim.reset()
dt = sim.get_physics_dt()
for a in arts.values():
    a.update(dt)

q_closed = {sub: a.data.joint_pos.clone() for sub, a in arts.items()}
for sub, a in arts.items():
    log(f"{sub}: dof={a.joint_names} closed={[round(x,3) for x in a.data.joint_pos.cpu().numpy().reshape(-1).tolist()]}")

# passive settle (stability under gravity, no command)
for _ in range(120):
    sim.step()
    for a in arts.values():
        a.update(dt)
for sub, a in arts.items():
    drift = (a.data.joint_pos - q_closed[sub]).abs().max().item()
    log(f"{sub}: max joint drift after 1s settle = {drift:.4f}")

# command OPEN (bound with larger magnitude per joint)
def open_target(a):
    lim = a.data.joint_pos_limits[0]
    lo, hi = lim[:, 0], lim[:, 1]
    return torch.where(hi.abs() >= lo.abs(), hi, lo).unsqueeze(0)

tgts = {sub: open_target(a) for sub, a in arts.items()}
for sub, a in arts.items():
    a.set_joint_position_target(tgts[sub])
for _ in range(500):
    for a in arts.values():
        a.write_data_to_sim()
    sim.step()
    for a in arts.values():
        a.update(dt)

log("================ RESULTS ================")
for sub, a in arts.items():
    q = a.data.joint_pos[0]
    t = tgts[sub][0]
    for jn, qi, ti in zip(a.joint_names, q.cpu().numpy().tolist(), t.cpu().numpy().tolist()):
        frac = (qi / ti) if abs(ti) > 1e-6 else 1.0
        verdict = "OPENS_OK" if frac > 0.8 else ("PARTIAL" if frac > 0.1 else "JAMMED")
        log(f"{sub:11s} {jn[-28:]:28s} reached={qi:+.3f} target={ti:+.3f} -> {frac*100:5.1f}% {verdict}")
os._exit(0)
