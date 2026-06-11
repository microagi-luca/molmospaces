"""Reproduce (and fix) the molmospaces README issue: "collision groups are reverted by
default when using IsaacLab's InteractiveScene".

Root cause: isaacsim.core.cloner.GridCloner.filter_collisions() sets
PhysxSceneAPI.invertCollisionGroupFilter = True (cloner.py:446), and Isaac Lab's
InteractiveScene calls it unconditionally on CPU device (interactive_scene.py:214).
Under inverted semantics, the scene's USD-authored collision groups
(structural <-/-> articulable furniture, authored as "do NOT collide") flip to
"ONLY these collide" -> embedded furniture collides with its enclosing counters/walls
(jams/explosions) and stops colliding with everything else (fall-through).

Usage: <scene.usda> repro   -> load via InteractiveScene as-is  (expect BREAKAGE)
       <scene.usda> fixed   -> + flip invert flag off and disable cloner groups (expect OK)
"""
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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from pxr import PhysxSchema, UsdPhysics

SCENE_USD = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "repro"  # "repro" | "fixed"
DRAWER_ART = "drawer_372d9ee41d70550432c30a66a6e5b331_1_0_0_art"
CABINET_ART = "cabinet_9ec3f0517bfaa6873677305ad67285c7_1_0_0_art"


def log(*a):
    print(">>>", *a, flush=True)


@configclass
class ReproSceneCfg(InteractiveSceneCfg):
    house = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=SCENE_USD),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
    )
    drawer = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Geometry/" + DRAWER_ART,
        spawn=None,
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=2000.0, damping=150.0)},
    )
    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Geometry/" + CABINET_ART,
        spawn=None,
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=2000.0, damping=150.0)},
    )


sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cpu"))
scene = InteractiveScene(ReproSceneCfg(num_envs=1, env_spacing=25.0))
stage = omni.usd.get_context().get_stage()

# ---------- AUDIT: what did InteractiveScene do to collision filtering? ----------
ps_path = getattr(scene, "physics_scene_path", "/physicsScene")
physx = PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath(str(ps_path)))
inv_attr = physx.GetInvertCollisionGroupFilterAttr()
log(f"AUDIT physicsScene={ps_path} invertCollisionGroupFilter={inv_attr.Get()}")
groups = [p.GetPath().pathString for p in stage.Traverse() if p.IsA(UsdPhysics.CollisionGroup)]
log(f"AUDIT collision groups on stage ({len(groups)}):")
for g in groups[:12]:
    log("   ", g)

# ---------- FIX (mode=fixed): restore normal semantics, drop cloner groups ----------
if MODE == "fixed":
    inv_attr.Set(False)
    n_off = 0
    for g in groups:
        if "/collisions" in g.lower() or "env" in g.split("/")[-1].lower():
            prim = stage.GetPrimAtPath(g)
            if prim and not g.startswith(str(scene.env_prim_paths[0])):
                prim.SetActive(False)
                n_off += 1
    log(f"FIX applied: invert=False, deactivated {n_off} cloner collision groups")

sim.reset()
dt = sim.get_physics_dt()
drawer, cabinet = scene["drawer"], scene["cabinet"]
for a in (drawer, cabinet):
    a.update(dt)
log("drawer dof:", drawer.joint_names)
log("cabinet dof:", cabinet.joint_names)

q0d, q0c = drawer.data.joint_pos.clone(), cabinet.data.joint_pos.clone()
p0d = drawer.data.root_pos_w.clone()

# ---------- passive settle: does the furniture sit still or explode/jam? ----------
for _ in range(180):
    sim.step()
    drawer.update(dt)
    cabinet.update(dt)
drift_d = (drawer.data.joint_pos - q0d).abs().max().item()
drift_c = (cabinet.data.joint_pos - q0c).abs().max().item()
root_disp = (drawer.data.root_pos_w - p0d).norm().item()
log(f"SETTLE drawer joint drift={drift_d:.4f}  cabinet joint drift={drift_c:.4f}  drawer ROOT displacement={root_disp:.4f} m")

# ---------- command OPEN ----------
def open_target(a):
    lim = a.data.joint_pos_limits[0]
    lo, hi = lim[:, 0], lim[:, 1]
    return torch.where(hi.abs() >= lo.abs(), hi, lo).unsqueeze(0)

td, tc = open_target(drawer), open_target(cabinet)
drawer.set_joint_position_target(td)
cabinet.set_joint_position_target(tc)
for _ in range(500):
    drawer.write_data_to_sim()
    cabinet.write_data_to_sim()
    sim.step()
    drawer.update(dt)
    cabinet.update(dt)

log("================ RESULTS (mode=%s) ================" % MODE)
for name, art, tgt in (("drawer", drawer, td), ("cabinet", cabinet, tc)):
    q, t = art.data.joint_pos[0], tgt[0]
    for jn, qi, ti in zip(art.joint_names, q.tolist(), t.tolist()):
        frac = qi / ti if abs(ti) > 1e-6 else 1.0
        verdict = "OPENS_OK" if frac > 0.8 else ("PARTIAL" if frac > 0.1 else "JAMMED")
        log(f"{name:8s} {jn[-30:]:30s} reached={qi:+.3f} target={ti:+.3f} -> {frac * 100:5.1f}% {verdict}")
os._exit(0)
