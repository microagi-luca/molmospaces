"""Headless Isaac Sim smoke test: launch app, run GPU PhysX, drop a cube, confirm gravity."""
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext

print(">>> creating SimulationContext (GPU PhysX)")
sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))

sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
light = sim_utils.DomeLightCfg(intensity=2000.0)
light.func("/World/Light", light)

box = RigidObject(
    RigidObjectCfg(
        prim_path="/World/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
)

print(">>> sim.reset()")
sim.reset()
print(">>> physics device:", sim.device)

box.update(sim.get_physics_dt())
z0 = box.data.root_pos_w[0, 2].item()
for _ in range(120):
    sim.step()
    box.update(sim.get_physics_dt())
z1 = box.data.root_pos_w[0, 2].item()

print(f">>> box z: {z0:.4f} -> {z1:.4f}  (should drop toward ~0.1)")
print("FELL" if z1 < z0 - 0.5 else "DID_NOT_FALL")

simulation_app.close()
print("SMOKE_OK")
