"""Minimal Isaac Sim 6.0 startup test: launch with a chosen experience and update frames.
Tells us whether the renderer initializes on Blackwell without the 5.1 segfault."""
import os

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp

cfg = {"headless": True}
exp = os.environ.get("ISAAC_EXPERIENCE")
if exp:
    cfg["experience"] = exp
app = SimulationApp(cfg)
print(">>> APP STARTED OK", flush=True)
for i in range(30):
    app.update()
print(">>> UPDATED 30 FRAMES — RENDER6_OK", flush=True)
app.close()
os._exit(0)
