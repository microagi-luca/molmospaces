"""Headless MuJoCo (EGL) render of an iTHOR MJCF scene with all articulated joints opening.
Produces an MP4 + a preview PNG. Works on the Blackwell GPU where Isaac's RTX renderer crashes."""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np

XML = sys.argv[1]
OUT_MP4 = sys.argv[2]
PREVIEW_PNG = sys.argv[3] if len(sys.argv) > 3 else OUT_MP4.replace(".mp4", "_preview.png")
NAME_FILTER = sys.argv[4] if len(sys.argv) > 4 else ""  # e.g. "drawer" / "cabinet"; "" = all

W, H, N, FPS = 1280, 720, 180, 30

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# hide walls/ceiling/structural so we can see the furniture inside, and brighten the scene
def _body_chain_names(m, body_id):
    names, b = [], int(body_id)
    for _ in range(64):
        names.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").lower())
        if b == 0:
            break
        b = int(m.body_parentid[b])
    return names

if NAME_FILTER:
    # isolate: keep ONLY geoms belonging to object(s) whose body-chain matches NAME_FILTER
    for g in range(model.ngeom):
        chain = " ".join(_body_chain_names(model, model.geom_bodyid[g]))
        if NAME_FILTER.lower() not in chain:
            model.geom_rgba[g, 3] = 0.0
else:
    CULL = ("wall", "ceiling", "room_", "window", "structural")
    for g in range(model.ngeom):
        chain = _body_chain_names(model, model.geom_bodyid[g])
        gname = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        if any(any(s in nm for s in CULL) for nm in chain) or any(s in gname for s in CULL):
            model.geom_rgba[g, 3] = 0.0
model.vis.headlight.active = 1
model.vis.headlight.ambient[:] = 0.5
model.vis.headlight.diffuse[:] = 0.8
model.vis.headlight.specular[:] = 0.3

# articulated joints to animate (limited hinge/slide), optionally filtered by name
jids = []
for j in range(model.njnt):
    jt = model.jnt_type[j]
    if jt not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        continue
    if not model.jnt_limited[j]:
        continue
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
    if NAME_FILTER and NAME_FILTER.lower() not in jname.lower():
        continue
    jids.append(j)
print(f">>> animating {len(jids)} joints (filter={NAME_FILTER!r})", flush=True)

# frame on the centroid of the articulated furniture bodies
mujoco.mj_forward(model, data)
body_pts = np.array([data.xpos[model.jnt_bodyid[j]] for j in jids]) if jids else data.geom_xpos[: model.ngeom]
center = body_pts.mean(axis=0)
spread = float(np.linalg.norm(body_pts.max(axis=0) - body_pts.min(axis=0)))

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
if NAME_FILTER and jids:
    # frame tightly on the isolated object's visible geoms
    vis = [g for g in range(model.ngeom)
           if NAME_FILTER.lower() in " ".join(_body_chain_names(model, model.geom_bodyid[g]))]
    pts = data.geom_xpos[vis] if vis else body_pts
    c = pts.mean(axis=0)
    sz = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    cam.lookat[:] = c
    cam.distance = max(1.2, sz * 2.0)
    cam.azimuth = 50.0
    cam.elevation = -18.0
else:
    cam.lookat[:] = center
    cam.distance = max(2.6, spread * 1.1)
    cam.azimuth = 45.0
    cam.elevation = -38.0
base_az = float(cam.azimuth)

renderer = mujoco.Renderer(model, height=H, width=W)
q0 = data.qpos.copy()

frames = []
for i in range(N):
    # ease 0->1 over first 70%, hold open after
    t = min(1.0, (i / (N * 0.7)))
    a = t * t * (3 - 2 * t)  # smoothstep
    cam.azimuth = base_az - 12.0 + 24.0 * (i / N)  # gentle orbit for depth
    for j in jids:
        adr = model.jnt_qposadr[j]
        lo, hi = model.jnt_range[j]
        target = hi if abs(hi) >= abs(lo) else lo
        data.qpos[adr] = q0[adr] + a * (target - q0[adr])
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    frame = renderer.render()
    frames.append(frame.copy())
    if i == int(N * 0.75):
        imageio.imwrite(PREVIEW_PNG, frame)
        print(f">>> preview {PREVIEW_PNG} mean={frame.mean():.1f}", flush=True)

imageio.mimwrite(OUT_MP4, frames, fps=FPS, quality=8)
print(f">>> wrote {OUT_MP4} ({len(frames)} frames)", flush=True)
