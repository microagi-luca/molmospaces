"""Extract one MolmoSpaces drawer articulation from a converted scene into a standalone,
clone-safe USD asset for Isaac Lab RL.

- Copies the drawer `<name>_art` subtree (flattened) under a fresh /MolmoDrawer root.
- Strips the authored localPos0/localRot0 of FixedJointToWorld so omni.physx auto-computes
  the world anchor from each clone's spawned pose (the Franka root_joint pattern).
- Re-roots so the slider link sits at x=y=0 (height kept).
- Reports the grasp frame data needed by the RL env (slider-local bbox, axes).
"""
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

SRC, DST, MATCH = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "drawer_372d9ee4")

stage = Usd.Stage.Open(SRC)
art = next(
    p for p in stage.Traverse()
    if p.HasAPI(UsdPhysics.ArticulationRootAPI) and p.GetName().startswith(MATCH)
)
print(">>> source articulation:", art.GetPath())

flat = stage.Flatten()
layer = Sdf.Layer.CreateNew(DST)
root_spec = Sdf.CreatePrimInLayer(layer, "/MolmoDrawer")
root_spec.specifier = Sdf.SpecifierDef
root_spec.typeName = "Xform"
Sdf.CopySpec(flat, art.GetPath(), layer, Sdf.Path("/MolmoDrawer/" + art.GetName()))

out = Usd.Stage.Open(layer)
out.SetDefaultPrim(out.GetPrimAtPath("/MolmoDrawer"))
UsdGeom.SetStageUpAxis(out, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(out, 1.0)
out.SetMetadata(UsdPhysics.Tokens.kilogramsPerUnit, 1)

droot = out.GetPrimAtPath("/MolmoDrawer")

# strip authored joint frames on the fixed-to-world joint -> physx auto-computes per clone
n_stripped = 0
for p in Usd.PrimRange(droot):
    if p.GetName() == "FixedJointToWorld":
        for attr in ("physics:localPos0", "physics:localRot0", "physics:localPos1", "physics:localRot1"):
            if p.HasProperty(attr):
                p.RemoveProperty(attr)
                n_stripped += 1
        print(">>> stripped joint frames on", p.GetPath(), f"({n_stripped} attrs)")

# find links + joint
slider = joint = None
for p in Usd.PrimRange(droot):
    if p.GetName().endswith("_1_1_0") and p.HasAPI(UsdPhysics.RigidBodyAPI):
        slider = p
    if p.IsA(UsdPhysics.PrismaticJoint):
        joint = p
print(">>> slider link:", slider.GetName())
print(">>> joint:", joint.GetName(), "| upper limit:", UsdPhysics.PrismaticJoint(joint).GetUpperLimitAttr().Get())

cache = UsdGeom.XformCache()
sliderM = cache.GetLocalToWorldTransform(slider)
spos = sliderM.ExtractTranslation()
print(f">>> slider world pos (pre-shift): ({spos[0]:.4f}, {spos[1]:.4f}, {spos[2]:.4f})")

# re-root: shift so slider is at x=y=0 (keep z). IMPORTANT: author the shift on the
# *_art CHILD prim, not the asset root — Isaac Lab's spawner overwrites the root prim's
# transform with init_state pos/rot, which would silently discard a root-level shift.
art_child = out.GetPrimAtPath("/MolmoDrawer/" + art.GetName())
xf = UsdGeom.Xformable(art_child)
ops = xf.GetOrderedXformOps()
m = xf.GetLocalTransformation()
m.SetTranslateOnly(m.ExtractTranslation() + Gf.Vec3d(-spos[0], -spos[1], 0.0))
op = ops[0] if ops else xf.AddTransformOp()
op.Set(m)

# grasp frame: collision-geom AABB in slider-local coords
cache2 = UsdGeom.XformCache()  # fresh cache after the shift
sliderM2 = cache2.GetLocalToWorldTransform(slider)
inv = sliderM2.GetInverse()
mn = np.full(3, np.inf)
mx = np.full(3, -np.inf)
for g in Usd.PrimRange(slider):
    if not g.HasAPI(UsdPhysics.CollisionAPI):
        continue
    gp = UsdGeom.Gprim(g)
    if not gp:
        continue
    M = cache2.GetLocalToWorldTransform(g) * inv  # geom -> slider local
    if g.IsA(UsdGeom.Cube):
        s = (g.GetAttribute("size").Get() or 2.0) / 2.0
        ext = [(-s, s)] * 3
    else:
        e = UsdGeom.Boundable(g).GetExtentAttr().Get()
        ext = [(e[0][i], e[1][i]) for i in range(3)] if e else [(-0.05, 0.05)] * 3
    for cx in ext[0]:
        for cy in ext[1]:
            for cz in ext[2]:
                w = M.Transform(Gf.Vec3d(cx, cy, cz))
                mn = np.minimum(mn, [w[0], w[1], w[2]])
                mx = np.maximum(mx, [w[0], w[1], w[2]])
print(f">>> slider-local AABB: min={np.round(mn,4).tolist()} max={np.round(mx,4).tolist()}")

# world axes expressed in slider-local frame
R_inv = sliderM2.ExtractRotation().GetInverse()
open_l = R_inv.TransformDir(Gf.Vec3d(0, 1, 0))   # drawer opens toward world +Y
up_l = R_inv.TransformDir(Gf.Vec3d(0, 0, 1))
open_l = np.array([open_l[0], open_l[1], open_l[2]])
up_l = np.array([up_l[0], up_l[1], up_l[2]])
ax = int(np.argmax(np.abs(open_l)))
sign = 1.0 if open_l[ax] > 0 else -1.0
grasp = (mn + mx) / 2.0
grasp[ax] = mx[ax] if sign > 0 else mn[ax]
print(f">>> open dir (slider-local): {np.round(open_l,3).tolist()}  -> face axis {ax} sign {sign}")
print(f">>> up dir   (slider-local): {np.round(up_l,3).tolist()}")
print(f">>> GRASP_LOCAL_POS = {np.round(grasp,4).tolist()}")
print(f">>> INWARD_AXIS_LOCAL = {np.round(-open_l,3).tolist()}")
print(f">>> UP_AXIS_LOCAL = {np.round(up_l,3).tolist()}")

out.GetRootLayer().Save()
print(">>> saved", DST)
