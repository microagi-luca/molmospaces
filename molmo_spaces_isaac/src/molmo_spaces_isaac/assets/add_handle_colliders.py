"""Post-conversion pass: add collision boxes for furniture handles.

The iTHOR converter authors collision geometry as a set of boxes that cover the panels of
drawers/doors but NOT the protruding handle (which exists only in the visual mesh). As a
result, grippers cannot grasp handles in Isaac. This pass, run on a converted scene (or
asset) USD, detects the handle as the region of the visual mesh that protrudes beyond the
collision-box AABB of the same link, and authors a matching guide-purpose collision Cube.

Heuristic (per jointed link):
  1. AABB of the link's collision geoms, in link-local frame.
  2. Visual-mesh points in link-local frame; protrusion = how far points extend beyond the
     collision AABB along each axis/sign. The largest protrusion >= MIN_PROTRUSION is the
     handle direction (drawer/door front).
  3. Points beyond the collision AABB (+margin) along that direction form the handle; author
     a box collider over their AABB.

Usage: python add_handle_colliders.py <scene_or_asset.usda> [more.usda ...]
"""
import sys

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

MIN_PROTRUSION_PRISMATIC = 0.008  # m — drawer handles can be shallow bars
MIN_PROTRUSION_REVOLUTE = 0.015  # m — doors: stricter, generic axis search
SELECT_MARGIN = 0.003  # m beyond the collision AABB to count as handle points
MIN_POINTS = 8


def _corners(lo, hi):
    return [Gf.Vec3d(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


def _gprim_local_aabb(g):
    """AABB of a Gprim in its own frame (Cube via size, Mesh/other via extent)."""
    if g.GetPrim().IsA(UsdGeom.Cube):
        s = (g.GetPrim().GetAttribute("size").Get() or 2.0) / 2.0
        return np.array([-s, -s, -s]), np.array([s, s, s])
    ext = UsdGeom.Boundable(g.GetPrim()).GetExtentAttr().Get()
    if ext:
        return np.array([ext[0][0], ext[0][1], ext[0][2]]), np.array([ext[1][0], ext[1][1], ext[1][2]])
    return None, None


def add_handle_colliders(stage: Usd.Stage) -> int:
    cache = UsdGeom.XformCache()
    n_added = 0

    # moving links = body0 of articulation joints (converter convention: body0 is the child link)
    links = {}  # path -> (is_prismatic, axis_index)
    for prim in stage.Traverse():
        is_pris = prim.IsA(UsdPhysics.PrismaticJoint)
        if is_pris or prim.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.Joint(prim)
            t = j.GetBody0Rel().GetTargets()
            if t:
                ax = {"X": 0, "Y": 1, "Z": 2}.get(prim.GetAttribute("physics:axis").Get() or "Z", 2)
                links[str(t[0])] = (is_pris, ax)

    for link_path in sorted(links):
        is_pris, joint_axis = links[link_path]
        link = stage.GetPrimAtPath(link_path)
        if not link:
            continue
        if any(c.GetName().endswith("_handle_collision") for c in link.GetChildren()):
            continue  # idempotent

        inv_link = cache.GetLocalToWorldTransform(link).GetInverse()
        col_lo = np.full(3, np.inf)
        col_hi = np.full(3, -np.inf)
        vis_pts = []
        for child in Usd.PrimRange(link):
            name = child.GetName()
            g = UsdGeom.Gprim(child)
            if not g:
                continue
            to_link = cache.GetLocalToWorldTransform(child) * inv_link
            if "_collision_" in name:
                lo, hi = _gprim_local_aabb(g)
                if lo is None:
                    continue
                for c in _corners(lo, hi):
                    w = to_link.Transform(c)
                    col_lo = np.minimum(col_lo, [w[0], w[1], w[2]])
                    col_hi = np.maximum(col_hi, [w[0], w[1], w[2]])
            elif "_visual_" in name and child.IsA(UsdGeom.Mesh):
                pts = UsdGeom.Mesh(child).GetPointsAttr().Get()
                if not pts:
                    continue
                arr = np.array([[p[0], p[1], p[2]] for p in pts])
                m = np.array(to_link).T  # row-major Gf -> column transform
                arr = (m[:3, :3] @ arr.T).T + m[:3, 3]
                vis_pts.append(arr)

        if not np.isfinite(col_lo).all() or not vis_pts:
            continue
        pts = np.concatenate(vis_pts, axis=0)

        # largest protrusion of visual beyond collision AABB. For drawers (prismatic) the
        # handle is on the sliding front: restrict to the joint axis dimension.
        axes = [joint_axis] if is_pris else [0, 1, 2]
        min_prot = MIN_PROTRUSION_PRISMATIC if is_pris else MIN_PROTRUSION_REVOLUTE
        best = (None, None, 0.0)  # axis, sign, amount
        for k in axes:
            pp = pts[:, k].max() - col_hi[k]
            pn = col_lo[k] - pts[:, k].min()
            if pp > best[2]:
                best = (k, +1, pp)
            if pn > best[2]:
                best = (k, -1, pn)
        k, s, amount = best
        if k is None or amount < min_prot:
            continue

        bound = col_hi[k] if s > 0 else col_lo[k]
        sel = pts[(s * pts[:, k]) > (s * bound + SELECT_MARGIN)]
        if sel.shape[0] < MIN_POINTS:
            continue
        h_lo, h_hi = sel.min(axis=0), sel.max(axis=0)
        center = (h_lo + h_hi) / 2.0
        half = np.maximum((h_hi - h_lo) / 2.0, 0.004)  # >=8mm thick so it's grippable

        def _author_box(name, c, hf):
            cube = UsdGeom.Cube.Define(stage, link.GetPath().AppendChild(name))
            cube.GetPrim().GetAttribute("size").Set(2.0)
            xf = UsdGeom.Xformable(cube)
            xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in c]))
            xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in hf]))
            cube.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        # classify the protruding region: a discrete bar (thin in one transverse dim) vs a
        # full front plate (the visual face is proud of the collision face; the painted
        # handle has no geometry in the iTHOR meshes)
        t1, t2 = [i for i in range(3) if i != k]
        ext = h_hi - h_lo
        w_dim, h_dim = (t1, t2) if ext[t1] >= ext[t2] else (t2, t1)
        if ext[h_dim] <= 0.05:
            # real bar: faithful, but ensure >=2.5cm depth so a gripper can hold it
            half[k] = max(half[k], 0.0125)
            center[k] = bound + s * half[k]
            _author_box(link.GetName() + "_handle_collision", center, half)
            kind = "bar"
        else:
            # plate: author it faithfully, plus a synthetic graspable bar (affordance for
            # the texture-painted handle: centered, near the top of the face, 3cm proud)
            _author_box(link.GetName() + "_front_collision", center, half)
            bar_c = np.zeros(3)
            bar_h = np.zeros(3)
            bar_c[w_dim] = center[w_dim]
            bar_h[w_dim] = 0.35 * half[w_dim]
            bar_h[h_dim] = 0.0125
            bar_c[h_dim] = h_hi[h_dim] - 0.035
            bar_h[k] = 0.015
            face = h_hi[k] if s > 0 else h_lo[k]
            bar_c[k] = face + s * bar_h[k]
            _author_box(link.GetName() + "_handle_collision", bar_c, bar_h)
            kind = f"plate+synthetic-bar c={np.round(bar_c,4).tolist()} h={np.round(bar_h,4).tolist()}"
        n_added += 1
        print(
            f"[handle] {link.GetName()}: axis={'xyz'[k]}{'+' if s>0 else '-'} protrusion={amount:.3f}m {kind}",
            flush=True,
        )

    return n_added


def add_handle_colliders_to_file(path: str) -> int:
    stage = Usd.Stage.Open(path)
    if stage is None:
        print(f"[handle] ERROR: cannot open {path}")
        return 0
    n = add_handle_colliders(stage)
    stage.GetRootLayer().Save()
    print(f"[handle] {path}: added {n} handle colliders")
    return n


if __name__ == "__main__":
    for f in sys.argv[1:]:
        add_handle_colliders_to_file(f)
