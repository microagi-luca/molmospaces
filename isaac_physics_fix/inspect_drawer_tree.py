"""Explain why a THOR drawer doesn't parse as a PhysX articulation: show the prim
subtree, which prims carry RigidBody/ArticulationRoot/Collision APIs, and the joint's
body0/body1 targets relative to the articulation root."""
import sys

from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open(sys.argv[1])


def apis(prim):
    tags = []
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        tags.append("ARTIC_ROOT")
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        tags.append("RIGID")
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        tags.append("COLL")
    return ",".join(tags) or "-"


# first drawer articulation root
root = None
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and "drawer" in prim.GetName().lower():
        root = prim
        break
print("ROOT:", root.GetPath(), "[", apis(root), "]")
print("ROOT children (depth<=2):")
for child in Usd.PrimRange(root):
    depth = child.GetPath().pathString.count("/") - root.GetPath().pathString.count("/")
    if depth <= 2:
        print(f"  {'  '*depth}{child.GetName()} [{apis(child)}] <{child.GetTypeName()}>")

# now find the drawer's joint anywhere and show its bodies
prefix = root.GetName().rsplit("_1_0_0", 1)[0]  # e.g. drawer_<hash>
print(f"\nJoints whose name starts with '{prefix}':")
for prim in stage.Traverse():
    if (prim.IsA(UsdPhysics.PrismaticJoint) or prim.IsA(UsdPhysics.RevoluteJoint)) and prim.GetName().startswith(prefix):
        j = UsdPhysics.Joint(prim)
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        print(f"  JOINT {prim.GetPath()}")
        print(f"     body0 -> {[str(t) for t in b0]}")
        print(f"     body1 -> {[str(t) for t in b1]}")
        # are those bodies under the articulation root subtree?
        for t in list(b0) + list(b1):
            under = str(t).startswith(root.GetPath().pathString)
            tp = stage.GetPrimAtPath(t)
            print(f"     {t}  under_root={under}  apis=[{apis(tp) if tp else 'MISSING'}]")
