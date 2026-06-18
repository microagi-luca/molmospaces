"""Open a MolmoSpaces USD (no sim/GPU needed) and audit its physics setup:
joints, drive gains, limits, collision filtering. Helps explain drawer/contact issues."""
import sys

from pxr import Usd, UsdPhysics


def fget(attr, default=None):
    return attr.Get() if attr and attr.IsValid() and attr.HasAuthoredValue() else default


def main(path: str) -> None:
    stage = Usd.Stage.Open(path)
    if stage is None:
        print(f"FAILED to open {path}")
        return

    joints, rev, pris, art_roots, rigids, colliders, filtered = [], 0, 0, 0, 0, 0, 0
    drive_missing = []

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_roots += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigids += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders += 1
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            filtered += 1

        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            is_rev = prim.IsA(UsdPhysics.RevoluteJoint)
            rev += int(is_rev)
            pris += int(not is_rev)
            j = UsdPhysics.RevoluteJoint(prim) if is_rev else UsdPhysics.PrismaticJoint(prim)
            axis = fget(j.GetAxisAttr())
            lo = fget(j.GetLowerLimitAttr())
            hi = fget(j.GetUpperLimitAttr())

            drive_token = "angular" if is_rev else "linear"
            has_drive = prim.HasAPI(UsdPhysics.DriveAPI, drive_token)
            stiff = damp = tgt = maxf = None
            if has_drive:
                d = UsdPhysics.DriveAPI(prim, drive_token)
                stiff = fget(d.GetStiffnessAttr())
                damp = fget(d.GetDampingAttr())
                tgt = fget(d.GetTargetPositionAttr())
                maxf = fget(d.GetMaxForceAttr())
            else:
                drive_missing.append(prim.GetName())

            armature = fget(prim.GetAttribute("physxJoint:armature"))
            friction = fget(prim.GetAttribute("physxJoint:jointFriction"))

            joints.append(
                f"  [{'REV' if is_rev else 'PRIS'}] {prim.GetName()}  axis={axis} "
                f"limits=({lo},{hi}) drive={'Y' if has_drive else 'N'} "
                f"k={stiff} c={damp} tgt={tgt} maxF={maxf} arm={armature} fric={friction}"
            )

    print(f"=== {path} ===")
    print(
        f"articulation_roots={art_roots} rigid_bodies={rigids} colliders={colliders} "
        f"filtered_pairs={filtered} joints={len(joints)} (rev={rev} pris={pris})"
    )
    print(f"joints WITHOUT a drive: {len(drive_missing)}")
    if drive_missing:
        print("  -> " + ", ".join(drive_missing[:30]))
    print("--- joint detail (first 40) ---")
    for line in joints[:40]:
        print(line)


if __name__ == "__main__":
    main(sys.argv[1])
