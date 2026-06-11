# Isaac Lab physics fix — iTHOR articulations

This branch fixes the MolmoSpaces → USD converter so that iTHOR articulated furniture
(drawers, cabinets, dishwashers, ovens, …) loads as **valid PhysX reduced-coordinate
articulations** in Isaac Sim / Isaac Lab. Before this fix, the published USD scenes could
not be driven through Isaac Lab's `Articulation` API and several joints had no drive.

## The problems

Audited on `ithor/FloorPlan1_physics` (Isaac Sim 5.1, Isaac Lab 2.3.x):

1. **Missing joint drives** — the *published* USD (HF dataset, converted 2026-01-20) had
   no drive on the drawers and several appliances, so they could not be actuated.
   *This is already fixed in the current converter code* — reconverting from MJCF applies
   the per-asset drives from `usd_assets_parameters.yaml` (the published USD was just
   stale). No code change needed; just reconvert.

2. **Malformed articulation hierarchy** (fixed here) — `convert_body_flatten_articulated`
   in `molmo_spaces_isaac/assets/utils/ithor.py` flattened every link as a **sibling**
   under `/Geometry` and applied `ArticulationRootAPI` to the top body, which was an empty
   `Xform` with no `RigidBodyAPI`. PhysX builds an articulation from the rigid bodies/joints
   in the **subtree** of the prim carrying `ArticulationRootAPI` — that subtree was empty,
   so no articulation ever formed. Isaac Lab's `Articulation` API failed with
   *"did not match any rigid bodies / articulations"*.

## The fix

In `convert_body_flatten_articulated` (`assets/utils/ithor.py`):

- On the root call, create a per-object **group** Xform `<name>_art` that carries
  `ArticulationRootAPI` (+ `PhysxArticulationAPI` when `--use-physx`).
- **Nest every link** (root + children) under that group, so PhysX finds all links in the
  articulation-root subtree.
- Give the **base link** a `RigidBodyAPI` and a **fixed joint to the world**
  (`FixedJointToWorld`) so the static furniture stays put → fixed-base articulation; only
  the jointed child links move.

## Validation (RTX PRO 6000 Blackwell, headless Isaac Sim 5.1)

Reconvert + drive each joint to its limit, measuring % opened and stability:

| Scene | Object | Joint type | Opened |
|-------|--------|-----------|--------|
| FloorPlan1 | Cabinet (2 doors) | revolute | 100% / 99.9% |
| FloorPlan1 | Drawer | prismatic | 95.2% |
| FloorPlan1 | Dishwasher | revolute | 85.6% |
| FloorPlan1 | Oven door | revolute | 99.8% |
| FloorPlan1 | Oven broiler | prismatic | 21.7% ⚠️ (per-asset tuning TODO) |
| FloorPlan10 | Cabinet (2 doors) | revolute | 100% / 100% |
| FloorPlan10 | Drawer | prismatic | 92.1% |

All articulations **bind** in Isaac Lab and show **zero joint drift** under a 1 s passive
gravity settle (fixed-base holds). The oven broiler underperforms — likely a genuine
contact/heavy-part tuning issue on that sub-part, not a structural one.

## Reconvert recipe

```bash
# 1) MJCF source scenes
ms-download --type mjcf --install-dir assets/mjcf --scenes ithor
# 2) reconvert one scene to USD (empty --thor-usd-dir is fine for testing furniture;
#    referenced small THOR objects just warn+skip)
ms-convert-houses --mode convert-single \
  --scene-path assets/mjcf/scenes/ithor/FloorPlan1_physics.xml \
  --output-dir out --thor-usd-dir /path/to/thor_usd_or_empty \
  --is-ithor --dataset ithor --use-physx
```

## Validation scripts (in this directory)

All take a scene `scene.usda` as the first arg; the `drive_*` ones launch headless Isaac Lab.

- `inspect_usd_physics.py <usd>` — GPU-free audit: joints, drive gains, limits, collision filtering.
- `inspect_drawer_tree.py <usd>` — GPU-free: show an articulation's prim subtree + joint body targets.
- `smoke_isaac.py` — headless GPU-PhysX smoke test (drops a cube).
- `drive_drawer.py <usd>` — bind one drawer via Isaac Lab `Articulation`, command open, report %.
- `drive_articulations_multi.py <usd> [substrings...]` — drive multiple articulation types in one launch.
- `drive_joint_maxcoord.py <usd>` — drive a joint as a maximal-coordinate constraint (no articulation).

## Reproducing the environment

Verified on Debian 12 + NVIDIA RTX PRO 6000 Blackwell (sm_120), driver `nvidia-open`
610.43.02, CUDA 13. The project uses **uv**; `requirements-frozen.txt` pins the exact
259-package working set (Isaac Sim 5.1.0.0, Isaac Lab 2.3.2.post1, torch 2.12.0+cu130).

```bash
# system deps (Debian) — libGLU is REQUIRED or Isaac Sim hangs on startup (neuray/MDL)
sudo apt-get install -y build-essential dkms linux-headers-$(uname -r) \
     git libglu1-mesa libsm6 libice6 libxt6 libpython3.11

# python env (uv); setuptools<81 build constraint is needed for a legacy transitive dep
cd molmo_spaces_isaac
uv venv --python 3.11 .venv && source .venv/bin/activate
printf 'setuptools<81\nwheel<0.46\n' > /tmp/build_constraints.txt
uv pip install -b /tmp/build_constraints.txt -e '.[dev,sim]'
# ms-download needs molmospaces-resources (not a dep of this package):
uv pip install -b /tmp/build_constraints.txt --index-strategy unsafe-best-match \
     --extra-index-url https://test.pypi.org/simple/ 'molmospaces-resources==0.0.1b4'
```

Run headless with `OMNI_KIT_ACCEPT_EULA=YES python -u <script>` (use `-u`; output is
otherwise block-buffered). A **pixi** equivalent can be authored from the same package
list if a conda-based, fully-locked environment is preferred.
