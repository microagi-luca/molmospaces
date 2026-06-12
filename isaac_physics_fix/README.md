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

---

## Addendum (2026-06-11, part 2)

### Reproduced & fixed: the README "collision groups reverted" issue

Upstream's README warns that "collision groups are reverted by default when using
IsaacLab's `InteractiveScene`". Root cause found and reproduced
(`repro_interactive_scene_collisions.py`, modes `repro`/`fixed`):

- `GridCloner.filter_collisions()` sets `PhysxSceneAPI.invertCollisionGroupFilter = True`
  (cloner.py:446); Isaac Lab's `InteractiveScene` calls it **unconditionally on CPU device**
  (interactive_scene.py:214).
- Inversion flips `filteredGroups` semantics from "do NOT collide" to "ONLY these collide",
  wrecking the scene's authored structural/articulable filtering: embedded furniture starts
  colliding with its enclosing counters (jams) and stops colliding with everything else.
- **Repro numbers (FloorPlan1)**: drawer 0.2% open = JAMMED (vs 95.2% direct-load);
  cabinet door self-opens 1.09 rad during a passive settle.
- **Fix (2 lines, before `sim.reset()`)**: set `invertCollisionGroupFilter = False` and
  deactivate the cloner's `/World/collisions/*` groups. Restores exact direct-load behavior
  (drawer 95.2%, cabinets 100%/99.9%, zero drift). Upstream's PR#16 workaround instead locks
  all articulation joints + collapses collision groups — functional furniture is lost.
- Multi-env note: with `env_spacing` larger than the scene, inter-env contact is impossible,
  so dropping the cloner filtering is safe (the same reason Arena's `collision_group=0` works).

### Isaac Sim 6.0: Blackwell rendering works (5.1 crash confirmed fixed)

- Isaac Sim **5.1**'s RTX renderer crashes on RTX PRO 6000 Blackwell (known NVIDIA issue;
  viewport Hydra-engine segfault). Physics is unaffected.
- The **pip** `isaacsim[all]==6.0.0` early release is missing `isaacsim.anim.robot.schema`,
  blocking every experience. The **official standalone zip** is complete:
  `https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-6.0.0-linux-x86_64.zip`
  (needs Python 3.12 if used via pip; standalone bundles its own).
- With the standalone, headless RTX rendering on Blackwell works: `isaac6_smoke.py`
  (cube render) and `isaac6_render_scene.py` (loads the converted scene, authors OPEN
  drive targets on drawer/cabinet joints, captures RGB via Replicator, writes MP4).
- Result videos: `videos/fp1_isaac_open.mp4` — Isaac-rendered iTHOR kitchen with the
  drawer stack sliding open and cabinet doors swinging.

---

## Addendum (2026-06-11, part 3): RL MVP — PPO learns to open the MolmoSpaces drawer

End-to-end RL pipeline validated in Isaac Lab (5.1 venv, headless GPU PhysX, no cameras):

- `extract_drawer_asset.py` — pulls one drawer articulation out of the converted FloorPlan1
  kitchen into a standalone, **clone-safe** USD (`~/assets_rl/molmo_drawer.usda`): strips the
  authored `FixedJointToWorld` frames so omni.physx auto-computes per-clone anchors (the
  Franka `root_joint` pattern), and bakes the re-centering shift on the `_art` child prim
  (the spawner overwrites the asset root's transform — a root-level shift is silently lost).
- `train_molmo_drawer.py` — Direct-workflow env (surgical adaptation of
  `isaaclab_tasks.direct.franka_cabinet`) + rsl_rl PPO with the stock FrankaCabinet
  hyperparameters. Franka + the MolmoSpaces drawer, 2048 envs on GPU.
  The iTHOR drawer is **handle-less**, so the grasp targets a top-hook: fingertips dip over
  the rim behind the front panel and pull (frontal handle-grasp shaping never opens it).
- Physics probes inside the env: per-clone anchors hold (4/4 envs identical, stable);
  written joint state holds; a 60 N world-frame pull slides the drawer (0 → 0.122 m).

**Learning curve (2048 envs, 400 iterations, 13.1M steps in 2m53s):**

| iteration | ~0 | ~80 | ~160 | ~240 | ~320 | 400 |
|---|---|---|---|---|---|---|
| drawer open (frac of 0.319 m travel, episode-mean) | 0.000 | 0.104 | 0.200 | 0.278 | 0.389 | **0.546** |

Mean reward 663 (reach-only baseline) → **2290**. Checkpoints in `~/rl_logs/molmo_drawer/` on the VM.

---

## Addendum (2026-06-12): handle colliders + in-context training -> TRUE in-scene transfer

The isolation->in-scene gap is CLOSED. Chain of fixes, each verified:

1. **Handle colliders in the converter** (`assets/add_handle_colliders.py`, auto-run by
   `ms-convert-houses`): the iTHOR drawer "handles" are texture-only — the visual mesh is a
   flat plate 1cm proud of the collision face (verified vertex-by-vertex). The pass authors
   (a) the faithful front-plate collider and (b) a synthesized graspable handle BAR
   (16.8x2.5x3cm, centered near the top of the face) as a documented affordance.
2. **In-context retraining** (`train_molmo_drawer.py`): frontal handle-grasp shaping
   (grasp at the bar, fingers straddle it vertically, finger_reward restored) + a static
   occluder plate over the drawer top that makes the old top-hook strategy impossible.
   PPO, 2048 envs, 2000 iters (~20 min): drawer_open_frac 0 -> 0.76, reward 2922.
3. **Closed-loop kitchen evaluation** (`eval_in_kitchen.py`): the policy runs against the
   real FloorPlan1 drawer inside the full kitchen (InteractiveScene + our collision-group
   fix). Gotcha found: `to_target` obs is WORLD-frame (template inheritance), so the
   90deg-rotated kitchen was out-of-distribution -> spawn the kitchen rotated into the
   training frame (drawer at origin, physically identical).
   **RESULT: drawer pulled to 0.275m = 86.2% of travel, fully physically, no assist** —
   approach -> handle grip -> pull -> hold.

Artifacts: `trained/model_1999_handle.pt` (in-context policy), `trained/rollout_kitchen.npz`
(in-kitchen closed-loop trajectory), `trained/molmo_drawer.usda` (asset with handle).
