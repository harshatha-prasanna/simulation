# Task 1 Implementation Plan: Scene Construction

**Goal:** Build a tabletop scene in `assets/descriptions/DropCubeInBinEnv.xml` with the Panda arm, a cube and a bin, a useful `fixed` camera, and a `home` keyframe. No control code.

**Why it matters later:** Tasks 2 and 3 load this exact file and look up bodies by name, so the names and the keyframe set here have to be right.

## Key numbers (from the provided files)

| Thing | Value | Where it comes from |
|---|---|---|
| Tabletop surface height | z = 0.76 m | table top box: center 0.735 + half-thickness 0.025 |
| Tabletop extent | x ∈ [-0.8, 0.8], y ∈ [-0.575, 0.575] | table top box size `.8 .575` |
| Bin inner opening | ~9 cm × 9 cm, walls ~5.5 cm tall | `bin_body.xml` wall positions/sizes |
| Gripper max opening | 8 cm (2 × 0.04 m finger travel) | `finger` joint range `0 0.04` |
| Panda reach | ~0.85 m | Franka spec |
| Gripper command | 0 = closed, 255 = open | `actuator8` ctrlrange `0 255` |

## Body names later tasks need

| Name | Used by | How we provide it |
|---|---|---|
| `panda_arm` | Task 2 (robot base pose for planner) | wrapper body around the Panda include, placed on the table |
| `panda_hand_tcp` | Task 2 (planner target point) | new empty body inside `hand`, between the fingertips |
| `panda_hand` | Task 3 (TCP position) | not added; Task 3's code has a TODO to verify names, so we'll point it at `panda_hand_tcp` there |
| `cube` | Tasks 2 and 3 | new body we write |
| `bin` | Tasks 2 and 3 | wrapper body around the provided `simpleWoodBin` |

## Steps

### 1. Understand the file structure
- [x] `<include>` pastes another file in place. Each object has an **asset** file (meshes, materials, defaults, actuators), which goes at the top level, and a **body** file (the physical object), which goes inside `<worldbody>`.
- [x] Bodies nest to form the **kinematic tree**. A body's `pos`/`quat` are relative to its parent.

### 2. Add the Panda arm
- [x] Include `panda/mjcf/panda_asset.xml` at the top level.
- [x] Inside `<worldbody>`, add `<body name="panda_arm" pos="-0.6 0 0.76">` wrapping the `panda_body.xml` include. That puts the base on the tabletop near one end, facing +x along the table.
- [x] Check that the mesh files load. Not needed: MuJoCo resolves file paths relative to each included file, so the meshes loaded as-is.

### 3. Add the TCP marker
- [x] In `panda_body.xml`, add `<body name="panda_hand_tcp" pos="0 0 0.1034"/>` inside `hand`. 0.1034 m is the standard Franka hand-to-fingertip-center offset.
- [x] This is the only edit to a provided asset file; there's no way to add a child body from outside an include.

### 4. Add the bin
- [x] Include `bin/bin_asset.xml` at the top level.
- [x] Wrap `bin/bin_body.xml` in `<body name="bin" pos="...">` on the tabletop (z = 0.76 + 0.005 for the bin floor's half-thickness), in front of the arm and offset sideways from the cube.
- [x] The bin has no joint, so it's fixed to the world. That's fine for this task.

### 5. Add the cube
- [x] `<body name="cube" pos="x y 0.78">` with `<freejoint name="cube_joint"/>` and a box geom `size="0.02 0.02 0.02"` (4 cm cube, half-sizes), blue.
- [x] The free joint gives the cube 6 degrees of freedom but **7 position values** (xyz + a 4-number quaternion), because rotations are stored as quaternions.

### 6. Add the `home` keyframe
- [x] Order of `qpos` follows the order joints appear in the tree: Panda (7 arm + 2 fingers = 9), then cube (7) = **16 values**.
- [x] `qpos = "0 0.392699 0 -1.9635 0 2.35619 0.785398 0.04 0.04  <cube x y z> 1 0 0 0"`
- [x] `ctrl = "0 0.392699 0 -1.9635 0 2.35619 0.785398 255"`. The actuators are **position controllers** ("go to this angle"), so ctrl must equal the home angles, or the arm will be pulled toward 0 when the sim runs. 255 = gripper open.

### 7. Set the `fixed` camera
- [x] Replace the straight-down camera with an angled view showing the arm, cube and bin, similar to `assets/media/panda_scene.png`.

### 8. Verify
- [x] Model loads; print `nq` (expect 16), `nu` (expect 8), `nkey` (expect 1).
- [x] All five names resolve: `panda_arm`, `panda_hand_tcp`, `cube`, `bin`, keyframe `home`.
- [x] Reset to `home`, simulate ~2 s, and check:
  - [x] the arm stays at home (joint error small),
  - [x] the cube rests on the table (z ≈ 0.78, not falling or jittering),
  - [x] no warnings or instabilities.
- [x] The cube and bin are within reach of the arm (distance from base < ~0.8 m), and the TCP at home is above the table.
- [x] Render from `fixed` and look at the image.

### 9. Deliverable
- [x] Run `task1/scene_construction.ipynb` (or `scene_construction.py`) to produce the final render.
- [x] Save renders to `renders/` and log results in `PROGRESS.md`.

## Results (2026-10-09)

- `python task1/verify_scene.py` (from the repo root): **all 21 checks pass**.
  - `nq=16`, `nv=15`, `nu=8`, one keyframe; every required body name resolves.
  - Home pose: base at (-0.6, 0, 0.76), TCP at (0.015, 0, 0.93). Cube is 0.52 m from the base and the bin 0.54 m, both within reach.
  - After 2 s of simulation: max arm joint error 0.009 rad, gripper still open, cube at z = 0.7798 with velocity ~1e-14 (resting, no jitter), no NaNs or MuJoCo warnings.
- Renders: `renders/task1_scene_fixed.png`, plus the executed `task1/scene_construction.ipynb`.
- Found while running the notebook: the provided `render()` helper doesn't apply the keyframe, so it draws the arm at all-zero joints (pointing straight up). The notebook's empty last cell now renders the `home` pose instead.

## Decisions and their reasons
- **Wrapper bodies instead of renaming** (`panda_arm`, `bin`): this keeps the provided files untouched except for the one TCP line.
- **4 cm cube**: fits the 8 cm gripper opening and the 9 cm bin with room to spare.
- **Fixed bin**: Task 1 doesn't ask for a movable bin. Task 3 suggests randomizing the bin position, which can be done by moving the body in code later.
