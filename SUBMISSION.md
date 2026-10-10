# ML/Sim Onboarding Submission: Harshatha Prasanna

Branch `ml_onboarding` (commit `b6c05b4`). Tasks 1–3 attempted; Task 0 (optional tutorial) skipped.

| Task | Status | Headline result | Where to look |
|---|---|---|---|
| 1. Scene construction | ✅ Done | Panda + cube + bin + `home` keyframe; 21/21 automated checks pass | `assets/descriptions/DropCubeInBinEnv.xml`, `task1/verify_scene.py`, `renders/` |
| 2. Imitation learning | ✅ Done (policy 0% success) | Expert 10/10; BC policy reaches within ~0.5 cm and grasps, but doesn't lift; limitations analysed | `task2/imitation_learning.ipynb` (Results cell), `task2/videos/` |
| 3. Reinforcement learning | ✅ Done (partial) | PPO, 3.1M steps on a T4: hand–cube 46 → 4.5 cm, grasping 0 → 67% of steps; no cube-in-bin yet | `task3/reinforcement_learning.ipynb` (Results cell), `task3/results/` |

Each task folder has an `IMPLEMENTATION.md` with the plan, checks and decisions. `PROGRESS.md` is the full log.

## Changes to the provided files (and why)
- **`panda_asset.xml`, collisions:** enabled for the `collision` class (the `panda` default sets `contype/conaffinity=0`), so the fingers can make contact with the cube.
- **`panda_asset.xml`, gripper strength:** gains scaled ×10 with the same open/closed positions. At about 2 N the cube slid out of the grip; more solver iterations, higher friction, elliptic cones and noslip were each tested first.
- **`panda_body.xml`:** added a `panda_hand_tcp` body (fingertip midpoint), matching the planner URDF's 0.1034 m offset.
- **Task 3 only:** `DropCubeInBinEnv_mjx.xml` + `panda_asset_mjx.xml`, a light-collision copy where only the fingertip pads collide (MJX ran ~117× slower with the arm meshes).

## Adjustments to the templates
- **Task 2 expert:** `mplib` isn't available for Apple Silicon or Python 3.13 (Colab's current version), so the expert is an IK controller (damped least squares on MuJoCo's Jacobian, straight-line hand motion), equivalent to `plan_screw` without collision checking. Task 2 runs locally.
- **Task 3 versions:** `brax==0.12.5` (the repo's `0.12.1` isn't compatible with `mujoco-mjx==3.3.4`). On Colab: an explicit `jax-cuda12-plugin==0.4.35` and `nvidia-cuda-nvcc-cu12==12.6.85` (see the notebook setup cell).
- **Task 2 template:** `TimeLimit` wrapper added for evaluation so failed episodes end; training data shuffled.
- **Task 3 template:**
  - `_body_pos` reads `xpos` (Brax's `x` leaves out the world body).
  - `done` is a float in `reset` and `step`, as PPO's scan requires.
  - `network_factory` is passed to `ppo.train`.
  - `render` receives physics states.
  - Actions are scaled from [-1, 1] to joint-target changes and a 0–255 gripper command.
  - Success is rewarded every step instead of ending the episode.

## How to run
- **Tasks 1–2 (local):** `conda env create -f environment.yml`, then `pip install torch "imageio[ffmpeg]" moviepy==1.0.3` and `pip install --no-deps brax==0.12.5`. Then `python task1/verify_scene.py` from the repo root and run `task2/imitation_learning.ipynb` (~5 min on CPU). `task2/pick_place_dataset.pkl` (19 MB) isn't included; the notebook regenerates it in ~1 min.
- **Task 3 (Colab GPU):** put `assets/descriptions/DropCubeInBinEnv.xml` in `MyDrive/tritondroid/`, open `task3/reinforcement_learning.ipynb` in Colab with a T4, and run all.
