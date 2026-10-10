# Onboarding Progress: Harshatha Prasanna

Branch: `ml_onboarding` (commit `b6c05b4`). Deadline: Fri 2026-10-09, 11:59 PM PT. Submitting as a zip.

## Status

| Task | Status | Notes |
|---|---|---|
| Setup | Done | Env works locally; starter scene renders |
| Task 0: XML tutorial (optional) | Not started | |
| Task 1: Scene construction | Done | All 21 checks in `task1/verify_scene.py` pass; see `task1/IMPLEMENTATION.md` |
| Task 2: Imitation learning | Done | Expert 10/10; BC policy 0/10 but reaches and grasps; limitations written up in the notebook |
| Task 3: Reinforcement learning | Done (partial success) | PPO on Colab T4: reaches + grasps (67% of steps), no cube-in-bin yet |

## Setup

- Machine: MacBook M1 Pro, 16 GB RAM, no NVIDIA GPU.
- Conda env `mujoco_cpu` from `environment.yml`, plus `pip install torch` (2.14.1).
- Versions: mujoco 3.3.4, gymnasium 0.29.1, jax 0.4.35 (CPU), brax 0.12.1, flax 0.10.4, torch 2.14.1, numpy 2.2.6.
- Verified: every package imports; a Gymnasium MuJoCo env passes `check_env`; `pip check` is clean; the starter XML loads and renders (`renders/00_starter_scene.png`, top-down view of the empty table).
- Setup problems hit:
  - The first `conda env create` failed with a network error (`HTTP 000 CONNECTION FAILED`); a retry worked.
  - The pip section of `environment.yml` upgrades numpy from the pinned 1.26.4 to 2.2.6. Tested: Gymnasium still works.
  - **PyTorch and MJX/Brax can't be imported in the same Python process** (two copies of the OpenMP runtime: conda's `libomp` and torch's bundled one). This doesn't matter in practice because Task 2 (torch) and Task 3 (MJX/Brax) are separate notebooks, but they must not share a kernel.

## Issues found in the starter repo

- `mplib` only ships Linux x86_64 wheels, so it can't be installed on Apple Silicon.
- `environment.yml` doesn't include PyTorch, which Task 2 needs.
- The provided Panda MJCF uses body names `link0`…`hand` and the bin is `simpleWoodBin`, but Tasks 2 and 3 look up `panda_arm`, `panda_hand_tcp`, `panda_hand`, `cube` and `bin`.
- There's no cube asset; it has to be written by hand.
- The `home` keyframe in the README has 9 values (arm and fingers). With a free-moving cube the model has 16 position values, so the keyframe also needs the cube's pose.
- The Colab links and Task 3's asset download use the `onboarding` branch, not `ml_onboarding`.
- `mplib` also has no build for Python 3.13, which Colab now runs, so the planner isn't available on either machine. Replaced with an IK-based expert.
- **Collisions were disabled for the whole Panda arm** in `panda_asset.xml` (the `panda` default sets `contype/conaffinity=0` and the `collision` class never re-enables them). Fixed for the `collision` class.
- **The gripper was too weak** (~2 N), so the cube slid out of the fingers. Gripper actuator gains scaled ×10 (same open/closed positions). Tested solver iterations, friction, elliptic cones and noslip first; none helped on their own.
- Task 2 evaluation needs a `TimeLimit` wrapper so failed episodes end; training data is shuffled and the learning rate raised.

## Log

- 2026-10-08: Shallow-cloned `ml_onboarding` (the full repo is ~5 GB; a plain clone failed partway). Read all task docs.
- 2026-10-09: The clone was found deleted (cause unknown) and was re-cloned. Set up and verified the env.
- 2026-10-09: Registered the `mujoco_cpu` Jupyter kernel (the notebooks ask for it by name; it wasn't registered).
- 2026-10-09: **Task 1 done.**
  - Added the Panda arm (wrapper body `panda_arm`), the bin (wrapper body `bin`) and a 4 cm free-moving `cube` to `DropCubeInBinEnv.xml`.
  - Added a `panda_hand_tcp` body between the fingertips (one line in `panda_body.xml`).
  - Added the `home` keyframe with 16 qpos values (9 arm/finger + 7 cube) and matching ctrl, so the position actuators hold the pose.
  - Set an angled `fixed` camera.
  - Verified with `task1/verify_scene.py` (all checks pass) and executed `scene_construction.ipynb`.
- 2026-10-09: **Task 2 so far.**
  - Colab setup attempted (Drive folder + setup notebook), but `mplib` can't install on Colab's Python 3.13. Switched to running Task 2 locally with an IK expert (damped least squares on MuJoCo's Jacobian).
  - Adjusted the robot model (collisions, gripper strength) as above. Expert success: **10/10**, checked visually in the video.
  - BC v1 (obs = qpos+qvel, 300 demos / 77k pairs, 100 epochs): 0/5. The policy heads toward the cube but misses by 2–5 cm and closes the gripper early (gripper outputs ranged −559 to 715).
  - BC v2 (+ hand position and hand→cube / cube→bin vectors, gripper snapped to open/closed at eval): 0/5. The gripper is clean but reaching is still 3–5 cm off and the arm drifts: covariate shift.
  - BC v3: predict joint *changes* + DART noise-injected demos → reaching error down to 0.3–0.6 cm.
  - BC v4 (final): + grasp 1 cm higher → grasps the cube in 5/5 dev rollouts but doesn't lift it (hand drifts while gripping).
  - Final notebook run (`task2/imitation_learning.ipynb`, executed end to end): expert 10/10; 261 noisy demos / 65,541 pairs; BC 0/10. Videos in `task2/videos/`.
  - BC v5 (tried, reverted): + last command and a grip timer → worse: no grasps, arm runs away. Causal confusion / copycat problem.
  - BC limitation write-up: covariate shift (DART evidence) + the expert's hidden clock (phase changes on a timer the policy can't see).
- 2026-10-09: **Task 3.**
  - Feasibility: all packages have Python 3.13 builds; the team's `brax==0.12.1` isn't compatible with MJX 3.3.4, so used `brax==0.12.5`. The full-collision scene was ~117× too slow in MJX, so made a light-collision copy (`DropCubeInBinEnv_mjx.xml` / `panda_asset_mjx.xml`, fingertip pads only; the expert still grasps 10/10).
  - Template adjustments: `_body_pos` off by one body (Brax `x` excludes the world body, so "success" was always true), `done` dtype mismatch breaking PPO's scan, `network_factory` never passed, `render` given env States, raw [-1,1] actions sent as motor targets, success ending the episode (hover exploit).
  - Staged reward (reach → grasp → lift → carry → success, plus an action penalty); checked locally that each stage raises the reward.
  - Colab run (T4, 1,024 envs, ~48 min, 3.1M steps): reward −1.2 → 461, hand–cube 46 → 4.5 cm, grasping 0 → 67% of steps, lift 0.87 cm, success 0%. Video shows grab-and-hold (local optimum). Results in `task3/results/` and the notebook.
