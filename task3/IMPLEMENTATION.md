# Task 3 Implementation Plan: Reinforcement Learning (PPO on MJX)

**Goal:** Train a policy with PPO to pick up the cube and drop it in the bin using **reward feedback only** (no expert).
**Runs on:** Google Colab T4 GPU (MJX + Brax), with outputs saved to Google Drive (`tritondroid/task3`).
**Notebook:** `task3/reinforcement_learning.ipynb`
**Time box:** ~2.5 h total; training run ~40 min.

## Definition of done (from the task README)
- [x] Environment compiles and trains with PPO end-to-end (Colab T4, 3.1M steps).
- [x] Rollout video runs with the trained policy (`results/ppo_rollout.mp4`).
- [~] Success metric increases relative to random initialization: every stage metric improves strongly (reach 46→4.5 cm, grasp 0→67% of steps), but strict cube-in-bin success stays 0% in this budget.
- [x] Can explain how each reward term affects behavior (table in the notebook's last cell).

## Steps

### 1. Feasibility checks (done)
- [x] **Python 3.13 on Colab:** JAX 0.4.35 (+ CUDA 12 plugin), MuJoCo/MJX 3.3.4, Brax, Flax, Optax and Orbax all have 3.13 builds at the versions tested locally, so Colab runs the same stack.
- [x] **Version compatibility:** `brax==0.12.1` + `mujoco-mjx==3.3.4` fails (`Data.__init__() got an unexpected keyword argument 'contact'`). Fixed with `brax==0.12.5`, the first release after MJX 3.3.4 (installed with `--no-deps`, so nothing else changed).
- [x] **MJX speed:** with the arm's collision meshes on (needed in Task 2), MJX had 1,928 contact slots, a 91 s compile and 344 ms/step on CPU. A Task 3-only light-collision scene (`DropCubeInBinEnv_mjx.xml` + `panda_asset_mjx.xml`, where only the fingertip pads, cube, table, bin and floor collide) gives 668 slots, an 8 s compile and 2.9 ms/step. The Task 2 expert still grasps 10/10 on it.

### 2. Environment, `PickAndPlace` (done, tested locally on CPU)
- [x] `_init_env`: body IDs (`panda_hand_tcp`, `cube`, `bin`), fingertip pad geoms for the grasp check, action scaling.
- [x] `reset`: cube ±5 cm and arm ±0.02 rad randomization; motor targets start at the start pose with the gripper open. The bin has no joint, so it stays fixed.
- [x] `step`: actions in [-1, 1] → arm joint-target changes (≤ 0.04 rad/step) and a 0–255 gripper command.
- [x] `check_grasp`: contact-based, so the cube must touch a pad on both fingers.
- [x] `check_success`: cube inside the bin walls, below the rim, and released.
- [x] `_compute_reward`: staged reach → grasp → lift → carry → success, plus a small action penalty.
- [x] **Template adjustments:** (a) `_body_pos` used `x.pos[body_id]`, but Brax's `x` excludes the world body, which returned the *next* body's position (spotted because "success" was always true); (b) `reset` returned `done` as bool but `step` as float, breaking PPO's `scan`; (c) `network_factory` was built but never passed to `ppo.train`; (d) `env.render` was given env States instead of physics states.
- [x] **Checks:** random actions give reward ≈ 0 and no success; cube placed in the bin → success; hand at cube 0.9 → gripped 1.96 → lifted 10 cm 4.07 → in bin 10.0.
- [x] Tiny local PPO run (20k steps): reward −0.79 → 9.55 and hand–cube distance 0.31 → 0.18 m, so the whole pipeline (train → eval → inference → render) works.

### 3. Colab run
- [x] Setup cell: mount Drive, clone the team repo, re-apply our scene changes with code (verified byte-identical to the local files), install pinned versions.
- [x] Speed test (1M steps) → size the main run to ~40 min.
- [x] Main PPO run with checkpoints saved to Drive at every evaluation.
- [x] Training curves + rollout video saved to Drive.

### 4. Results and write-up
- [x] Compare the untrained policy (first eval) with the trained policy: reward, hand–cube distance, grasp fraction, success fraction.
- [x] Copy results into the notebook and `PROGRESS.md`.

## Config changes
| Setting | Template | Used | Why |
|---|---|---|---|
| `num_timesteps` | 10M | from measured speed × 40 min | fit the time box |
| `episode_length` | 500 | 300 | 3 s is plenty (the expert needs ~2.5 s); shorter episodes = more resets and variety |
| `unroll_length` | 50 | 10 | more PPO updates per env step under a short budget |
| `num_evals` | 100 | 20 | fewer pauses for evaluation |

## Outcome
- 1,205 env steps/s on a T4 → 3.1M steps in ~48 min. Reward −1.2 → 461; hand–cube distance 46 → 4.5 cm; grasping 0 → 67% of steps; lift starting (0.87 cm avg); success 0%.
- Video: reaches and grasps quickly, then holds the cube near the table (a local optimum). Next steps are in the notebook's Results section.
- Extra Colab fixes needed: explicit `jax-cuda12-plugin` pin (avoid jaxlib 0.4.34) and `nvidia-cuda-nvcc-cu12==12.6.85` (12.9 breaks `import jax` on 0.4.35).
