# Task 2 Implementation Plan: Imitation Learning (Behavior Cloning)

**Goal:** Teach the Panda to pick up the cube and drop it in the bin by **copying an expert**. The expert is a scripted controller; the learner is a small neural network.

**Runs on:** this Mac (local `mujoco_cpu` env).
**Notebook:** `task2/imitation_learning.ipynb`
**Scene:** `assets/descriptions/DropCubeInBinEnv.xml` (built in Task 1)

## Change of plan (2026-10-09), which overrides Step 0 and parts of Step 2 below
- **`mplib` can't be installed anywhere available:** it has no build for Apple Silicon or for Python 3.13, which Colab now runs (checked PyPI and its GitHub releases). So the expert is an **IK-based scripted controller** instead: damped least squares on MuJoCo's Jacobian, moving the hand in straight lines through the same 6 phases. The rest of the task (dataset, network, training, evaluation) is unchanged.
- Without `mplib`, nothing needs Linux, so **Task 2 runs locally and Colab isn't needed.** The Colab/Drive setup is kept for Task 3.
- **Two fixes to the provided robot file** (`panda/mjcf/panda_asset.xml`), both found while getting the expert to grasp:
  1. **Collisions were off for the whole arm.** The `panda` default sets `contype="0" conaffinity="0"` and the `collision` class never re-enables them, so the fingers passed through the cube. Re-enabled for the `collision` class only; visual meshes stay non-colliding.
  2. **The gripper was too weak** (~2 N squeeze), so the cube slid out of the fingers. Scaling the gripper actuator gains ×10 fixed it (same open/closed positions). Tested alternatives first: more solver iterations, higher friction, elliptic friction cones and noslip each did nothing on their own.
- Result: the **expert succeeds 10/10** (verified from the video frames, not just the success flag).

## Definition of done (from the task README)
- [x] Notebook executes without errors (all cells, run end to end locally).
- [x] A non-empty dataset file is collected (261 demos, 65,541 pairs).
- [x] The trained policy loads and runs in the environment (0/10 success, videos saved).
- [x] Can explain one BC limitation observed during rollout (covariate shift + the expert's hidden clock; see the notebook's Results cell).

A high success rate is **not** required.

## Key facts this plan relies on

| Fact | Value | Why it matters |
|---|---|---|
| Action | 8 numbers: 7 joint-angle targets + gripper (0 = closed, 255 = open) | The arm's motors are position controllers, so the planner's joint angles can be sent straight in as actions |
| Observation (starter) | `[qpos, qvel]` = 16 + 15 = 31 numbers | `qpos` already includes the cube's position, so the network can "see" the cube |
| Env step length | `frame_skip=20` × timestep 0.001 s = **0.02 s** | The planner's `time_step` should also be 0.02, so one planned waypoint = one env step |
| TCP offset | 0.1034 m from the hand, in **both** the MuJoCo scene and the planner's URDF | The planner and simulator agree on where "the hand" is |
| Robot base | body `panda_arm` at (-0.6, 0, 0.76) | Passed to `planner.set_base_pose` so the planner works in world coordinates |
| Planner API | Notebook uses 7-number poses → **mplib 0.1.x** style; 0.2.x needs `mplib.Pose` objects | Pin `mplib==0.1.1` (has a Python 3.12 build) |

## Steps

### Step 0: Colab setup (~20 min)
- [ ] Zip `assets/` and `task2/` locally and upload them to Google Drive.
- [ ] Open `imitation_learning.ipynb` from Drive in Colab and mount Drive so files survive disconnects.
- [ ] Install: `mujoco==3.3.4`, `gymnasium[mujoco]==0.29.1`, `mplib==0.1.1`, `mediapy`, `imageio`. PyTorch is preinstalled on Colab.
- [ ] Set `MUJOCO_GL=egl` (headless GPU rendering). The notebook's setup cells set `glfw`, which needs a screen Colab doesn't have.
- [ ] **Check:** the scene loads, `nq == 16`, and one frame renders from the `fixed` camera.
- [ ] **Check:** `import mplib` works and the planner builds with the notebook's link/joint names.

### Step 1: Gymnasium environment, `TrajEnv` (~45 min)
*Concept: an **environment** wraps the simulator in a standard `reset()` / `step(action)` interface, so any learning code can use it.*
- [ ] Use the second (more complete) `TrajEnv` cell; the first is an older duplicate.
- [ ] `_get_obs`: keep `[qpos, qvel]` to start (31 numbers).
- [ ] `is_grasped`: both finger pads touching the cube (checked via MuJoCo's contact list), with a TCP-to-cube distance check as a backup.
- [ ] `terminated` (success): cube center inside the bin's inner walls in x/y (±0.04 m from the bin center) and low (below the wall tops), i.e. it actually fell in.
- [ ] `reset_model`: start from `home`, then move the cube to a random spot in a **small** box (about ±5 cm) around its default position. Leave the arm at home at first.
- [ ] **Check:** reset 5 times → the cube lands in different spots, still resting on the table. Manually place the cube in the bin → `terminated` is True.

### Step 2: Expert script, `pick_cube_solution` (~2–3 h, highest risk)
*Concept: a **motion planner** computes joint-angle paths that move the hand to a target pose without collisions. It isn't learning; it's geometry.*
- [ ] Grasp orientation: reuse the TCP's orientation at the home pose (fingers pointing down) for every phase, so the gripper always approaches the same way (`tips.md`: standardize the approach).
- [ ] Phases, each planned with `plan_screw` (straight-line hand motion) and falling back to `plan_pose` if it fails:
  1. move to ~10 cm above the cube, gripper open (255)
  2. descend until the TCP is at cube height
  3. close the gripper (0) and hold for a few steps to let it squeeze
  4. lift ~15 cm
  5. move to ~15 cm above the bin center
  6. open the gripper (255) and wait a few steps for the cube to drop
- [ ] Execute each planned path one waypoint per env step: action = `[7 planned joint angles, gripper]`.
- [ ] Record `(obs, action)` at **every** step into `episode_data`.
- [ ] Return True only if the episode ends with `env.terminated`.
- [ ] **Check:** first get **one** successful episode with a recorded video, then measure the success rate over 10 episodes (target ≥ 70%).

### Step 3: Collect the dataset (~20–40 min of compute)
- [ ] Reduce `EPISODES` from 10,000 to **50** first, then increase (e.g. 200–300) if time allows.
- [ ] Reset the planner's base pose each episode (already in the template).
- [ ] Save only successful episodes to `pick_place_dataset.pkl`; copy it to Drive.
- [ ] **Check:** print the number of trajectories, the steps per trajectory, and the obs/action shapes (31 and 8).

### Step 4: Dataset class, `TrajectoryDataset` (~15 min)
*Concept: for BC the order doesn't matter. Every (observation, action) pair is one independent training example.*
- [ ] Flatten all episodes into one list of pairs, convert to `float32` tensors once in `__init__`.
- [ ] `__len__` = number of pairs; `__getitem__` returns `(obs[i], action[i])`.
- [ ] **Check:** `len(dataset) > 0`, and one sample has shapes `(31,)` and `(8,)`.

### Step 5: Policy network, `Actor` (~20 min)
*Concept: the network outputs a **Gaussian** (a mean action plus a spread, `log_std`) rather than a single answer. The spread says how unsure it is.*
- [ ] MLP trunk (e.g. 31 → 256 → 256, ReLU) with two heads: `mean` (8) and `log_std` (8), `log_std` clamped to [-5, 2].
- [ ] **Normalize** inputs/outputs using dataset mean/std (joint angles ~±3 and gripper 0–255 are on very different scales). Store the stats inside the model so evaluation uses them too.
- [ ] **Check:** a forward pass on a batch gives `mean` and `log_std` of shape `(batch, 8)`.

### Step 6: Training loop (~30 min + training time)
*Concept: **negative log-likelihood (NLL)** loss rewards the network for putting high probability on the expert's action.*
- [ ] Use the provided `nll_loss`, the Adam optimizer, and **`shuffle=True`** (as `tips.md` recommends).
- [ ] Raise the learning rate from 3e-5 to ~3e-4 if loss decreases too slowly.
- [ ] Every few epochs: run `eval_policy` (deterministic mean action), print loss and success, and save `best_policy.pth` when success improves (tie-break on loss).
- [ ] **Check:** the loss curve goes down; the checkpoint file exists and is copied to Drive.

### Step 7: Evaluate and explain (~30 min)
- [ ] Load `best_policy.pth`, run 5–10 episodes, record videos, report the success rate.
- [ ] Watch the videos and describe what goes wrong.
- [ ] Write up one BC limitation, most likely **covariate shift**: the network only saw the expert's perfect paths, so a small error puts it in states it never saw, and the errors compound.

### Step 8: Wrap-up
- [ ] Download the executed notebook, dataset size, success rates and videos back into the local repo.
- [ ] Log results in `PROGRESS.md`.

## Fallback (if time runs short)
Submit whatever stage is reached, in this order of value: a working env → an expert that succeeds sometimes (with video) → a small dataset → a trained policy even if it never succeeds → a written explanation of covariate shift. Record clearly what was and wasn't finished.

## Risks
- **Planner failures** (unreachable pose, collision, solver not converging): reduce cube randomization, raise the approach height, and retry.
- **Gripper slipping**: hold the closed gripper for more steps before lifting; lift more slowly.
- **Colab disconnects**: keep everything on Drive and save the dataset and checkpoints as soon as they're made.
- **Time**: if the expert takes much longer than 3 h, switch to the fallback.
