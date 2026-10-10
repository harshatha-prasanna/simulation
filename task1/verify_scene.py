"""Checks for the Task 1 scene: structure, names, keyframe, physics stability, reach.

Run from the repo root:  python task1/verify_scene.py
"""
import os
import sys

import imageio
import mujoco as mj
import numpy as np

XML = os.path.abspath("assets/descriptions/DropCubeInBinEnv.xml")
RENDER_DIR = os.path.abspath("renders")

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


model = mj.MjModel.from_xml_path(XML)
data = mj.MjData(model)

# --- Structure ---
print(f"nq={model.nq} nv={model.nv} nu={model.nu} nkey={model.nkey} nbody={model.nbody}")
check(model.nq == 16, "nq == 16 (9 Panda + 7 cube)")
check(model.nv == 15, "nv == 15 (9 Panda + 6 cube)")
check(model.nu == 8, "nu == 8 actuators (7 arm + 1 gripper)")
check(model.nkey == 1, "exactly one keyframe")

# --- Names later tasks look up ---
for name in ["panda_arm", "panda_hand_tcp", "cube", "bin"]:
    check(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) >= 0, f"body '{name}' exists")
check(mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home") >= 0, "keyframe 'home' exists")
check(mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "fixed") >= 0, "camera 'fixed' exists")

# --- Reset to home and inspect positions ---
mj.mj_resetDataKeyframe(model, data, model.key("home").id)
mj.mj_forward(model, data)
home_qpos = data.qpos.copy()
base = data.body("panda_arm").xpos.copy()
tcp = data.body("panda_hand_tcp").xpos.copy()
cube = data.body("cube").xpos.copy()
bin_ = data.body("bin").xpos.copy()
print(f"base={base.round(3)} tcp={tcp.round(3)} cube={cube.round(3)} bin={bin_.round(3)}")

check(tcp[2] > 0.76 + 0.1, f"TCP at home is above the table (z={tcp[2]:.3f})")
for label, p in [("cube", cube), ("bin", bin_)]:
    d = np.linalg.norm(p - base)
    check(d < 0.8, f"{label} within reach of the base (dist={d:.3f} m < 0.8)")
check(np.linalg.norm(cube[:2] - bin_[:2]) > 0.15, "cube does not start inside/next to the bin")

# --- Physics: simulate 2 s from home ---
render_frames = []
steps = int(2.0 / model.opt.timestep)
cube_z = []
for i in range(steps):
    mj.mj_step(model, data)
    cube_z.append(data.body("cube").xpos[2])
check(not np.any(np.isnan(data.qpos)), "no NaNs after 2 s")
check(all(w.number == 0 for w in data.warning), "no MuJoCo warnings")

arm_err = np.abs(data.qpos[:7] - home_qpos[:7]).max()
check(arm_err < 0.02, f"arm holds home pose (max joint error {arm_err:.4f} rad)")
finger = data.qpos[7:9]
check(np.all(np.abs(finger - 0.04) < 0.005), f"gripper stays open (fingers {finger.round(4)})")

cz = data.body("cube").xpos[2]
check(abs(cz - 0.78) < 0.005, f"cube rests on table (z={cz:.4f}, expected ~0.78)")
jitter = np.std(cube_z[-500:])
check(jitter < 1e-4, f"cube is still, not jittering (z std over last 0.5 s = {jitter:.2e})")
cube_speed = np.linalg.norm(data.qvel[9:15])
check(cube_speed < 1e-2, f"cube velocity ~0 (|qvel|={cube_speed:.2e})")

# --- Render from the fixed camera (at home) ---
os.makedirs(RENDER_DIR, exist_ok=True)
mj.mj_resetDataKeyframe(model, data, model.key("home").id)
mj.mj_forward(model, data)
with mj.Renderer(model, 480, 640) as r:
    r.update_scene(data, "fixed")
    out = os.path.join(RENDER_DIR, "task1_scene_fixed.png")
    imageio.imwrite(out, r.render())
print("saved", out)

print()
print("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED")
sys.exit(1 if failures else 0)
