"""Push the compiled UR5e to the viser web viewer (localhost:8080).

Static single-frame render: no RL env, no warp/GPU. Run with the cached
mjlab venv and the repo root on PYTHONPATH:

  PYTHONPATH=/home/testuser/morl-ur5e-manipulation \
    ~/.cache/uv/archive-v0/Uq2E5JX04Mzu5F1k/bin/python sim/ur5e_mjlab/view_ur5e.py
"""

import time

import mujoco
import viser

from mjlab.entity import Entity
from mjlab.viewer.viser import MjlabViserScene
from sim.ur5e_mjlab.ur5e_constants import get_ur5e_robot_cfg


def main() -> None:
  robot = Entity(get_ur5e_robot_cfg())
  model = robot.compile()
  data = mujoco.MjData(model)

  kf = next(i for i in range(model.nkey) if model.key(i).name == "init_state")
  mujoco.mj_resetDataKeyframe(model, data, kf)
  mujoco.mj_forward(model, data)

  server = viser.ViserServer(port=8080, label="ur5e")
  scene = MjlabViserScene(server=server, mj_model=model, num_envs=1)
  scene.update_from_mjdata(data)
  print("UR5e pushed to viser. Open http://localhost:8080 in your browser.", flush=True)

  while True:
    time.sleep(1.0)


if __name__ == "__main__":
  main()