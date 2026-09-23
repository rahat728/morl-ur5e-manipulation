"""Sweep the UR5e shoulder_pan joint in the viser web viewer (localhost:8080).

Same mechanics as the headless-verified sweep: command a sine offset on
actuator 0 (shoulder_pan) on top of the home pose and step the sim.
Steps ~10 sim steps per render push to keep viser updates realtime.
"""

import math
import time

import mujoco
import viser

from mjlab.entity.entity import Entity
from mjlab.viewer.viser import MjlabViserScene
from ur5e_constants import get_ur5e_robot_cfg


def main() -> None:
  robot = Entity(get_ur5e_robot_cfg())
  model = robot.spec.compile()
  data = mujoco.MjData(model)

  key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
  mujoco.mj_resetDataKeyframe(model, data, key_id)

  print("Actuator order:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)])
  print("Home qpos:", [round(float(x), 4) for x in data.qpos])

  home_ctrl = data.ctrl.copy()
  target = home_ctrl.copy()

  server = viser.ViserServer(port=8080, label="ur5e-move")
  scene = MjlabViserScene(server=server, mj_model=model, num_envs=1)
  scene.update_from_mjdata(data)
  print("UR5e sweep live. Open http://localhost:8080 in your browser.", flush=True)

  t0 = time.time()
  while True:
    for _ in range(10):
      t = time.time() - t0
      target[0] = home_ctrl[0] + 0.5 * (1 + math.sin(t))
      data.ctrl[:] = target
      mujoco.mj_step(model, data)
    scene.update_from_mjdata(data)
    time.sleep(0.02)


if __name__ == "__main__":
  main()