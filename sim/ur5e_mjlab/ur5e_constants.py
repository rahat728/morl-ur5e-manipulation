"""UR5e constants and entity config for mjlab.

Mirrors mjlab's own asset_zoo pattern (see i2rt_yam/yam_constants.py), but
uses XmlActuatorCfg to reuse the position actuators already tuned in the
mujoco_menagerie UR5e MJCF, rather than re-deriving PD gains by hand.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

# Adjust this path to wherever you placed the copied mujoco_menagerie assets
# inside your project repo, e.g. assets/universal_robots_ur5e/ur5e.xml
THIS_DIR = Path(__file__).parent
UR5E_XML: Path = THIS_DIR / "assets" / "universal_robots_ur5e" / "ur5e.xml"
assert UR5E_XML.exists(), f"UR5e MJCF not found at {UR5E_XML}"


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(UR5E_XML))
  # The menagerie UR5e MJCF ships every geom unnamed (name == ""),
  # distinguished only by `class` (visual / collision / eef_collision).
  # CollisionCfg matches by name regex, so we assign explicit names to
  # the collision-class geoms here before mjlab ever sees the spec.
  col_idx = 0
  for g in spec.geoms:
    cls = g.classname.name if g.classname else ""
    if cls in ("collision", "eef_collision"):
      g.name = f"{cls}_{col_idx}"
      col_idx += 1
  return spec


##
# Actuator config.
##

# The MJCF already defines tuned position actuators (<general> elements with
# gainprm/biasprm) for all 6 joints. Wrap them as-is instead of guessing gains.
ARM_ACTUATORS = (
  XmlActuatorCfg(
    target_names_expr=(
      "shoulder_pan_joint",
      "shoulder_lift_joint",
      "elbow_joint",
      "wrist_1_joint",
      "wrist_2_joint",
      "wrist_3_joint",
    ),
    command_field="position",
  ),
)

##
# Keyframe config.
##

# The MJCF ships a "home" keyframe: qpos="-1.5708 -1.5708 1.5708 -1.5708 -1.5708 0"
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "shoulder_pan_joint": -1.5708,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Matches the names we assign in get_spec(): "collision_0", "eef_collision_8", etc.
# Start simple: enable all collision geoms. Refine later (e.g. disable
# self-collision pairs that are never physically possible) once this loads.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*collision.*",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=0,
)

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=ARM_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_ur5e_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


if __name__ == "__main__":
  # Quick visual sanity check: does the robot load and look right?
  # Run with: python ur5e_constants.py
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_ur5e_robot_cfg())
  viewer.launch(robot.spec.compile())