"""MOMDP reward terms for UR5e manipulation.

Implements the three objectives from docs/momdp_spec.md:
  r_success  -- task success (reaching a target position)
  r_energy   -- negative joint power (torque * velocity)
  r_smooth   -- negative joint-space jerk/acceleration penalty

Follows mjlab's own manipulation-task reward conventions
(see mjlab/tasks/manipulation/mdp/rewards.py) -- Gaussian kernels over
squared position error, not the linear-distance sketch in the first draft
of momdp_spec.md. This file supersedes that draft's exact formulas; the
spec doc's *structure* (3 objectives, vector reward, CMDP bounds) is
unchanged.

Requires the robot entity to expose the "attachment_site" site (present in
the menagerie UR5e MJCF as the tool-mount / end-effector point).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def r_success(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Gaussian reaching reward: exp(-||p_ee - p_target||^2 / std^2).

  Mirrors mjlab's staged_position_reward / bring_object_reward pattern.
  `command_name` must resolve (via env.command_manager) to a command object
  exposing `.target_pos` in world frame -- define this alongside the task's
  command manager config, not here.
  """
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  position_error = torch.sum(
    torch.square(command.target_pos - ee_pos_w), dim=-1  # type: ignore[attr-defined]
  )
  return torch.exp(-position_error / std**2)


def r_energy(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Negative joint power: -sum(|actuator_force * joint_vel|).

  Uses `actuator_force` (scalar actuation-space output), not `joint_torques`
  (deliberately unimplemented in mjlab -- ambiguous for non-trivial
  actuation, see entity/data.py). Valid here because the UR5e MJCF defines
  one direct position actuator per joint -- actuation space == joint space,
  so actuator_force and joint_vel index-align 1:1 without remapping.
  `actuator_force` is indexed by actuator space (`actuator_ids`), while
  `joint_vel` is indexed by joint space (`joint_ids`); those only align
  because of the 1:1 actuator/joint mapping above.
  """
  robot: Entity = env.scene[asset_cfg.name]
  power = robot.data.actuator_force[:, asset_cfg.actuator_ids] * robot.data.joint_vel[:, asset_cfg.joint_ids]
  return -torch.sum(torch.abs(power), dim=-1)


def r_smooth(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Negative joint-space jerk/acceleration penalty: -||qddot||^2.

  Direct analogue of mjlab's joint_velocity_hinge_penalty pattern, but
  penalizing acceleration magnitude unconditionally rather than only above
  a hinge threshold -- appropriate here since smoothness is one of our
  named MORL objectives, not a soft regularizer on top of a single-scalar
  reward.
  """
  robot: Entity = env.scene[asset_cfg.name]
  joint_acc = robot.data.joint_acc[:, asset_cfg.joint_ids]
  return -torch.sum(torch.square(joint_acc), dim=-1)