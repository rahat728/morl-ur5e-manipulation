# Vision-Guided MORL for UR5e Manipulation

4-month fast-track: DP3 point-cloud encoder → multi-head MO critic (GCR-PPO-style) → preference-conditioned diffusion policy → UR5e, CMDP safety.

## Repo structure
- `calibration/` — robot kinematics calibration, camera extrinsics
- `configs/` — launch configs, network settings, hyperparameters
- `checkpoints/` — trained model checkpoints (gitignored, sync via external storage)
- `scripts/` — utility scripts (sim/real bring-up, data collection)
- `docs/` — plan docs, notes

## Machines
- **Laptop**: real UR5e control (ROS2 Jazzy, UR driver, MoveIt)
- **Lab PC**: IsaacLab/mjlab training (pending access)