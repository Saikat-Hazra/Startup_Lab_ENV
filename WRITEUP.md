# Startup Lab Env Writeup

## Problem
Autonomous startup agents often fail due to short-term decision bias, weak memory of past outcomes, and repeated costly actions.

## Approach
- Implemented a startup simulation environment with multi-agent competition dynamics.
- Added memory and reflection modules to capture outcomes and suggest strategy shifts.
- Exposed runtime controls through a FastAPI backend and interactive frontend.

## Training Notes
- The training pipeline runs from `training/train.py`.
- Artifacts are exported to `training/training_output/`.
- Mandatory validation plots are generated as:
  - `training/training_output/reward_curve.png`
  - `training/training_output/loss_curve.png`

## Outcome
The final submission package is organized for validator access with root `Dockerfile`, root `inference.py`, OpenEnv config, and linked deliverables from `README.md`.
