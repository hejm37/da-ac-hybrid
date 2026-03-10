# Distributions-as-Actions Actor-Critic (DA-AC)

Reference implementation of Distributions-as-Actions Actor-Critic (DA-AC) for hybrid control settings. See [*Distributions as Actions: A Unified Framework for Diverse Action Spaces*](https://openreview.net/forum?id=4ol71wMPY8), accepted at ICLR 2026.

> **Note:** This repository is the hybrid-control codebase. Continuous and discrete control implementations are maintained in [the main DA-AC repository](https://github.com/hejm37/da-ac).

## Implemented Algorithms

This repository includes DA-AC and some hybrid-action baselines built on TD3.

- **Distributions-as-Actions Actor-Critic (DA-AC)**
    - Goal: [`da_ac_main_goal.py`](da_ac_hybrid/da_ac_main_goal.py)
    - Platform: [`da_ac_main_platform.py`](da_ac_hybrid/da_ac_main_platform.py)
    - Catch Point: [`da_ac_main_direction_catch.py`](da_ac_hybrid/da_ac_main_direction_catch.py)
    - Hard Goal: [`da_ac_main_hard_goal.py`](da_ac_hybrid/da_ac_main_hard_goal.py)
    - Hard Move variants: [`da_ac_main_hard_move.py`](da_ac_hybrid/da_ac_main_hard_move.py)

- **Baselines**
    - [PA-DDPG](https://arxiv.org/abs/1511.04143) (TD3-based, referred to as PATD3): [`patd3_main_goal.py`](da_ac_hybrid/patd3_main_goal.py), [`patd3_main_platform.py`](da_ac_hybrid/patd3_main_platform.py), [`patd3_main_direction_catch.py`](da_ac_hybrid/patd3_main_direction_catch.py), [`patd3_main_hard_goal.py`](da_ac_hybrid/patd3_main_hard_goal.py), [`patd3_main_hard_move.py`](da_ac_hybrid/patd3_main_hard_move.py)
    - [PDQN](https://arxiv.org/abs/1810.06394) (TD3-based): [`pdqn_td3_main_goal.py`](da_ac_hybrid/pdqn_td3_main_goal.py), [`pdqn_td3_main_platform.py`](da_ac_hybrid/pdqn_td3_main_platform.py), [`pdqn_td3_main_direction_catch.py`](da_ac_hybrid/pdqn_td3_main_direction_catch.py), [`pdqn_td3_main_hard_goal.py`](da_ac_hybrid/pdqn_td3_main_hard_goal.py), [`pdqn_td3_main_hard_move.py`](da_ac_hybrid/pdqn_td3_main_hard_move.py)
    - [HHQN](https://arxiv.org/abs/1903.04959) (TD3-based): [`hhqn_td3_main_goal.py`](da_ac_hybrid/hhqn_td3_main_goal.py), [`hhqn_td3_main_platform.py`](da_ac_hybrid/hhqn_td3_main_platform.py), [`hhqn_td3_main_direction_catch.py`](da_ac_hybrid/hhqn_td3_main_direction_catch.py), [`hhqn_td3_main_hard_goal.py`](da_ac_hybrid/hhqn_td3_main_hard_goal.py), [`hhqn_td3_main_hard_move.py`](da_ac_hybrid/hhqn_td3_main_hard_move.py)

## Getting Started

### Prerequisites

- Python 3.9
- `pip`

### Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run Experiments

Example commands:

```bash
# DA-AC (Platform)
python da_ac_hybrid/da_ac_main_platform.py --seed 1

# P-TD3 (Goal)
python da_ac_hybrid/patd3_main_goal.py --seed 1

# PDQN-TD3 (Direction Catch)
python da_ac_hybrid/pdqn_td3_main_direction_catch.py --seed 1

# HHQN-TD3 (Hard Move with n=4)
python da_ac_hybrid/hhqn_td3_main_hard_move.py --action_n_dim 4 --seed 1
```

Most script-level hyperparameters are configured through each script's `argparse` options.

## Repository Structure

- [`da_ac_hybrid/`](da_ac_hybrid): training scripts, agents, wrappers, and environments
- [`da_ac_hybrid/agents/`](da_ac_hybrid/agents): DA-AC and baseline agent implementations
- [`da_ac_hybrid/common/`](da_ac_hybrid/common): shared wrappers and replay utilities
- [`da_ac_hybrid/envs/`](da_ac_hybrid/envs): Goal, Platform, and multi-agent hybrid environments
- [`results/`](results): saved models and logs
- [`requirements.txt`](requirements.txt): Python dependencies
- [`LICENSE`](LICENSE): license terms

## Citation

If you use this codebase, please cite:

```bibtex
@inproceedings{he2026distributions,
    title={Distributions as Actions: A Unified Framework for Diverse Action Spaces},
    author={Jiamin He and A. Rupam Mahmood and Martha White},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=4ol71wMPY8}
}
```

## License

This repository is distributed under the Apache License 2.0. See [`LICENSE`](LICENSE).

Most files are modified or adopted from
[TJU-DRL-LAB/self-supervised-rl](https://github.com/TJU-DRL-LAB/self-supervised-rl/blob/ece95621b8c49f154f96cf7d395b95362a3b3d4e).

Files authored in this repository and not adopted from that upstream source are:
- [`da_ac_hybrid/da_ac_main_*.py`](da_ac_hybrid)
- [`da_ac_hybrid/agents/da_ac.py`](da_ac_hybrid/agents/da_ac.py)
