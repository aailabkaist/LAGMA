# LAGMA: LAtent Goal-guided Multi-Agent Reinforcement Learning

# Note
This codebase accompanies the paper submission "**LAGMA: LAtent Goal-guided Multi-Agent Reinforcement Learning**" and is based on [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac) which are open-sourced.
The paper is accepted by [ICML2024](https://icml.cc/Conferences/2024/).

PyMARL is [WhiRL](http://whirl.cs.ox.ac.uk)'s framework for deep multi-agent reinforcement learning.

# Run an experiment
To train LAGMA on SC2 (dense reward) setting tasks, run the following command:
```
python3 src/main.py --config=lagma_sc2 --env-config=sc2 with env_args.map_name=5m_vs_6m
```

To train LAGMA on SC2 (sparse reward) setting tasks, run the following command:
```
python3 src/main.py --config=lagma_sc2_sparse_3m --env-config=sc2_sparse with env_args.map_name=3m
```

To train LAGMA on GRF (sparse reward) setting tasks, run the following command:
```
python3 src/main.py --config=academy_3_vs_1_with_keeper --env-config=lagma_grf_3_vs_1WK
```

# Publication
If you find this repository useful, please cite our paper:
```
@inproceedings{na2024lagma,
  title={LAGMA: LAtent Goal-guided Multi-Agent Reinforcement Learning},
  author={Na, Hyungho and and Moon, Il-chul},
  booktitle={The Forty-first International Conference on Machine Learning},
  year={2024}
}
```
