
# Installation
- install requirements from requirements.txt file:

   ```bash
    pip install -r requirements.txt

# Usage
- To run experiments to train a model run 'main.py' with desired configurations, for example:
  
  ```bash
  python main.py rps_v2 --episode_num 100000 --tau 0.01 --buffer_capacity 2500000 --use_target_policy_smoothing True --algo MATD3 --tau 0.01 --lookahead True --lookahead_alpha 0.3 --lookahead_step_1 500

- To evaluate learned models after training is done run 'evaluate.py', for example:
  ```bash
  python evaluate.py --env_name rps_v2 --algo MATD3 --folder optimizer:Adam-lr0.001-Lookahead:TrueAlpha:0.5Steps:204004000/seed391

# References:
- MADDPG paper: [Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).
- MATD3 paper:[Reducing Overestimation Bias in Multi-Agent Domains Using Double Centralized Critics] https://arxiv.org/abs/1910.01465
- MADDPG code: base implementation for MADDPG was taken from [maddpg-pettingzoo-pytorch](https://github.com/Git-123-Hub/maddpg-pettingzoo-pytorch.git).
- Extragradient code: implementation for Extragradient from [Variational-Inequality-GAN](https://github.com/GauthierGidel/Variational-Inequality-GAN.git)

