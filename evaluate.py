import argparse
import os
import random
import torch
import pickle


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from MADDPG import MADDPG
from MATD3 import MATD3
from main import get_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='simple_adversary_v2', help='name of the env',
                        choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2', 'rps_v2'])
    parser.add_argument('--folder', type=str, help='name of the folder where model is saved')
    parser.add_argument('--episode-num', type=int, default=10, help='total episode num during evaluation')
    parser.add_argument('--episode-length', type=int, default=50, help='steps per episode')
    parser.add_argument('--algo', type=str, default='MADDPG', help='algorithm name', choices=['MADDPG','MATD3'])

    args = parser.parse_args()
    
    seed = 540
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    model_dir = os.path.join('./results', args.env_name, args.algo, args.folder)
    print(model_dir)
    assert os.path.exists(model_dir)
    gif_dir = os.path.join(model_dir, 'gif')
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    gif_num = len([file for file in os.listdir(gif_dir)])  # current number of gif

    env, dim_info = get_env(args.env_name, args.episode_length)
    optimizer = 'Adam' if 'Adam' in args.folder else 'SGD'
    print('optimizer: ', optimizer)
    if args.algo == 'MADDPG':
        model = MADDPG.load(optimizer, dim_info, os.path.join(model_dir, 'model.pt'))
    elif args.algo == 'MATD3':
        model = MATD3.load(dim_info, optimizer, os.path.join(model_dir, 'model.pt'))

    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent: np.zeros(args.episode_num) for agent in env.agents}
    agent_distances = {agent: [] for agent in env.agents}
    for episode in range(args.episode_num):
        states = env.reset(seed = seed)
        agent_reward = {agent: 0 for agent in env.agents}  # agent reward of the current episode
        frame_list = []  # used to save gif
        while env.agents:  # interact with the env for an episode
            actions = model.select_action(states)
            next_states, rewards, dones, infos = env.step(actions)
            frame_list.append(Image.fromarray(env.render(mode='rgb_array')))
            states = next_states

            for agent_id, reward in rewards.items():  # update reward
                agent_reward[agent_id] += reward
            # for agent_id, obs in states.items():
            #     agent_distances[agent_id].append([obs[2],obs[3]])

        env.close()
        message = f'episode {episode + 1}, '
        # episode finishes, record reward
        for agent_id, reward in agent_reward.items():
            episode_rewards[agent_id][episode] = reward
            message += f'{agent_id}: {reward:>4f}; '
        print(message)
        # save gif
        frame_list[0].save(os.path.join(gif_dir, f'out{gif_num + episode + 1}.gif'),
                           save_all=True, append_images=frame_list[1:], duration=1, loop=0)
    

    with open(os.path.join(model_dir, 'evaluation_rewards.pkl'), 'wb') as f:  # save training data
            pickle.dump(episode_rewards, f)
    with open(os.path.join(model_dir, 'evaluation_distances.pkl'), 'wb') as f:  # save training data
            pickle.dump(agent_distances, f)
    # training finishes, plot reward
    # fig, ax = plt.subplots()
    # x = range(1, args.episode_num + 1)
    # for agent_id, rewards in episode_rewards.items():
    #     ax.plot(x, rewards, label=agent_id)
    # ax.legend()
    # ax.set_xlabel('episode')
    # ax.set_ylabel('reward')
    # total_files = len([file for file in os.listdir(model_dir)])
    # title = f'evaluate result of {args.algo} solve {args.env_name} {total_files - 3}'
    # ax.set_title(title)
    # plt.savefig(os.path.join(model_dir, title))
