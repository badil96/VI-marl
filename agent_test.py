import argparse
import os
import pickle
import random
import torch
import copy
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2
from pettingzoo.classic import rps_v2
import mp_v0
from MADDPG import MADDPG


test_seeds = [ 530,  527,  538,  561,  141, 1217,  974, 1168,  964, 1061,  791,
        639,  335,  464,  398,  958,  792,  616,   39,  262, 1199,  376,
       1318,  609,  153,  714, 1153,  221,  449,  602,   30,  951, 1067,
       1073,  293,  911, 1474,  923,  993, 1031,  838, 1474,  434, 1269,
       1037,  190, 1403, 1364,   77,  391,  827,  377,  246,  765,  742,
        286, 1328,  547, 1490, 1439,  918, 1201,  125, 1181,  193,  805,
        224,  139,  877,  531,  421,  182,  383, 1336,  368,  303,  990,
        238,   16,  938, 1279,  305,  221,  931, 1323, 1365, 1021,  959,
        138,  254,  739, 1130, 1120, 1416,  258,  460,  370, 1317,   32,
         27] # randonly generated list for test seeds for regenertion purposes


def get_env(env_name, ep_len=25):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v2':
        new_env = simple_adversary_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_spread_v2':
        new_env = simple_spread_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_tag_v2':
        new_env = simple_tag_v2.parallel_env(num_good=1, num_adversaries=2, num_obstacles=2, max_cycles=ep_len)
    if env_name == 'rps_v2':
        new_env = rps_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'mp_v0':
        new_env = mp_v0.parallel_env()
  

    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        if env_name in ['rps_v2', 'mp_v0']:
            _dim_info[agent_id] = []  # [obs_dim, act_dim]
            _dim_info[agent_id].append(1)
            _dim_info[agent_id].append(new_env.action_space(agent_id).n)
        else:
            _dim_info[agent_id] = []  # [obs_dim, act_dim]
            _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
            _dim_info[agent_id].append(new_env.action_space(agent_id).n)

    return new_env, _dim_info



def agent_test(agents, env, episode,  results_dir, steps=25, save_gifs=False): 
    test_env, dim_info = get_env(env, steps)
    if save_gifs:
        frame_list = []  # used to save gif
        gif_dir = os.path.join(results_dir, 'gif_test')
        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)

    if env in ['rps_v2','mp_v0']: #RPS or MP
        prop_dist = {}
        for agent_id, agent in agents.agents.items():
            probs = []
            obs_env = {'rps_v2':[0,1,2], 'mp_v0':[0,1]}
            obs = np.array(np.random.choice(obs_env[env], 1500))
            with torch.no_grad():
                o = torch.from_numpy(obs).unsqueeze(1).float()
                _, logits = agent.action(o, model_out=True)
                p = torch.nn.functional.softmax(logits, dim=-1)
            prop_dist[agent_id] =  p
        return prop_dist


    if env == 'simple_adversary_v2': #simple adversary
        agent_rewards = {agent: [] for agent in test_env.agents}
        for i in test_seeds:
            states = test_env.reset(seed=int(i))
            # reward of each episode of each agent
            while test_env.agents:  # interact with the env for an episode
                actions = agents.select_action(states)
                next_states, rewards, term, trunc, infos = test_env.step(actions)       
                states = next_states
            if save_gifs:
                #generate and store gif for the last episode
                frame_list.append(Image.fromarray(test_env.render(mode='rgb_array')))
            for agent_id, r in rewards.items():  # record reward
                    agent_rewards[agent_id].append(r) 


            test_env.close()
        # save gifs for all test seeds
        if save_gifs:
            frame_list[0].save(os.path.join(gif_dir, f'out{episode}.gif'),
                         save_all=True, append_images=frame_list[1:], duration=1, loop=0)
        return agent_rewards

    if env == 'simple_tag_v2': #simple tag
        agent_rewards = {agent: [0]*100 for agent in test_env.agents}
        agent_distances = {agent: [[] for _ in range(100)] for agent in test_env.agents}
        for i in range(len(test_seeds)):
            states = test_env.reset(seed=int(test_seeds[i]))
            while test_env.agents:  # interact with the env for an episode
                actions = agents.select_action(states)
                next_states, rewards, dones, infos = test_env.step(actions)
                # frame_list.append(Image.fromarray(test_env.render(mode='rgb_array'))) #here if want to store the whole episode gifs
                states = next_states
                for agent_id, r in rewards.items():  # record reward
                    agent_rewards[agent_id][i] += r 
                for agent_id, obs in states.items():
                    agent_distances[agent_id][i].append([obs[2],obs[3]])
            if save_gifs:
                #generate and store gif for the last episode
                frame_list.append(Image.fromarray(test_env.render(mode='rgb_array'))) #here if want to store only last episode gifs
        test_env.close()
        if save_gifs:
            #safe gifs for for all test seeds
            frame_list[0].save(os.path.join(gif_dir, f'out{episode}.gif'),
                           save_all=True, append_images=frame_list[1:], duration=1, loop=0)
        return agent_rewards, agent_distances
