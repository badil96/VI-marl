import argparse
import os
import pickle
import random

#import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2
from pettingzoo.classic import rps_v2
import torch
import copy
from PIL import Image

import mp_v0

from MADDPG import MADDPG
from MATD3 import MATD3

from agent_test import agent_test

    
def lookahead_avg(param, param_k, alpha=0.5):
  with torch.no_grad(): return param + alpha * (param_k - param)

def lookahead(old_actor, new_actor):
  for (old_name, old_param), (new_name, new_param) in zip(old_actor.named_parameters(), new_actor.named_parameters()):
    new_param.data = lookahead_avg(old_param.data, new_param.data)


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

def get_running_reward(arr: np.ndarray, window=100):
            """calculate the running reward, i.e. average of last `window` elements from rewards"""
            running_reward = np.zeros_like(arr)
            for i in range(window - 1):
                running_reward[i] = np.mean(arr[:i + 1])
            for i in range(window - 1, len(arr)):
                running_reward[i] = np.mean(arr[i - window + 1:i + 1])
            return running_reward



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='simple_adversary_v2', help='name of the env',
                        choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2', 'rps_v2','mp_v0'])
    parser.add_argument('--episode_num', type=int, default=30000,
                        help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=25, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=100,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=1024,
                        help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.001, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.001, help='learning rate of critic')

    parser.add_argument('--algo', type=str, default='MADDPG', help='training algorithm', choices=['MADDPG', 'MATD3'])
    parser.add_argument('--optimizer', type=str, default='Adam', help='base optimizer for training', choices=['Adam', 'SGD', 'ExtraAdam', 'OptimisticGD'])
    parser.add_argument('--policy_freq', type=float, default=2, help='policy update frequency')
    parser.add_argument('--use_target_policy_smoothing', type=bool, default=False, help='wheather to add noise to target actions when updating critics for policy smoothing or not')
    parser.add_argument('--lookahead', type=bool, default=False, help='wheather to use lookahead or not')
    parser.add_argument('--lookahead_alpha', type=float, default=0.5, help='lookahead alpha')
    parser.add_argument('--lookahead_step_1', type=int, default=20, help='lookahead step interval')
    parser.add_argument('--lookahead_step_2', type=int, default=None, help='nested lookahead step interval')
    parser.add_argument('--lookahead_step_3', type=int, default=None, help='2nd nested lookahead step interval')
    args = parser.parse_args()

    


    seeds = [ 77, 391,  827,  377,  246, 64, 210 ,  379,  832,  1155]
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # create folder to save result
        env_dir = os.path.join('./results', args.env_name, args.algo, 'optimizer:'+ args.optimizer +'-lr'+ str(args.actor_lr)+ '-Lookahead:'+str(args.lookahead)+'Alpha:'+ str(args.lookahead_alpha)+ 'Steps:'+ str(args.lookahead_step_1)+ str(args.lookahead_step_2)+str(args.lookahead_step_3))
        if not os.path.exists(env_dir):
            os.makedirs(env_dir)
        total_files = len([file for file in os.listdir(env_dir)])
        result_dir = os.path.join(env_dir, 'seed'+f'{seed}')
        os.makedirs(result_dir)

        env, dim_info = get_env(args.env_name, args.episode_length)

        if args.algo == 'MADDPG':
            model = MADDPG(args.optimizer, dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr,
                     result_dir)
        elif args.algo == 'MATD3':
            model = MATD3(args.optimizer, dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, args.policy_freq, args.use_target_policy_smoothing, result_dir)
        
        step = 0  # global step counter
        agent_num = env.num_agents
        # reward of each episode of each agent
        episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}
        test_scores_dict = {}
        test_distances_dict = {}
        model.logger.info(f'Start training {args.algo} on {args.env_name} with base optimizer {args.optimizer} and seed {seed}')    
        # LA
        if args.lookahead:
            model.logger.info('Lookahead is enabled with alpha {} and steps {}, {}, {}'.format(args.lookahead_alpha, args.lookahead_step_1, args.lookahead_step_2, args.lookahead_step_3))
            la_step = args.lookahead_step_1
            la_ss_step = args.lookahead_step_2
            la_sss_step = args.lookahead_step_3
            la_actors = {}
            la_ss_actors = {}
            la_sss_actors = {}
            la_critics = {}
            la_ss_critics = {}
            la_sss_critics = {}
            la_critics1 = {}
            la_ss_critics1 = {}
            la_sss_critics1 = {}
            la_critics2 = {}
            la_ss_critics2 = {}
            la_sss_critics2 = {}
            for agent_id, agent in model.agents.items():
            # append every actor params to the list
                la_actors[agent_id] = copy.deepcopy(agent.actor)
                la_ss_actors[agent_id] = copy.deepcopy(agent.actor) if la_ss_step is not None else None
                la_sss_actors[agent_id] = copy.deepcopy(agent.actor) if la_sss_step is not None else None

                # append every critic params to the list
                if args.algo == 'MADDPG':
                    la_critics[agent_id] = copy.deepcopy(agent.critic)
                    la_ss_critics[agent_id] = copy.deepcopy(agent.critic) if la_ss_step is not None else None
                    la_sss_critics[agent_id] = copy.deepcopy(agent.critic) if la_sss_step is not None else None

                if args.algo == 'MATD3':
                    la_critics1[agent_id] = copy.deepcopy(agent.critic1)
                    la_ss_critics1[agent_id] = copy.deepcopy(agent.critic1) if la_ss_step is not None else None
                    la_sss_critics1[agent_id] = copy.deepcopy(agent.critic1) if la_sss_step is not None else None

                    la_critics2[agent_id] = copy.deepcopy(agent.critic2)
                    la_ss_critics2[agent_id] = copy.deepcopy(agent.critic2) if la_ss_step is not None else None
                    la_sss_critics2[agent_id] = copy.deepcopy(agent.critic2) if la_sss_step is not None else None

           
        for episode in range(args.episode_num):
            obs = env.reset()
            agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
            while env.agents:  # interact with the env for an episode
                step += 1
                obs = obs[0] if isinstance(obs, tuple) else obs
                if step < args.random_steps:
                    action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
                else:
                    action = model.select_action(obs)

                next_obs, reward, term, trunc = env.step(action)
                # env.render()
                done = term or trunc
                model.add(obs, action, reward, next_obs, done)

                for agent_id, r in reward.items():  # update reward
                    agent_reward[agent_id] += r

                if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                    if model.buffer_size > args.batch_size:
                        model.learn(args.batch_size, args.gamma)
                        model.update_target(args.tau)
                        # for agent_id, agent in model.agents.items():
                        #     agent.actor_scheduler.step()
                        #     agent.critic_scheduler.step()

                obs = next_obs

            # episode finishes
            for agent_id, r in agent_reward.items():  # record reward
                episode_rewards[agent_id][episode] = r

            if (episode + 1) % 100 == 0:  # print info every 100 episodes
                message = f'episode {episode + 1}, '
                sum_reward = 0
                for agent_id, r in agent_reward.items():  # record reward
                    message += f'{agent_id}: {r:>4f}; '
                    sum_reward += r
                message += f'sum reward: {sum_reward}'
                print(message)
                # print('actor_lr:', agent.actor_scheduler.get_last_lr()) 
                # print('critic_lr:', agent.critic_scheduler.get_last_lr()) 

            # Perform  lookahead step according to LA_STEP
            if args.lookahead and ((episode + 1) % la_step == 0):
              for agent_id, agent in model.agents.items():
                #actor
                lookahead(la_actors[agent_id], agent.actor)
                #la_actor = copy.deepcopy(agent.actor)
                la_actors[agent_id] = copy.deepcopy(agent.actor)
            
                #critic
                if args.algo == 'MADDPG':
                    lookahead(la_critics[agent_id], agent.critic)
                    la_critics[agent_id] = copy.deepcopy(agent.critic)

                if args.algo == 'MATD3':
                    lookahead(la_critics1[agent_id], agent.critic1)
                    la_critics1[agent_id] = copy.deepcopy(agent.critic1)

                    lookahead(la_critics2[agent_id], agent.critic2)
                    la_critics2[agent_id] = copy.deepcopy(agent.critic2)

            # # #update target networks    
            # #   model.update_target(args.tau)
                          
            # Perform nested lookahead step according to LA_SS_STEP
            if args.lookahead and la_ss_step != None:
              if ((episode + 1) % la_ss_step == 0):
                for agent_id, agent in model.agents.items():
                    #actor
                    lookahead(la_ss_actors[agent_id], agent.actor)
                    # update both la and nested la copies
                    la_actors[agent_id] = copy.deepcopy(agent.actor)
                    la_ss_actors[agent_id] = copy.deepcopy(agent.actor)

                    
                    #critic
                    if args.algo == 'MADDPG':
                        lookahead(la_ss_critics[agent_id], agent.critic)
                        la_critics[agent_id] = copy.deepcopy(agent.critic)
                        la_ss_critics[agent_id] = copy.deepcopy(agent.critic)
                    if args.algo == 'MATD3':
                        lookahead(la_ss_critics1[agent_id], agent.critic1)
                        #la_critic = copy.deepcopy(agent.critic)
                        la_critics1[agent_id] = copy.deepcopy(agent.critic1)
                        la_ss_critics1[agent_id] = copy.deepcopy(agent.critic1)

                        lookahead(la_ss_critics2[agent_id], agent.critic2)
                        #la_critic = copy.deepcopy(agent.critic)
                        la_critics2[agent_id] = copy.deepcopy(agent.critic2)
                        la_ss_critics2[agent_id] = copy.deepcopy(agent.critic2)


             # Perform 2nd nested lookahead step according to LA_SS_STEP
            if args.lookahead and la_sss_step != None:
                if ((episode + 1) % la_sss_step == 0):
                    for agent_id, agent in model.agents.items():
                        #actor
                        lookahead(la_sss_actors[agent_id], agent.actor)
                        la_actors[agent_id] = copy.deepcopy(agent.actor)
                        la_ss_actors[agent_id] = copy.deepcopy(agent.actor)
                        la_sss_actors[agent_id] = copy.deepcopy(agent.actor)

                            #critic
                        if args.algo == 'MADDPG':
                            lookahead(la_sss_critics[agent_id], agent.critic)
                            la_critics[agent_id] = copy.deepcopy(agent.critic)
                            la_ss_critics[agent_id] = copy.deepcopy(agent.critic)
                            la_sss_critics[agent_id] =copy.deepcopy(agent.critic)
                        if args.algo == 'MATD3':
                            lookahead(la_sss_critics1[agent_id], agent.critic1)
                            #la_critic = copy.deepcopy(agent.critic)
                            la_critics1[agent_id] = copy.deepcopy(agent.critic1)
                            la_ss_critics1[agent_id] = copy.deepcopy(agent.critic1)
                            la_sss_critics1[agent_id] =copy.deepcopy(agent.critic1)

                            lookahead(la_sss_critics2[agent_id], agent.critic2)
                            #la_critic = copy.deepcopy(agent.critic)
                            la_critics2[agent_id] = copy.deepcopy(agent.critic2)
                            la_ss_critics2[agent_id] = copy.deepcopy(agent.critic2)
                            la_sss_critics2[agent_id] =copy.deepcopy(agent.critic2)

                  
            if (episode==0 or (episode + 1) % 1000 == 0):
                if args.env_name=='simple_tag_v2':
                    test_scores_dict[episode], test_distances_dict[episode] = agent_test(model, args.env_name, episode, results_dir =result_dir, steps = 25, save_gifs=False)
                else:
                    test_scores_dict[episode] = agent_test(model, args.env_name, episode, results_dir =result_dir, steps = 25, save_gifs=False)   

            
        with open(os.path.join(result_dir, 'rewards_seed'+f'{seed}'+'.pkl'), 'wb') as f:  # save testing data
                pickle.dump(test_scores_dict, f)
        with open(os.path.join(result_dir, 'distances_seed'+f'{seed}'+'.pkl'), 'wb') as f:  # save testing data
                pickle.dump(test_distances_dict, f)
        model.save(episode_rewards)  # save model
