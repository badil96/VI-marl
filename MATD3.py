import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from Agent import AgentMATD3
from Buffer import Buffer


def setup_logger(filename):
    """ set up logger with filename. """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


class MATD3:
    """A MATD3 agent"""

    def __init__(self, optimizer, dim_info, capacity, batch_size, actor_lr, critic_lr, policy_freq,use_target_policy_smoothing,  res_dir="./results"):
        # sum all the dims of each agent to get input dim for critic
        print('-----------MATD3----------------')
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())
        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.buffers = {}
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = AgentMATD3(optimizer, obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr)
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, 'cpu')
        self.dim_info = dim_info
        self.buffer_size = self.buffers[agent_id].__len__()

        self.batch_size = batch_size
        self.res_dir = res_dir  # directory to save the training result
        self.logger = setup_logger(os.path.join(res_dir, 'matd3.log'))
        self.actor_norms = {agent_id: [] for agent_id in self.agents.keys()}
        self.critic_norms = {agent_id: [] for agent_id in self.agents.keys()}
        self.policy_freq = policy_freq
        self.train_iter = 0

        self.use_target_policy_smoothing = use_target_policy_smoothing
        self.action_noise_std = 0.2
        self.action_noise_clip = 0.5

    def add(self, obs, action, reward, next_obs, done):
        # NOTE that the experience is a dict with agent name as its key
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            if isinstance(a, int):
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                a = np.eye(self.dim_info[agent_id][1])[a]

            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = done[agent_id]
            self.buffers[agent_id].add(o, a, r, next_o, d)
        self.buffer_size = self.buffers[agent_id].__len__()

    def clear(self):
        for agent_id, _ in self.dim_info.items():
            self.buffers[agent_id].clear()
            self.buffer_size = self.buffers[agent_id].__len__()


    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        try:
            total_num = len(self.buffers['agent_0'])
        except:
            total_num = len(self.buffers['player_0'])
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, d = buffer.sample(indices)
            obs[agent_id] = o
            act[agent_id] = a
            reward[agent_id] = r
            next_obs[agent_id] = n_o
            done[agent_id] = d
            # calculate next_action using target_network and next_state
            next_act[agent_id] = self.agents[agent_id].target_action(n_o, self.use_target_policy_smoothing,self.action_noise_std, self.action_noise_clip)

        return obs, act, reward, next_obs, done, next_act

    def select_action(self, obs, noise_scale=0.1):
        actions = {}
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float()
            a = self.agents[agent].action(o)  # torch.Size([1, action_size])
            # NOTE that the output is a tensor, convert it to int before input to the environment
            #action += noise_scale * np.random.randn(*action.shape)
            # action = np.clip(action, -1, 1)
            actions[agent] = a.squeeze(0).argmax().item()
            #self.logger.info(f'{agent} action: {actions[agent]}')
        return actions

    def learn(self, batch_size, gamma):
        self.train_iter += 1
        for agent_id, agent in self.agents.items():
            obs, act, reward, next_obs, done, next_act = self.sample(batch_size)
            # update critic
            critic1_value = agent.critic1_value(list(obs.values()), list(act.values()))
            critic2_value = agent.critic2_value(list(obs.values()), list(act.values()))

            # calculate target critic value
            next_target_critic1_value = agent.target_critic1_value(list(next_obs.values()),
                                                                 list(next_act.values()))
            next_target_critic2_value = agent.target_critic2_value(list(next_obs.values()),
                                                                 list(next_act.values()))
            next_target_critic_value = torch.min(next_target_critic1_value, next_target_critic2_value)                                         
            target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - done[agent_id])

            critic1_loss = F.mse_loss(critic1_value, target_value.detach(), reduction='mean') 
            critic2_loss = F.mse_loss(critic2_value, target_value.detach(), reduction='mean')
            critic1_norm = agent.update_critic1(critic1_loss)
            critic2_norm = agent.update_critic2(critic2_loss)
            self.critic_norms[agent_id].append(critic1_norm)
            # update actor every policy_freq iters
            if self.train_iter % self.policy_freq == 0:
            # action of the current agent is calculated using its actor
                action, logits = agent.action(obs[agent_id], model_out=True)
                act[agent_id] = action
                actor_loss = -agent.critic1_value(list(obs.values()), list(act.values())).mean()
                actor_loss_pse = torch.pow(logits, 2).mean()
                actor_norm = agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
                self.actor_norms[agent_id].append(actor_norm)
                self.logger.info(f'agent{agent_id}: critic loss: {critic1_loss.item()}, actor loss: {actor_loss.item()}')

    def update_target(self, tau):
        #in the paper they update the targets only at the time as policy updates
        if self.train_iter % self.policy_freq != 0:
            return
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic1, agent.target_critic1)
            soft_update(agent.critic2, agent.target_critic2)

    def save(self, reward):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},  # actor parameter
            os.path.join(self.res_dir, 'model.pt')
        )
        with open(os.path.join(self.res_dir, 'rewards.pkl'), 'wb') as f:  # save training data
            pickle.dump({'rewards': reward}, f)

        with open(os.path.join(self.res_dir, 'norms.pkl'), 'wb') as f:  # save training data
            pickle.dump({'actor_norms': self.actor_norms, 'critic_norms': self.critic_norms}, f)



    @classmethod
    def load(cls, optimizer, dim_info, file):
        """init matd3 using the model saved in `file`"""
        instance = cls(dim_info, optimizer, 0, 0, 0, 0, 0, False, os.path.dirname(file))
        data = torch.load(file)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance
