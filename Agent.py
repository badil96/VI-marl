from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR

import numpy as np 

from extragradient import ExtraAdam
from ogd import OptimisticGD

class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, optimizer, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr):
        self.actor = MLPNetwork(obs_dim, act_dim)

        # critic input all the observations and actions
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
        self.critic = MLPNetwork(global_obs_dim, 1)
        self.optimizer = optimizer
        if optimizer == 'Adam':
            self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        elif optimizer == 'SGD':
            self.actor_optimizer = SGD(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = SGD(self.critic.parameters(), lr=critic_lr)
        elif optimizer == 'ExtraAdam':
            self.actor_optimizer = ExtraAdam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = ExtraAdam(self.critic.parameters(), lr=critic_lr)
        elif optimizer == 'OptimisticGD':
            self.actor_optimizer = OptimisticGD(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = OptimisticGD(self.critic.parameters(), lr=critic_lr)

        print(f'Initialized agent with {self.actor_optimizer} optimizer.')
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        # self.actor_scheduler = StepLR(self.actor_optimizer, step_size=10000, gamma=0.1)
        # self.critic_scheduler = StepLR(self.critic_optimizer, step_size=10000, gamma=0.1)

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
        # but as mention in the doc, it may be removed in the future, so i implement it myself
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def action(self, obs, model_out=False):
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])

        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        # p = torch.nn.functional.softmax(logits)
        # action = self.gumbel_softmax(logits)
        action = F.gumbel_softmax(logits, hard=True)
        if model_out:
            return action, logits
        return action

    def target_action(self, obs):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])

        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        # action = self.gumbel_softmax(logits)
        action = F.gumbel_softmax(logits, hard=True)
        return action.squeeze(0).detach()

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        if self.optimizer == 'ExtraAdam':
            self.actor_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.extrapolation()
            self.actor_optimizer.zero_grad()
            loss.backward()
            grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in self.actor.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
        elif self.optimizer == 'Adam' or self.optimizer == 'OptimisticGD' or self.optimizer == 'SGD':
            self.actor_optimizer.zero_grad()
            loss.backward()
            grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in self.actor.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
        return grad_norm

    def update_critic(self, loss):
        if self.optimizer == 'ExtraAdam':
            self.critic_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.extrapolation()
            self.critic_optimizer.zero_grad()
            loss.backward()
            grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in self.critic.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        elif self.optimizer == 'Adam' or self.optimizer == 'OptimisticGD' or self.optimizer == 'SGD':
            self.critic_optimizer.zero_grad()
            loss.backward()
            grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in self.critic.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        return grad_norm

class AgentMATD3:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, optimizer, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr):
        self.actor = MLPNetwork(obs_dim, act_dim)

        # critic input all the observations and actions
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
        self.critic1 = MLPNetwork(global_obs_dim, 1)
        self.critic2 = MLPNetwork(global_obs_dim, 1)
        self.optimizer = optimizer
        if optimizer == 'Adam':
            self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
            self.critic1_optimizer = Adam(self.critic1.parameters(), lr=critic_lr)
            self.critic2_optimizer = Adam(self.critic2.parameters(), lr=critic_lr)
        elif optimizer == 'SGD':
            self.actor_optimizer = SGD(self.actor.parameters(), lr=actor_lr)
            self.critic1_optimizer = SGD(self.critic1.parameters(), lr=critic_lr)
            self.critic2_optimizer = SGD(self.critic2.parameters(), lr=critic_lr)
        elif optimizer == 'ExtraAdam':
            self.actor_optimizer = ExtraAdam(self.actor.parameters(), lr=actor_lr)
            self.critic1_optimizer = ExtraAdam(self.critic1.parameters(), lr=critic_lr)
            self.critic2_optimizer = ExtraAdam(self.critic2.parameters(), lr=critic_lr)
        elif optimizer == 'OptimisticGD':
            self.actor_optimizer = OptimisticGD(self.actor.parameters(), lr=actor_lr)
            self.critic1_optimizer = OptimisticGD(self.critic1.parameters(), lr=critic_lr)
            self.critic2_optimizer = OptimisticGD(self.critic2.parameters(), lr=critic_lr)
        print(f'Initialized agent with {self.actor_optimizer} optimizer.')
        self.target_actor = deepcopy(self.actor)
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)
        # self.actor_scheduler = StepLR(self.actor_optimizer, step_size=10000, gamma=0.1)
        # self.critic_scheduler = StepLR(self.critic_optimizer, step_size=10000, gamma=0.1)

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
        # but as mention in the doc, it may be removed in the future, so i implement it myself
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def action(self, obs, model_out=False):
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])

        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        # p = torch.nn.functional.softmax(logits)
        # action = self.gumbel_softmax(logits)
        action = F.gumbel_softmax(logits, hard=True)
        if model_out:
            return action, logits
        return action

    def target_action(self, obs, noise=False, action_noise_std=0.2, action_noise_clip=0.5):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])

        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        # action = self.gumbel_softmax(logits)
        if noise: #add target policy smoothing
          noise = torch.clip(torch.normal(0.0, action_noise_std,logits.shape), -action_noise_clip, action_noise_clip)
          logits = logits + noise
        action = F.gumbel_softmax(logits, hard=True)
        return action.squeeze(0).detach()

    def critic1_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic1(x).squeeze(1)  # tensor with a given length
        
    def critic2_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic2(x).squeeze(1)  # tensor with a given length

    def target_critic1_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic1(x).squeeze(1)  # tensor with a given length
        
    def target_critic2_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic2(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        if self.optimizer == 'ExtraAdam':
            self.actor_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.extrapolation()
            self.actor_optimizer.zero_grad()
            loss.backward()
            grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in self.actor.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
        elif self.optimizer == 'Adam' or self.optimizer == 'OptimisticGD' or self.optimizer == 'SGD':
            self.actor_optimizer.zero_grad()
            loss.backward()
            grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in self.actor.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
        return grad_norm

    def update_critic1(self, loss):
        if self.optimizer == 'ExtraAdam':
            self.critic1_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic1_optimizer.extrapolation()
            self.critic1_optimizer.zero_grad()
            loss.backward()
            grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in self.critic1.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 0.5)
            self.critic1_optimizer.step()
        elif self.optimizer == 'Adam' or self.optimizer == 'OptimisticGD' or self.optimizer == 'SGD':
            self.critic1_optimizer.zero_grad()
            loss.backward()
            grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in self.critic1.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 0.5)
            self.critic1_optimizer.step()
        return grad_norm
        
    def update_critic2(self, loss):
        if self.optimizer == 'ExtraAdam':
            self.critic2_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 0.5)
            self.critic2_optimizer.extrapolation()
            self.critic2_optimizer.zero_grad()
            loss.backward()
            grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in self.critic2.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 0.5)
            self.critic2_optimizer.step()
        elif self.optimizer == 'Adam' or self.optimizer == 'OptimisticGD' or self.optimizer == 'SGD':
            self.critic2_optimizer.zero_grad()
            loss.backward()
            grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in self.critic2.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 0.5)
            self.critic2_optimizer.step()
        return grad_norm



class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)