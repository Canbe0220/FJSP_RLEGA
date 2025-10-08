import copy
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from extraction import DecisionMaking

class Memory:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.terminals = []
        self.action_indexes = []
      
        self.curr_proc_adj = []
        self.batch_idxes = []
        self.norm_opes = []
        self.norm_macs = []
        self.mask_proc = []       


    def clear_memory(self):
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.terminals[:]
        del self.action_indexes[:]
        
        del self.curr_proc_adj[:]
        del self.batch_idxes[:]
        del self.norm_opes[:]
        del self.norm_macs[:]
        del self.mask_proc[:]


class PPO:
    def __init__(self, train_paras, model_paras, ppo_paras, num_envs=None):
        self.learning_rate = ppo_paras["learning_rate"]  # learning rate
        self.discount_factor = ppo_paras["discount_factor"]  # discount factor
        self.K_epoch = ppo_paras["K_epoch"]  # Update policy for K epochs
        self.clip_ratio = ppo_paras["clip_ratio"]  # clip ratio
        self.policy_coe = ppo_paras["policy_coe"]  # coefficient for policy loss
        self.value_coe = ppo_paras["value_coe"]  # coefficient for value loss
        self.entropy_coe = ppo_paras["entropy_coe"]  # coefficient for entropy term
        self.num_envs = num_envs  # Number of parallel instances
        self.minibatch_size = ppo_paras["minibatch_size"]  # batch size for updating
        self.device = train_paras["device"]  # PyTorch device

        self.policy = DecisionMaking(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.learning_rate)
        self.MseLoss = nn.MSELoss()

    def update(self, memory, train_paras):
        device = train_paras["device"]

        # Flatten the data in memory (in the dim of parallel instances and decision points)
        old_curr_proc_adj = torch.stack(memory.curr_proc_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_norm_opes = torch.stack(memory.norm_opes, dim=0).transpose(0, 1).flatten(0, 1)
        old_norm_macs = torch.stack(memory.norm_macs, dim=0).transpose(0, 1).flatten(0, 1)
        old_mask_proc = torch.stack(memory.mask_proc, dim=0).transpose(0, 1).flatten(0, 1)
        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0,1)
        memory_terminals = torch.stack(memory.terminals, dim=0).transpose(0,1)
        old_logprobs = torch.stack(memory.logprobs, dim=0).transpose(0,1).flatten(0,1)
        old_action_indexes = torch.stack(memory.action_indexes, dim=0).transpose(0,1).flatten(0, 1)

        # Estimate and normalize the rewards
        rewards_envs = []
        discounted_rewards = 0
        for i in range(self.num_envs):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memory_rewards[i]), reversed(memory_terminals[i])):
                if is_terminal:
                    discounted_rewards += discounted_reward
                    discounted_reward = 0
                discounted_reward = reward + (self.discount_factor * discounted_reward)
                rewards.insert(0, discounted_reward)
            discounted_rewards += discounted_reward
            rewards = torch.tensor(rewards, dtype=torch.float64).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards_envs.append(rewards)
        rewards_envs = torch.cat(rewards_envs)

        loss_epochs = 0
        full_batch_size = old_curr_proc_adj.size(0)
        num_complete_minibatches = math.floor(full_batch_size / self.minibatch_size)
        # Optimize policy for K epochs:
        for _ in range(self.K_epoch):
            for i in range(num_complete_minibatches + 1):
                if i < num_complete_minibatches:
                    start_idx = i * self.minibatch_size
                    end_idx = (i + 1) * self.minibatch_size
                else:
                    start_idx = i * self.minibatch_size
                    end_idx = full_batch_size
                logprobs, state_values, dist_entropy = \
                    self.policy.evaluate(old_curr_proc_adj[start_idx: end_idx, :, :],                        
                                         old_norm_opes[start_idx: end_idx, :, :],
                                         old_norm_macs[start_idx: end_idx, :, :],
                                         old_mask_proc[start_idx: end_idx, :, :],
                                         old_action_indexes[start_idx: end_idx])

                ratios = torch.exp(logprobs - old_logprobs[start_idx: end_idx].detach())
                advantages = rewards_envs[start_idx: end_idx] - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                loss = - self.policy_coe * torch.min(surr1, surr2)\
                       + self.value_coe * self.MseLoss(state_values, rewards_envs[start_idx: end_idx])\
                       - self.entropy_coe * dist_entropy \

                loss_epochs += loss.mean().detach()
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss_epochs.item() / self.K_epoch, \
               discounted_rewards.item() / (self.num_envs * train_paras["update_timestep"])
