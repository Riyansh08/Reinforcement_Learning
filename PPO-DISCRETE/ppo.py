from re import S
from torch.distribution import Categorical 
from layers import Actor , Critic
import numpy as np

import torch
import os
import math 
import copy
import random 
import time

class PPO_Discrete():
    
    def __init__(self , **kwargs):
        self.__dict__.update(kwargs)
        
        # Initialize the actor and critic networks
        self.actor = Actor(self.state_dim , self.hidden_dim , self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim , self.hidden_dim).to(self.device)
        
        # Initialize the optimizer
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters() , lr = self.lr) 
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters() , lr = self.lr)
        
        #Building trajectory holder 
        self.s_holder = np.zeros((self.T_horizon , self.state_dim) , dtype = np.float32)
        self.a_holder = np.zeros((self.T_horizon , 1) , dtype = np.int64)
        self.r_holder = np.zeros((self.T_horizon , 1) , dtype = np.float32)
        self.s_next_holder = np.zeros((self.T_horizon , self.state_dim) , dtype = np.float32)
        self.logp_a_holder = np.zeros((self.T_horizon , 1), dtype = np.float32)
        self.done_holder = np.zeros((self.T_horizon , 1) , dtype = np.bool_)
        self.dw_holder = np.zeros((self.T_horizon , 1) , dtype = np.bool_)
        
    
    #Selects action with help of Actor Network
    def select_action(self , state , deterministic):
        state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            probs = self.actor.get_action(state)
            if deterministic:
                a = torch.argmax(probs , dim = -1)
                return a , None #used during inference 
            else:
                m = Categorical(probs)
                a = m.sample().item()
                pi_a = m.log_prob(a).item()
                return a, pi_a #for training 
            
    def train(self):
        self.entropy_coef *= self.entropy_coef_decay 
        s = torch.from_numpy(self.s_holder).to(self.device)
        a = torch.from_numpy(self.a_holder).to(self.device)
        r = torch.from_numpy(self.r_holder).to(self.device)
        s_next = torch.from_numpy(self.s_next_holder).to(self.device)
        old_prob = torch.from_numpy(self.logp_a_holder).to(self.device)
        done = torch.from_numpy(self.done_holder).to(self.device)
        dw = torch.from_numpy(self.dw_holder).to(self.device)
        
        # Calculate TD + GAE 
        
        with torch.no_grad():
            vs = self.critic(s)
            vs_next = self.critic(s_next)
            
            deltas = r + self.gamma * vs_next * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()
            
            adv = [0]
            
            for dlt , done  in zip(deltas[::-1] , done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~done)
                adv.append(advantage)
            
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv , dtype = torch.float32).to(self.device)
            vs = torch.tensor(vs , dtype = torch.float32).to(self.device)
            td_target = vs + adv
            
            if self.adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-10)
            
            # PPO update 
            
            optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))
            for _ in range(self.K_epochs):
                
                perm = np.arange(s.shape[0])
                np.random.shuffle(perm)
                perm = torch.LongTensor(perm).to(self.device)
                s , a , td_target , adv , old_prob = s[perm].clone() , a[perm].clone() , td_target[perm].clone() , adv[perm].clone() , old_prob[perm].clone()
                
                for i in range(optim_iter_num):
                    index = slice(i * self.batch_size , min((i+1) * self.batch_size , s.shape[0]))
                    prob = self.actor.get_action(s[index] , softmax_dim = -1)
                    entropy = Categorical(prob).entropy().sum(0 , keepdim = True)
                    prob_a = prob.gather(1 , a[index]
                                         )
                    ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob[index]))
                    
                    surr1 = ratio * adv[index]
                    surr2 = torch.clamp(ratio , 1 - self.clip_ratio , 1 + self.clip_ratio) * adv[index]
                    a_loss = -torch.min(surr1 , surr2) - self.entropy_coef * entropy 
                    self.actor_optimizer.zero_grad()
                    a_loss.mean().backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters() , self.max_grad_norm)
                    self.actor_optimizer.step()
                    
                    
                    c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                    for name,param in self.critic.named_parameters():
                        if 'weight' in name:
                            c_loss += self.l2_reg * param.pow(2).sum()
                        
                        self.critic.optimizer.zero_grad()
                        c_loss.backward()
                        self.critic.optimizer.step()
    
    
    def put_data(self, s ,a ,r ,s_next ,done , dw , idx , logp):
        self.s_holder[idx] = s 
        self.a_holder[idx] = a 
        self.r_holder[idx] = r 
        self.s_next_holder[idx] = s_next 
        self.logp_a_holder[idx] = logp
        self.done_holder[idx] = done 
        self.dw_holder[idx] = dw
        
    def save(self , episode):
        torch.save(self.actor.state_dict() , "./model/ppo_actor_{}".format(episode))
        torch.save(self.critic.state_dict() , "./model/ppo_critic_{}".format(episode))
        
    
    def load(self, episode):
        self.actor.load_state_dict(torch.load("./model/ppo_actor_{}".format(episode)))
        self.critic.load_state_dict(torch.load("./model/ppo_critic_{}".format(episode)))