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
        
        
        
    
        
        





