import torch 
import torch.nn as nn
import numpy as np
import argparse
import torch.nn.functional as F

def build_net(layer_shape , hid_activation , output_activation):
    layers = []
    
    for i in range(len(layer_shape)-1):
        act = hid_activation if i < len(layer_shape)-2 else output_activation 
        layers.append(nn.Linear(layer_shape[i] , layer_shape[i+1]))
        layers.append(act)
    
    return nn.Sequential(*layers) 

# Double Q Network - Critic 
class Double_Q_Net(nn.Module):
    def __init__(self , state_dim , hid_shape , action_dim):
        super(Double_Q_Net , self).__init__()
        
        layers = [state_dim + list(hid_shape) + action_dim]
        self.Q1 = build_net(layers , nn.ReLU , nn.Identity)
        self.Q2 = build_net(layers , nn.ReLU , nn.Identity)
    
    def forward(self , state):
        q1 = self.Q1(state)
        q2 = self.Q2(state)
        return q1 , q2

# Actor Network 

class Policy_Net(nn.Module):
    def __init__(self , state_dim , hid_shape , action_dim):
        super(Policy_Net, self).__init__()
        layers = [state_dim + list(hid_shape) + action_dim]
        self.P = build_net(layers , nn.ReLU, nn.Identity)
    
    def forward(self , state):
        logits = self.P(state)
        probs = F.softmax(logits , dim = -1)
        return probs 
    
class ReplayBuffer(object):
    def __init__(self, state_dim , action_dim , dvc , max_size = int(1e6)):
        self.max_size = max_size
        self.ptr = 0 
        self.size = 0
        self.dvc = dvc
        
        self.s = torch.zeros((max_size , state_dim ) , dtype = torch.float , device = dvc)
        self.a = torch.zeros((max_size, 1),dtype=torch.long,device=dvc)
        self.r = torch.zeros((max_size, 1),dtype=torch.float,device=dvc)
        self.s_next = torch.zeros((max_size, state_dim),dtype=torch.float,device=dvc)
        self.dw = torch.zeros((max_size, 1),dtype=torch.bool,device=dvc)
        
    def add(self , s , a , r , s_next , dw):
        self.s[self.ptr] = s 
        self.a[self.ptr] = a 
        self.r[self.ptr] = r 
        self.s_next[self.ptr] = s_next 
        self.dw[self.ptr] = dw 
        self.ptr = (self.ptr + 1) % self.max_size 
        self.size = min(self.size + 1 , self.max_size)
    
    def sample(self , batch_size):
        idx = np. random.randint(0 , self.size , size = (batch_size,) , device = self.dvc )
        return self.s[idx] , self.a[idx] , self.r[idx] , self.s_next[idx] , self.dw[idx]
        
    def evaluate_policy(env , agent , turns = 3 ):
        total_scores = 0 
        
        for j in range(turns):
            s , info = env.reset()
            done = False 
            while not done:
                a = agent.select_action(s ,  deterministic = True)
                s_next , r , done , tr , info = env.step(a)
                done = (done or tr)
                total_scores +=r 
                s = s_next 
        return int(total_scores/ turns ) 