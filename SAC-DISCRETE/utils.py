import torch 
import torch.nn as nn
import numpy as np
import argparse
import torch.nn.functional as F

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