from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F 

class Actor(nn.Module):
    def __init__(self , state_dim , hidden_dim , action_dim ):
        super(Actor , self).__init__()
        self.fc1 = nn.Linear(state_dim  , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim , hidden_dim)
        self.fc3 = nn.Linear(hidden_dim , action_dim)
        
    def forward(self , state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return x
    
    def get_action(self , state):
        x = self.forward(state)
        prob = F.softmax(self.fc3(x), dim = -1)
        return prob 
     
#Evaluates the state value function - subracted from the return to calculate the advantage function   
class Critic(nn.Module):
    def __init__(self , state_dim , hidden_dim):
        super(Critic , self).__init__()
        self.fc1 = nn.Linear(state_dim , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim , hidden_dim)
        self.fc3 = nn.Linear(hidden_dim ,1 )# output is reward 
    
    def forward(self , state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x 