
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_net(layer_shape, hid_activation, output_activation):
    layers = []
    for i in range(len(layer_shape) - 1):
        act = hid_activation() if i < len(layer_shape) - 2 else output_activation()
        layers.append(nn.Linear(layer_shape[i], layer_shape[i+1]))
        layers.append(act)
    return nn.Sequential(*layers)

# Double Q Network – Critic
class Double_Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state):
        q1 = self.Q1(state)
        q2 = self.Q2(state)
        return q1, q2

# Actor Network
class Policy_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Policy_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.P = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state):
        logits = self.P(state)
        probs = F.softmax(logits, dim=-1)
        return probs
