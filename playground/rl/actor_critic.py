import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from playground.rl.utils import make_mlp


# Policy
class Actor(nn.Module):
    def _distribution(self,obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self,pi,act):
        raise NotImplementedError

    def forward(self,obs,act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.net = make_mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, action):
        return pi.log_prob(action)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.net = make_mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, action):
        return pi.log_prob(action).sum(axis=-1)  


class MLPCritic(nn.Module):
    def __init__(self , obs_dim , hidden_sizes , activation):
        super().__init__()
        self.v = make_mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self,obs):
        return torch.squeeze(self.v(obs), -1)


