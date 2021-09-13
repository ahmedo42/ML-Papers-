import scipy.signal
import torch.nn as nn
from gym.spaces import Box, Discrete

from playground.rl.actor_critic import (MLPCategoricalActor, MLPCritic,
                                        MLPGaussianActor)


class MLPActorCritic(nn.Module):
    def __init__(
        self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh
    ):

        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation
            )

        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(
                obs_dim, action_space.shape[0], hidden_sizes, activation
            )

        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, observation):
        with torch.no_grad():
            pi = self.pi._distribution(observation)
            action = pi.sample()
            logp_action = self.pi._log_prob_from_distribution(pi, action)
            v = self.v(observation)
        return action.numpy(), v.numpy(), logp_action.numpy()

    def act(self, observation):
        return self.step(observation)[0]