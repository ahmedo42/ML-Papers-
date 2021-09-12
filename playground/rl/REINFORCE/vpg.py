import torch
import numpy as np
import gym
from playground.rl.policies import MLPActorCritic  
from playground.rl.buffers import VPGBuffer


class VPG:
    def __init__(self,env : gym.Env, ac_policy : str = "mlp",seed : int = 17, ac_kwargs : dict = dict()):
        self.buffer = VPGBuffer(env.observation_space,env.action_space,4000)
        self.model = None
        if ac_policy == 'mlp':
            self.model = MLPActorCritic(env.observation_space,env.action_space,**ac_kwargs)



    def train(self ,epochs : int = 50, steps_per_epoch : int = 4000, max_episode_len : int = 1000,
    policy_lr : float = 3e-4,vf_lr : float = 1e-3,adv_lambda :int  = 0.97 ):
        pass

    def _update(self):
        pass

    def _log(self):
        pass