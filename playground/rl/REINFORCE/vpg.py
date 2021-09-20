import gym
import numpy as np
import torch
import time
from torch.optim import Adam
from playground.rl.buffers import VPGBuffer
from playground.rl.policies import MLPActorCritic
from playground.rl.logger import EpochLogger


class VPG:
    def __init__(
        self,
        env: gym.Env,
        ac_policy: str = "mlp",
        ac_kwargs: dict = dict()
         ):
        self.env = env
        self.model = None
        if ac_policy == "mlp":
            self.model = MLPActorCritic(
                env.observation_space, env.action_space, **ac_kwargs
            )
            
        


        
    def train(
        self,
        epochs: int = 50,
        steps_per_epoch: int = 4000,
        max_episode_len: int = 1000,
        gamma: float = 0.99,
        adv_lambda: int = 0.95,
        policy_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        train_value_steps: int = 80,
        seed: int = 17,
        save_freq: int = 10,
        logger_kwargs : dict = dict()
    ):

        torch.manual_seed(seed)
        np.random.seed(seed)
        logger_kwargs['output_dir'] = f"./experiments/{self.env.unwrapped.__class__.__name__}/{self.__class__.__name__}/{seed}"

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.setup_pytorch_saver(self.model)
        self.buffer = VPGBuffer(self.env.observation_space.shape, self.env.action_space.shape, steps_per_epoch,gamma,adv_lambda)
        self.policy_optimizer = Adam(self.model.pi.parameters(),lr=policy_lr)
        self.value_optimizer = Adam(self.model.v.parameters(),lr=vf_lr)
        self.value_steps = train_value_steps


        observation = self.env.reset()
        start_time = time.time()
        episode_length = 0
        episode_ret = 0 # reward-to-go
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                observation = torch.as_tensor(observation,dtype=torch.float32)
                action , value , logp = self.model.step(observation)
                next_observation , reward , done , info = self.env.step(action)
                episode_length += 1
                episode_ret += reward

                self.buffer.store(observation,action,reward,value,logp)

                observation = next_observation

                timeout = episode_length == max_episode_len
                terminal = done or timeout
                epoch_ended = step == steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%episode_length, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, value, _ = self.model.step(torch.as_tensor(observation, dtype=torch.float32))
                    else:
                        v = 0
                    self.buffer.finish_path(v)
                    if terminal:
                        self.logger.store(EpRet=episode_ret, EpLen=episode_length)
                    observation, episode_ret, episode_length = self.env.reset(), 0, 0


            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                self.logger.save_state({'env': self.env}, None)
            
            # update parameters
            self._update()

            #log statistics
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossPi', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
            self.logger.log_tabular('Entropy', average_only=True)
            self.logger.log_tabular('KL', average_only=True)
            self.logger.log_tabular('Time', time.time()-start_time)
            self.logger.dump_tabular()

    def _update(self):
        data = self.buffer.get()

        # Get loss and info values before update
        policy_loss_old, pi_info_old = self._compute_policy_loss(data)
        policy_loss_old = policy_loss_old.item()
        value_loss_old = self._compute_value_loss(data).item()

        # Train policy with a single step of gradient descent
        self.policy_optimizer.zero_grad()
        policy_loss, pi_info = self._compute_policy_loss(data)
        policy_loss.backward()
        self.policy_optimizer.step()

        # Value function learning
        for i in range(self.value_steps):
            self.value_optimizer.zero_grad()
            value_loss = self._compute_value_loss(data)
            value_loss.backward()
            self.value_optimizer.step()

        kl_divergence, entropy = pi_info['kl'], pi_info_old['ent']
        self.logger.store(LossPi=policy_loss_old, LossV=value_loss_old,
                     KL=kl_divergence, Entropy=entropy,
                     DeltaLossPi=(policy_loss.item() - policy_loss_old),
                     DeltaLossV=(value_loss.item() - value_loss_old))


    def _compute_policy_loss(self,data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.model.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    def _compute_value_loss(self,data):
        obs, ret = data['obs'], data['ret']
        return ((self.model.v(obs) - ret)**2).mean()