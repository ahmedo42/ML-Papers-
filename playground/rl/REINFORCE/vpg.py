import gym
import numpy as np
import torch
from torch.optim import Adam
from playground.rl.buffers import VPGBuffer
from playground.rl.policies import MLPActorCritic


class VPG:
    def __init__(
        self,
        env: gym.Env,
        ac_policy: str = "mlp",
        seed: int = 17,
        ac_kwargs: dict = dict(),
    ):
        self.env = env
        self.buffer = VPGBuffer(env.observation_space.shape, env.action_space.shape, 4000)
        self.model = None
        if ac_policy == "mlp":
            self.model = MLPActorCritic(
                env.observation_space, env.action_space, **ac_kwargs
            )
        self.policy_optimizer = Adam)

        
    def train(
        self,
        epochs: int = 50,
        steps_per_epoch: int = 4000,
        max_episode_len: int = 1000,
        gamma: float = 0.99,
        policy_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        adv_lambda: int = 0.97,
    ):
        observation = self.env.reset()
        episode_length = 0
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                action , value , logp = self.model.act(observation)
                next_observation , reward , done , info = self.env.step(action)
                episode_length += 1

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
                    observation, ep_ret, episode_length = self.env.reset(), 0, 0


            self._update()

    def _update(self):
        data = self.buffer.get()

        # Get loss and info values before update
        pi_l_old, pi_info_old = self._compute_policy_loss(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self._compute_value_loss(data).item()

        # Train policy with a single step of gradient descent
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        mpi_avg_grads(ac.pi)    # average grads across MPI processes
        pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))
    def _log(self):
        pass

    def _compute_policy_loss(self):
        pass

    def _compute_value_loss(self):
        pass
