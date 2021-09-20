
import torch
import gym
import time
import numpy as np
import joblib
from playground.rl.logger import EpochLogger
from typing import Tuple , List,Union



def evaluate_agent(
    agent,
    env: str,
    n_eval_episodes: int = 10,
    render=True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:



    logger = EpochLogger()
    env = gym.make(env)

    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < n_eval_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        
        with torch.no_grad():
            obervation = torch.as_tensor(o,dtype=torch.float32)
            action = agent.act(obervation)
        o, r, d, _ = env.step(action)
        ep_ret += r
        ep_len += 1
        if d:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


        
