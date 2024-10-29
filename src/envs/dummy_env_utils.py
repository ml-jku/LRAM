import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import DummyVecEnv


class DummyEnv(gym.Env):
    def __init__(self, obs_dim=10, act_dim=1, ep_len=1000):
        super(DummyEnv, self).__init__()        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        self.ep_len = ep_len
        self.current_step = 0

    def reset(self):
        """Reset the environment and return the initial observation."""
        self.current_step = 0
        # Return an initial dummy observation
        return self.observation_space.sample()

    def step(self, action):
        """Take a step in the environment."""
        self.current_step += 1        
        observation = self.observation_space.sample()
        done = self.current_step >= self.ep_len
        return observation, 1, done, {} 

    def render(self, mode="human"):
        """Render the environment (optional, can be extended)."""
        pass

    def close(self):
        """Clean up resources (optional)."""
        pass


def get_dummyenv_constructor(envid, env_kwargs=None):
    env_kwargs = dict(env_kwargs) if env_kwargs is not None else {}
    def make():
        env = DummyEnv(**env_kwargs)
        env.name = envid
        return Monitor(env)
    return make


def get_dummyenv_constructors(envid, env_kwargs=None):
    if not isinstance(envid, list):
        envid = [envid] 
    return [get_dummyenv_constructor(eid, env_kwargs=env_kwargs) for eid in envid]


def make_dummyenv_envs(env_params, envid, make_eval_env=True):
    const_kwargs = {
        "envid": envid,
        "env_kwargs": env_params.get("env_kwargs", {}),
    }
    env = DummyVecEnv(get_dummyenv_constructors(**const_kwargs))
    eval_env = None
    if make_eval_env:
        eval_env = DummyVecEnv(get_dummyenv_constructors(**const_kwargs))
        eval_env.num_envs = 1
    env.num_envs = 1
    return env, eval_env
