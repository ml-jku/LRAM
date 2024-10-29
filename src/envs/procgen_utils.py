import time
import numpy as np
from copy import deepcopy
from procgen import ProcgenEnv
from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor, VecTransposeImage, VecNormalize
from stable_baselines3.common.env_util import DummyVecEnv


def get_procgen_constructor(envid, distribution_mode="easy", time_limit=None, env_kwargs=None):
    env_kwargs = dict(env_kwargs) if env_kwargs is not None else {}
    num_envs = env_kwargs.pop("num_envs", 1)
    norm_reward = env_kwargs.pop("norm_reward", False)
    def make():
        env = ProcgenEnv(env_name=envid, num_envs=num_envs,
                         distribution_mode=distribution_mode, **env_kwargs)
        # monitor to obtain ep_rew_mean, ep_rew_len + extract rgb images from dict states
        env = CustomVecMonitor(VecExtractDictObs(env, 'rgb'), time_limit=time_limit)
        env = VecTransposeImage(env)
        if norm_reward: 
            env = VecNormalize(env, norm_obs=False, norm_reward=True)
        env.name = envid
        return env
    return make


class CustomVecMonitor(VecMonitor):
    """
    Custom version of VecMonitor that allows for a timelimit.
    Once, timelimit is hit, we also need to reset the environment.
    We can however, not save the reset state there. 
    """
    def __init__(
        self,
        venv,
        filename=None,
        info_keywords=(),
        time_limit=None
    ):
        super().__init__(venv, filename, info_keywords)
        self.time_limit = time_limit

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])
        if self.time_limit is not None and (self.episode_lengths >= self.time_limit).any(): 
            # check if any is over timelimit, if yes, set done
            over_time = self.episode_lengths >= self.time_limit
            # send action -1 to reset ProcgenEnv: https://github.com/openai/procgen/issues/40#issuecomment-633720234
            reset_action = over_time * -1 
            reset_obs, reset_rewards, reset_done, reset_info = self.venv.step(reset_action)
            # get reset observation, ignore rest
            obs[over_time] = reset_obs[over_time]
            # set done where done or over_time
            dones = dones | over_time
            
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {"r": episode_return, "l": episode_length, "t": round(time.time() - self.t_start, 6)}
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
        return obs, rewards, dones, new_infos


class CustomDummyVecEnv(DummyVecEnv):
    """
    Custom version of DummyVecEnv that allows wrapping ProcgenEnvs. 
    By default, ProcgenEnvs are vectorized already. 
    Therefore wrapping different tasks in a single DummyVecEnv fails, due to returning of vectorized infor buffers.
    """
    def step_wait(self):
        for env_idx in range(self.num_envs):
            action = self.actions[env_idx]
            if not isinstance(action, np.ndarray): 
                action = np.array([action])
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                action
            )
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                # self.buf_infos[env_idx]terminal_observation"] = obs
                self.buf_infos[env_idx][0]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))
