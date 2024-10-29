import time
import gym
import numpy as np
import torch as th 
import pickle
import gzip
import copy
import lzma
import bz2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.save_util import save_to_pkl
from .models.image_encoders import ImpalaCNN
from .models.extractors import TextureFeatureExtractor


def save_to_compressed_pkl(path, obj) -> None:
    if not isinstance(path, str): 
        path = str(path)
    if path.endswith(".gz"):
        with gzip.open(path, "wb") as f:
            # Use protocol>=4 to support saving replay buffers >= 4Gb
            # See https://docs.python.org/3/library/pickle.html
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif path.endswith(".xz"): 
        with lzma.open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif path.endswith(".pbz2"): 
        with bz2.BZ2File(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif path.endswith(".dat"): 
        import blosc2
        pickled_data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        compressed_pickle = blosc2.compress(pickled_data)
        with open(path, "wb") as f:
            f.write(compressed_pickle)
            
            
class CustomReplayBuffer(ReplayBuffer): 
    
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        **kwargs
    ):
        super().__init__(buffer_size, observation_space, action_space, **kwargs)
        # add option to store seeds
        self.seeds = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
    def add(self, obs, next_obs, action, reward, done, infos, seed=None):
        if "prev_level_seed" in infos[0]:
            self.seeds[self.pos] = np.array([info["prev_level_seed"] for info in infos])
        super().add(obs, next_obs, action, reward, done, infos)
            
    def prune_envs(self, max_env_idx): 
        self.observations = self.observations[:, :max_env_idx]
        self.next_observations = self.next_observations[:, :max_env_idx]
        self.actions = self.actions[:, :max_env_idx]
        self.rewards = self.rewards[:, :max_env_idx]
        self.dones = self.dones[:, :max_env_idx]
        self.timeouts = self.timeouts[:, :max_env_idx]
        self.seeds = self.seeds[:, :max_env_idx]
    

class PPOWithBuffer(PPO):
    
    def __init__(self, policy, env, replay_buffer_size=100000, buffer_save_path=None, random=False,
                 policy_kwargs=None, to_uint8=False, save_on_full=False, compress=False, save_async=True,
                 save_first_n_envs=None, **kwargs):
        policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        if "activation_fn" in policy_kwargs: 
            act2fn = {"relu": th.nn.ReLU, "tanh": th.nn.Tanh, "sigmoid": th.nn.Sigmoid}
            policy_kwargs["activation_fn"] = act2fn[policy_kwargs["activation_fn"]]
        if "net_arch" in policy_kwargs and not isinstance(policy_kwargs["net_arch"], list):
            policy_kwargs["net_arch"] = [policy_kwargs["net_arch"]]
        if "impala" in policy_kwargs:
            impala = policy_kwargs.pop("impala")
            if impala: 
                policy_kwargs["features_extractor_class"] = ImpalaCNN
                # policy_kwargs["features_extractor_kwargs"] = dict(features_dim=256)
        if "texture" in policy_kwargs:
            texture = policy_kwargs.pop("texture")
            if texture: 
                policy_kwargs["features_extractor_class"] = TextureFeatureExtractor
                # policy_kwargs["features_extractor_kwargs"] = dict(features_dim=256)
        super().__init__(policy, env, policy_kwargs=policy_kwargs, **kwargs)
        self.save_buffer = buffer_save_path is not None
        self.save_on_full = save_on_full
        self.original_buffer_save_path = buffer_save_path
        self.replay_buffer_size = replay_buffer_size
        self.random = random
        self.compress = compress 
        self.to_uint8 = to_uint8
        self.save_async = save_async
        self.save_first_n_envs = save_first_n_envs
        self.buffer_idx = 0    
        if self.save_buffer:
            self.reinit_buffer() 
            if self.save_on_full and self.save_async: 
                self.executor = ThreadPoolExecutor(max_workers=2)
    
    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
                # make random values/log probs 

            if self.random: 
                actions = np.stack([env.action_space.sample() for _ in range(env.num_envs)])
                values, log_probs = th.from_numpy(np.zeros_like(actions)), th.from_numpy(np.zeros_like(actions))
            else: 
                with th.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    actions, values, log_probs = self.policy(obs_tensor)
                actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            
            # store obs to replay buffer
            if self.save_buffer and self.num_timesteps <= self.total_timesteps: 
                store_obs, store_next_obs = self._last_obs.copy(), new_obs.copy()
                store_rewards = rewards.copy()
                if hasattr(env, "norm_reward") and env.norm_reward: 
                    # get original reward instead 
                    store_rewards = env.get_original_reward()
                if self.to_uint8: 
                    # store image as uint8 
                    store_obs, store_next_obs = store_obs.astype(np.uint8), store_next_obs.astype(np.uint8)
                self.replay_buffer.add(obs=store_obs, 
                                       next_obs=store_next_obs, 
                                       action=actions.copy(),
                                       reward=store_rewards, done=dones.copy(), infos=infos.copy())
                if self.save_on_full and self.replay_buffer.full:
                    # Over 25M, we cannot store the whole thing, as this blows up disk space heavily. 
                    # ~2TB for only a few successful game.s 
                    if self.save_async:
                        print("Saving buffer asynchronously.")
                        buffer_copy = copy.deepcopy(self.replay_buffer)
                        buffer_save_path = copy.deepcopy(self.buffer_save_path)
                        self.executor.submit(self._save_buffer, self.compress, buffer_save_path, 
                                             buffer_copy, self.save_first_n_envs)
                    else: 
                        print("Saving buffer.")
                        self._save_buffer(self.compress, self.buffer_save_path, 
                                          self.replay_buffer, self.save_first_n_envs)
                    self.buffer_idx += 1
                    self.reinit_buffer()
            
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
    def learn(
        self,
        total_timesteps: int,
        callback = None,
        log_interval: int = 1,
        eval_env = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO",
        eval_log_path = None,
        reset_num_timesteps: bool = True,
    ) -> "PPO":
        self.total_timesteps = total_timesteps 
        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                if len(self.ep_success_buffer) > 0:
                    self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))    
                
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()
        
        if self.save_buffer:
            # dump replay buffer
            self._save_buffer(self.compress, self.buffer_save_path, self.replay_buffer, self.save_first_n_envs)
            if self.save_on_full and self.save_async: 
                self.close_executer()
        return self
    
    def _save_buffer(self, compress, buffer_save_path, replay_buffer_copy, save_first_n_envs=False):
        if save_first_n_envs: 
            replay_buffer_copy.prune_envs(save_first_n_envs)
        if compress:
            save_to_compressed_pkl(buffer_save_path, replay_buffer_copy)
        else:
            save_to_pkl(buffer_save_path, replay_buffer_copy, True)
        print("Done saving buffer")

    def reinit_buffer(self): 
        # needs to be the actual file name. 
        assert self.original_buffer_save_path is not None, "Please specify buffer save path."
        buffer_save_path = Path(self.original_buffer_save_path.replace(" ", ""))
        parent, name, suffix = buffer_save_path.parent, buffer_save_path.stem, buffer_save_path.suffix
        if self.compress: 
            assert suffix in [".gz", ".xz", ".pbz2", ".dat"], "Compressed replay buffer must have .gz or .xz or suffix."
        self.buffer_save_path = parent / f"{name}_{self.buffer_idx}{suffix}"
        self.buffer_save_path.parent.mkdir(parents=True, exist_ok=True)
        self.replay_buffer = CustomReplayBuffer(
            self.replay_buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
        )

    def close_executer(self):
        # Call this method to cleanly shut down the background thread
        self.executor.shutdown(wait=True)
