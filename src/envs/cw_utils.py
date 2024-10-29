import gym
import random
import numpy as np
from copy import deepcopy
from gym import RewardWrapper
from stable_baselines3.common.monitor import Monitor
from continualworld.envs import (
    MT50, META_WORLD_TIME_HORIZON,
    RandomizationWrapper, OneHotAdder, TimeLimit, SuccessCounter,
    get_task_name, get_subtasks,
    ContinualLearningEnv
)
from metaworld import Benchmark, _make_tasks, _MT_OVERRIDE
import metaworld.envs.mujoco.env_dict as _env_dict


class EpisodicRewardWrapper(RewardWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0 
        
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        if done:
            reward = self.episode_reward
            self.episode_reward = 0
        else: 
            reward = 0
        return observation, reward, done, info
    
    
class DropZeroDimsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.keep_dims = ~np.array([
            False, False, False, False, False, False, False, False, False, 
            False, False, False, False, False,  True,  True,  True,  True,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False,  True,  True,  True,  True,
            False, False, False
        ])
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                                shape=(self.keep_dims.sum(),), dtype=np.float32)
        
    def observation(self, obs):
        return obs[self.keep_dims]


class HideGoalWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # states are 39-dimensional in metaworld, last 3 dims represent 3D goal position
        self.is_goal = np.array([
            False, False, False, False, False, False, False, False, False, 
            False, False, False, False, False,  False,  False,  False,  False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False,  False,  False,  False,  False,
            True, True, True
        ])
        
    def observation(self, obs):
        obs[self.is_goal] = 0
        return obs
    

class CustomRandomizationWrapper(gym.Wrapper): 
    
    def __init__(self, env: gym.Env, subtasks, kind: str) -> None:
        super().__init__(env)
        self.subtasks = subtasks
        self.kind = kind

        env.set_task(subtasks[0])
        if kind == "random_init_all":
            env._freeze_rand_vec = False

        if kind == "random_init_fixed20":
            assert len(subtasks) >= 20

        if kind == "random_init_small_box":
            diff = env._random_reset_space.high - env._random_reset_space.low
            self.reset_space_low = env._random_reset_space.low + 0.45 * diff
            self.reset_space_high = env._random_reset_space.low + 0.55 * diff
    
    def reset(self, **kwargs) -> np.ndarray:
        if self.kind == "random_init_fixed": 
            self.env.set_task(self.subtasks[random.randint(0, len(self.subtasks) - 1)])
        elif self.kind == "random_init_fixed20":
            self.env.set_task(self.subtasks[random.randint(0, 19)])
        elif self.kind == "random_init_small_box":
            rand_vec = np.random.uniform(
                self.reset_space_low, self.reset_space_high, size=self.reset_space_low.size
            )
            self.env._last_rand_vec = rand_vec

        return self.env.reset(**kwargs)
    

class ML50(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.MT50_V2
        self._test_classes = _env_dict.MT50_V2
        train_kwargs = _env_dict.MT50_V2_ARGS_KWARGS
        # Note: we always fix seed=0 for train tasks, seed=1 for test tasks
        # this ensure that we have distinct goal locations for training and testing 
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs, _MT_OVERRIDE, seed=1)
        self._test_tasks = _make_tasks(self._test_classes, train_kwargs, _MT_OVERRIDE, seed=2)


# unfortunately, setting up the benchmark takes a long time. 
ML50Benchmark = None 


def get_ml50_subtasks(name, kind): 
    if kind == "test": 
        return [s for s in ML50Benchmark.test_tasks if s.env_name == name]
    return [s for s in ML50Benchmark.train_tasks if s.env_name == name]


def get_single_env(
        task,
        one_hot_idx: int = 0,
        one_hot_len: int = 1,
        randomization: str = "random_init_all",
        add_task_ids: bool = True, 
        episodic: bool = False,
        time_limit=None,
        kind=None,
        render_mode=None
):
    """
    Wrappers for original get_single_env() in CW. Adds functionality to optionally add task ids.

    Returns a single task environment.

    Appends one-hot embedding to the observation, so that the model that operates on many envs
    can differentiate between them.

    Args:
      task: task name or MT50 number
      one_hot_idx: one-hot identifier (indicates order among different tasks that we consider)
      one_hot_len: length of the one-hot encoding, number of tasks that we consider
      randomization: randomization kind, one of 'deterministic', 'random_init_all',
                     'random_init_fixed20', 'random_init_small_box'.
      add_task_ids: Bool.


    Returns:
      gym.Env: single-task environment
    """
    global ML50Benchmark
    task_name = get_task_name(task)
    if kind is not None: 
        if ML50Benchmark is None: 
            ML50Benchmark = ML50()
        env = ML50Benchmark.test_classes[task_name]() if kind == "test" else ML50Benchmark.train_classes[task_name]()
        env = CustomRandomizationWrapper(env, get_ml50_subtasks(task_name, kind), randomization)
    else: 
        env = MT50.train_classes[task_name]()
        env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
    if add_task_ids:
        env = OneHotAdder(env, one_hot_idx=one_hot_idx, one_hot_len=one_hot_len)
    time_limit = META_WORLD_TIME_HORIZON if time_limit is None else time_limit
    env = TimeLimit(env, time_limit)
    env = SuccessCounter(env)
    if episodic: 
        env = EpisodicRewardWrapper(env)
    env.time_limit = time_limit
    env.name = task_name
    env.num_envs = 1
    if render_mode is not None: 
        env.metadata.update({"render.modes": [render_mode]})
    return env


class ContinualLearningEnvv2(gym.Env):
    def __init__(self, envs, steps_per_env: int) -> None:
        """
        Same as ContinualLearningEnv, but removes observation_space asserts.
        v2 envs have a different observation space than v1 envs. Thus, cannot use the same asserts.
        """
        self.action_space = envs[0].action_space
        self.observation_space = deepcopy(envs[0].observation_space)
        self.envs = envs
        self.num_envs = len(envs)
        self.steps_per_env = steps_per_env
        self.steps_limit = self.num_envs * self.steps_per_env
        self.cur_step = 0
        self.cur_seq_idx = 0

    def _check_steps_bound(self) -> None:
        if self.cur_step >= self.steps_limit:
            raise RuntimeError("Steps limit exceeded for ContinualLearningEnv!")

    def pop_successes(self):
        all_successes = []
        self.avg_env_success = {}
        for env in self.envs:
            successes = env.pop_successes()
            all_successes += successes
            if len(successes) > 0:
                self.avg_env_success[env.name] = np.mean(successes)
        return all_successes

    def step(self, action):
        self._check_steps_bound()
        obs, reward, done, info = self.envs[self.cur_seq_idx].step(action)
        info["seq_idx"] = self.cur_seq_idx

        self.cur_step += 1
        if self.cur_step % self.steps_per_env == 0:
            # If we hit limit for current env, end the episode.
            # This may cause border episodes to be shorter than 200.
            done = True
            info["TimeLimit.truncated"] = True

            self.cur_seq_idx += 1

        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        self._check_steps_bound()
        return self.envs[self.cur_seq_idx].reset()


def get_cl_env(
        tasks, steps_per_task: int, randomization: str = "random_init_all", add_task_ids: bool = True, v2: bool = False
):
    """
    Wrappers for original get_single_env() in CW. Adds functionality to optionally add task ids.

    Returns continual learning environment.

    Args:
      tasks: list of task names or MT50 numbers
      steps_per_task: steps the agent will spend in each of single environments
      randomization: randomization kind, one of 'deterministic', 'random_init_all',
                     'random_init_fixed20', 'random_init_small_box'.

    Returns:
      gym.Env: continual learning environment
    """
    task_names = [get_task_name(task) for task in tasks]
    num_tasks = len(task_names)
    envs = []
    for i, task_name in enumerate(task_names):
        env = MT50.train_classes[task_name]()
        env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
        if add_task_ids:
            env = OneHotAdder(env, one_hot_idx=i, one_hot_len=num_tasks)
        # else:
        #     env = OneHotAdder(env, one_hot_idx=0, one_hot_len=1)
        env.name = task_name
        env = TimeLimit(env, META_WORLD_TIME_HORIZON)
        env = SuccessCounter(env)
        envs.append(env)
    if v2:
        cl_env = ContinualLearningEnvv2(envs, steps_per_task)
    else:
        cl_env = ContinualLearningEnv(envs, steps_per_task)
    cl_env.name = "ContinualLearningEnv"
    return cl_env


def get_single_cw_env(task, one_hot_idx, one_hot_len, randomization, add_task_ids, episodic=False,
                      time_limit=None, kind=None, render_mode=None):
    def make():
        return Monitor(get_single_env(task, one_hot_idx=one_hot_idx, one_hot_len=one_hot_len,
                                      randomization=randomization, add_task_ids=add_task_ids,
                                      episodic=episodic, time_limit=time_limit, kind=kind, render_mode=render_mode))
    return make


def get_cw_env_constructors(env_names, randomization, remove_task_ids=False, 
                            add_task_ids=True, episodic=False, time_limit=None, kind=None, render_mode=None):
    if not isinstance(env_names, (list, tuple)):
        env_names = [env_names]
    constructors = []
    one_hot_len = len(env_names) if not remove_task_ids else 1
    for i, task in enumerate(env_names):
        one_hot_idx = i if not remove_task_ids else 0
        constructors.append(
            get_single_cw_env(task, one_hot_idx=one_hot_idx, one_hot_len=one_hot_len,
                              randomization=randomization, add_task_ids=add_task_ids, episodic=episodic,
                              time_limit=time_limit,kind=kind, render_mode=render_mode)
        )
    return constructors
