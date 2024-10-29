import itertools
import gym
import numpy as np
import cv2
from nle import nethack
from minihack.envs import register
from minihack import MiniHackNavigation, LevelGenerator, RewardManager
from minihack.reward_manager import SequentialRewardManager
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from omegaconf.listconfig import ListConfig


class ExtraDictWrapper(gym.ObservationWrapper):
    """
    Wrapper to extract a specific key from a dictionary observation space.
        
    """
    def __init__(self, env: gym.Env, obs_key="tty_cursor") -> None:
        super().__init__(env)
        self.obs_key = obs_key
        self.observation_space = env.observation_space.spaces[obs_key]

    def observation(self, obs: dict):
        return obs[self.obs_key]


class MiniHackRoomCustom(MiniHackNavigation):
    def __init__(
        self,
        *args,
        size=5,
        n_monster=0,
        n_trap=0,
        penalty_step=0,
        random=True,
        lit=True,
        goal_pos=None,
        start_pos=None,
        sparse_reward=False,
        dense_reward=False,
        width=None, 
        **kwargs
    ):
        """
        Custom version of MinihackRoom, empty room in which goal location is present.  
        Action space is up, right, down, left, do nothing.
        
        Args:
            size (int): The size of the grid.
            n_monster (int): The number of monsters in the environment.
            n_trap (int): The number of traps in the environment.
            penalty_step (float): The penalty for each step taken. We turn it off by default. 
            random (bool): Whether to set start_pos and goal_pos randomly or not. 
            lit (bool): Whether the environment is lit or not. For DarkRoom doesn't matter, as agent only sees x-y. 
            goal_pos (tuple): The position of the goal.
            start_pos (tuple): The position of the starting point.
            sparse_reward (bool): Whether to use sparse rewards or not. Reward only obtained once at goal state, episode
                continues after reward is obtained. 
            dense_reward (bool): Whether to use dense rewards or not. Reward in every step at goal state. 
            width (int): The width of the grid.
            **kwargs: Additional keyword arguments.

        """
        self.goal_pos = goal_pos
        self.start_pos = start_pos 
        # sparse_reward --> reward can only be collected once, but episodes do not end when collected
        # dense_reward --> reward can be collected multiple times, but episodes do not end when collected
        self.sparse_reward = sparse_reward
        self.dense_reward = dense_reward
        self.size = size
        self.width = width 
        kwargs["max_episode_steps"] = kwargs.pop(
            "max_episode_steps", 100
        )
        lvl_gen = LevelGenerator(w=size if self.width is None else self.width, h=size, lit=lit)
        
        if not sparse_reward and not dense_reward:
            if random and goal_pos is None and start_pos is None:
                lvl_gen.add_goal_pos()
            else:
                lvl_gen.add_goal_pos((size - 1, size - 1) if goal_pos is None else goal_pos)
                lvl_gen.set_start_pos((0, 0) if start_pos is None else start_pos)
        else:
            lvl_gen.set_start_pos((0, 0) if start_pos is None else start_pos)
            lvl_gen.add_fountain(place=goal_pos)
            reward_manager = RewardManager()
            if sparse_reward:
                # if reaches fountain, give reward of 1, but only once
                # for some reason, the env would stop, once the goal is reached once
                # therefore we make a second event, without reward
                reward_manager.add_location_event(location="fountain", reward=1, repeatable=False)
                reward_manager.add_location_event(location="fountain", reward=0, repeatable=True)
            elif dense_reward:
                # if reaches fountain, give reward of 1, every time
                reward_manager.add_location_event(location="fountain", reward=1, repeatable=True)
            kwargs["reward_manager"] = kwargs.pop("reward_manager", reward_manager)
        for _ in range(n_monster):
            lvl_gen.add_monster()

        for _ in range(n_trap):
            lvl_gen.add_trap()
            
        # up, right, down, left, do nothing
        actions = tuple(nethack.CompassCardinalDirection) + (ord("."),)	       
         
        super().__init__(*args, des_file=lvl_gen.get_des(), 
                         actions=actions, penalty_step=penalty_step, **kwargs)


class MinihackKeyDoor(MiniHackNavigation):
    def __init__(
        self,
        *args,
        size=5,
        n_monster=0,
        n_trap=0,
        penalty_step=0,
        random=False,
        lit=True,
        goal_pos=None,
        start_pos=None,
        key_pos=None,
        width=None, 
        **kwargs
    ):
        """
        Custom version of MinihackRoom in which a key and a goal location is present. 
        The goal location is locked and can only be opened with the key.
        Reward is received ones for picking up the key. If the goal location is found, the agent 
        receives a reward in every timestep it stays on the goal location. 
        Action space is up, right, down, left, do nothing.

        Args:
            size (int): The size of the grid.
            n_monster (int): The number of monsters in the environment.
            n_trap (int): The number of traps in the environment.
            penalty_step (float): The penalty for each step taken. We turn it off by default. 
            random (bool): Whether to set start_pos and goal_pos randomly or not. 
            lit (bool): Whether the environment is lit or not. For DarkRoom doesn't matter, as agent only sees x-y. 
            goal_pos (tuple): The position of the goal.
            start_pos (tuple): The position of the starting point.
            key_pos (tuple): The position of the key.
            width (int): The width of the grid.
            **kwargs: Additional keyword arguments.

        """
        self.goal_pos = goal_pos
        self.start_pos = start_pos 
        self.key_pos = key_pos
        self.size = size
        self.width = width 
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)
        # make sure to use autopickup of key, no pickup action required then
        kwargs["autopickup"] = True
        lvl_gen = LevelGenerator(w=size if self.width is None else self.width, h=size, lit=lit)
        lvl_gen.set_start_pos((0, 0) if start_pos is None else start_pos)

        # add key
        lvl_gen.add_object(symbol="(", name="skeleton key", place=key_pos)
        # construct door - open by default, such that agent can walk through
        lvl_gen.add_door(place=goal_pos, state="open")
        
        # sequential reward manager ensures that key has to be collected before reward on door goal can be collected
        reward_manager = SequentialRewardManager()
        # if reaches key automatically pick up, give reward of 1 once
        reward_manager.add_location_event(location="key", reward=1, repeatable=False)
        # if reaches door and has key, give reward of 1, every time. if no key, no reward.
        reward_manager.add_location_event(location="door", reward=1, repeatable=True)
        kwargs["reward_manager"] = kwargs.pop("reward_manager", reward_manager)
        
        for _ in range(n_monster):
            lvl_gen.add_monster()

        for _ in range(n_trap):
            lvl_gen.add_trap()
            
        # up, right, down, left, do nothing
        actions = tuple(nethack.CompassCardinalDirection) + (ord("."),)	       
         
        super().__init__(*args, des_file=lvl_gen.get_des(), 
                         actions=actions, penalty_step=penalty_step, **kwargs)


class MiniHackRoom10x10Dark(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=10, **kwargs)


class MiniHackRoom17x17Dark(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=17, **kwargs)

        
class MiniHackRoom10x10DarkDense(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=10, dense_reward=True, **kwargs)


class MiniHackRoom17x17DarkDense(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=17, dense_reward=True, **kwargs)
        

class MiniHackRoom20x20DarkDense(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        max_episode_steps = kwargs.pop("max_episode_steps", 400)
        super().__init__(*args, size=20, dense_reward=True, max_episode_steps=max_episode_steps, **kwargs)


class MiniHackRoom40x20DarkDense(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        max_episode_steps = kwargs.pop("max_episode_steps", 800)
        super().__init__(*args, width=40, size=20, dense_reward=True, max_episode_steps=max_episode_steps, **kwargs)


class MiniHackRoom10x10DarkSparse(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=10, sparse_reward=True, **kwargs)
    

class MiniHackRoom17x17DarkSparse(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=17, sparse_reward=True, **kwargs)
        
        
class MiniHackKeyDoor10x10DarkDense(MinihackKeyDoor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=10, **kwargs)
        
        
class MiniHackKeyDoor5x5DarkDense(MinihackKeyDoor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=6, **kwargs)
        

class MiniHackKeyDoor20x20DarkDense(MinihackKeyDoor):
    
    def __init__(self, *args, **kwargs):
        max_episode_steps = kwargs.pop("max_episode_steps", 400)
        super().__init__(*args, size=20, max_episode_steps=max_episode_steps, **kwargs)


class MiniHackKeyDoor40x20DarkDense(MinihackKeyDoor):
    
    def __init__(self, *args, **kwargs):
        max_episode_steps = kwargs.pop("max_episode_steps", 800)
        super().__init__(*args, width=40, size=20, max_episode_steps=max_episode_steps, **kwargs)

        
class ToRealCoordinateWrapper(gym.ObservationWrapper):
    """
    Converts the screen minihack coordinates to real coordinates in range [0, env_size]
    
    Args: 
        env: Gym environment.
    """

    def __init__(self, env: gym.Env):
        gym.ObservationWrapper.__init__(self, env)
        # extract size from env, get starting x-y positions
        self.env_size = env.size
        self.env_width = env.width if hasattr(env, "width") else None
        self.origin = self.get_origin_xy(self.env_size)

    def observation(self, obs):
        return (obs[0] - self.origin[0], obs[1] - self.origin[1])
    
    def get_origin_xy(self, size): 
        if size == 10:
            return (8, 34)    
        elif size == 17: 
            return (4, 30)
        elif size == 20 and self.env_width == 40: 
            return (2, 20)
        elif size == 20: 
            return (2, 30)
        raise ValueError(f"Size {size} not supported.")
    
    
class WarpFrame(gym.ObservationWrapper):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84, grayscale=False) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        assert isinstance(env.observation_space, gym.spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1 if self.grayscale else 3),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        assert cv2 is not None, "OpenCV is not installed, you can do `pip install opencv-python`"
        if self.grayscale: 
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame if not self.grayscale else frame[:, :, None]


def register_dark_envs():     
    register(
        id="MiniHack-Room-Dark-10x10-v0",
        entry_point=MiniHackRoom10x10Dark,
    )
    register(
        id="MiniHack-Room-Dark-17x17-v0",
        entry_point=MiniHackRoom17x17Dark,
    )
    register(
        id="MiniHack-Room-Dark-Sparse-10x10-v0",
        entry_point=MiniHackRoom10x10DarkSparse,
    )
    register(
        id="MiniHack-Room-Dark-Sparse-17x17-v0",
        entry_point=MiniHackRoom17x17DarkSparse,
    )

    register(
        id="MiniHack-Room-Dark-Dense-10x10-v0",
        entry_point=MiniHackRoom10x10DarkDense,
    )
    register(
        id="MiniHack-Room-Dark-Dense-17x17-v0",
        entry_point=MiniHackRoom17x17DarkDense,
    )
    
    register(
        id="MiniHack-Room-Dark-Dense-20x20-v0",
        entry_point=MiniHackRoom20x20DarkDense,
    )
    register(
        id="MiniHack-Room-Dark-Dense-40x20-v0",
        entry_point=MiniHackRoom40x20DarkDense,
    )
    register(
        id="MiniHack-KeyDoor-Dark-Dense-10x10-v0",
        entry_point=MiniHackKeyDoor10x10DarkDense,
    )
    register(
        id="MiniHack-KeyDoor-Dark-Dense-5x5-v0",
        entry_point=MiniHackKeyDoor5x5DarkDense,
    )
    register(
        id="MiniHack-KeyDoor-Dark-Dense-20x20-v0",
        entry_point=MiniHackKeyDoor20x20DarkDense,
    )
    register(
        id="MiniHack-KeyDoor-Dark-Dense-40x20-v0",
        entry_point=MiniHackKeyDoor40x20DarkDense,
    )
    
    
register_dark_envs()


def get_minihack_constructor(envid, env_kwargs=None, goal_pos=None, start_pos=None, key_pos=None):
    env_kwargs = env_kwargs if env_kwargs is not None else {}
    # need be tuples if given
    goal_pos = tuple(goal_pos) if goal_pos is not None else goal_pos
    start_pos = tuple(start_pos) if start_pos is not None else start_pos
    key_pos = tuple(key_pos) if key_pos is not None else key_pos
    def make():
        if "Room-Dark" in envid: 
            env = gym.make(envid, goal_pos=goal_pos, start_pos=start_pos, **env_kwargs)
        elif "KeyDoor-Dark" in envid: 
            env = gym.make(envid, goal_pos=goal_pos, start_pos=start_pos, key_pos=key_pos, **env_kwargs)
        else: 
            env = gym.make(envid, **env_kwargs)
        if "Dense" in envid:                 
            env.name = f"{envid}_{str(start_pos).replace(' ', '')}_{str(goal_pos).replace(' ', '')}"
            if key_pos is not None: 
                env.name += f"_{str(key_pos).replace(' ', '')}"
        observation_keys = env_kwargs.get("observation_keys", ["tty_cursor"])
        env = ExtraDictWrapper(env, observation_keys[0])
        if observation_keys[0] == "tty_cursor": 
            env = ToRealCoordinateWrapper(env)
        if observation_keys[0] == "pixel_crop": 
            env = WarpFrame(env, width=84, height=84, grayscale=False)
        return Monitor(env)
    return make


def get_minihack_constructors(envid, env_kwargs=None, goal_pos=None, start_pos=None, key_pos=None):
    # Case 1: None --> convery to list
    # Case 2: single list --> convert to list of lists
    # Case 3: already list of lists --> then Omegaconf list --> do nothing
    if goal_pos is None: 
        goal_pos = [goal_pos]
    elif isinstance(goal_pos, (tuple, list, ListConfig)): 
        goal_pos = [goal_pos] if isinstance(goal_pos[0], int) else goal_pos
    if start_pos is None:
        start_pos = [start_pos] 
    elif isinstance(start_pos, (tuple, list, ListConfig)):
        start_pos = [start_pos] if isinstance(start_pos[0], int) else start_pos 
    if isinstance(key_pos, (tuple, list, ListConfig)):
        key_pos = [key_pos] if isinstance(key_pos[0], int) else key_pos 
        assert len(key_pos) == len(goal_pos), "Number of key positions and goal positions must be the same."
    # repeat shorter one
    if len(start_pos) < len(goal_pos):
        start_pos = itertools.cycle(start_pos)
    elif len(start_pos) > len(goal_pos):
        goal_pos = itertools.cycle(goal_pos)
    if key_pos is not None: 
        return [get_minihack_constructor(envid, env_kwargs, goal, start, key) for goal, start, key in zip(goal_pos, start_pos, key_pos)]
    else:  
        return [get_minihack_constructor(envid, env_kwargs, goal, start) for goal, start in zip(goal_pos, start_pos)]
    

def make_train_test_pos(num_pos=100, size=10, percent_train=0.8, width=None): 
    if width is None: 
        width = size
    n_train = int(num_pos * percent_train)
    pos = [(i, j) for i in range(width) for j in range(size)]
    start_pos = np.random.RandomState(seed=42).permutation(pos)[:num_pos]
    goal_pos = np.random.RandomState(seed=43).permutation(pos)[:num_pos]
    train_start, train_goal = start_pos[:n_train].tolist(), goal_pos[:n_train].tolist()
    test_start, test_goal = start_pos[n_train:].tolist(), goal_pos[n_train:].tolist()
    assert [start != goal for start, goal in zip(train_start, train_goal)], "Start and goal are same in train."
    assert [start != goal for start, goal in zip(test_start, test_goal)], "Start and goal are same in test."
    return train_start, train_goal, test_start, test_goal


def make_minihack_envs(env_params, make_eval_env=True):
    const_kwargs = {
        "envid": env_params.envid,
        "env_kwargs": env_params.get("env_kwargs", {}),
        "goal_pos": env_params.get("train_goal_pos", None), 
        "start_pos": env_params.get("train_start_pos", None)  
    }
    if hasattr(env_params, "train_key_pos"):
        const_kwargs["key_pos"] = env_params.train_key_pos
    env = DummyVecEnv(get_minihack_constructors(**const_kwargs))
    if make_eval_env:
        goal_pos, start_pos = env_params.get("eval_goal_pos", None), env_params.get("eval_start_pos", None)
        const_kwargs.update({"goal_pos": goal_pos, "start_pos": start_pos})
        if hasattr(env_params, "eval_key_pos"):
            const_kwargs["key_pos"] = env_params.eval_key_pos
        eval_env = DummyVecEnv(get_minihack_constructors(**const_kwargs))
        eval_env.num_envs = 1
    env.num_envs = 1
    return env, eval_env
