import json
import numpy as np
import robosuite
from tqdm import tqdm
from gym.wrappers import TimeLimit
from gym import spaces
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import DummyVecEnv
from .compatibility_wrapper import GymCompatibilityWrapper


COMMON_OPTIONS = {
    "ignore_done": False,
    "hard_reset": True,
    "reward_shaping": False,
    "controller_configs": {
        "type": 'OSC_POSE', 
        'input_max': 1,
        'input_min': -1, 
        'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 
        'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 
        'kp': 150,
        'damping': 1,
        'impedance_mode': 'fixed', 
        'kp_limits': [0, 300],
        'damping_limits': [0, 10], 
        'position_limits': None,
        'orientation_limits': None, 
        'uncouple_pos_ori': True,
        'control_delta': True, 
        'interpolation': None,
        'ramp_ratio': 0.2
    }
}

STATE_OPTIONS = {
    "use_object_obs": True,
    "use_camera_obs": False, 
    "render_visual_mesh": False,
    "has_offscreen_renderer": False,
    **COMMON_OPTIONS
}

VISION_OPTIONS = {
    "use_object_obs": True,
    "use_camera_obs": True, 
    "render_visual_mesh": True,
    "has_offscreen_renderer": True,
    "camera_depths": False,
    "camera_heights": 64,
    "camera_widths": 64,
    "render_gpu_device_id": 0,
    "camera_names": ["agentview","robot0_eye_in_hand"],
    **COMMON_OPTIONS
}

MIMICGEN_OBSTYPE_TO_DIM = {
    'object': 86, 'robot0_eef_pos': 3, 'robot0_eef_pos_rel_pod': 3, 'robot0_eef_pos_rel_pod_holder': 3,
    'robot0_eef_quat': 4, 'robot0_eef_quat_rel_pod': 4, 'robot0_eef_quat_rel_pod_holder': 4, 'robot0_eef_vel_ang': 3,
    'robot0_eef_vel_lin': 3, 'robot0_gripper_qpos': 2, 'robot0_gripper_qvel': 2, 'robot0_joint_pos': 7, 
    'robot0_joint_pos_cos': 7, 'robot0_joint_pos_sin': 7, 'robot0_joint_vel': 7, 'robot0_contact': 1, 
    'robot0_eef_force_norm': 1, 'robot0_eef_pos_rel_base': 3, 'robot0_eef_pos_rel_piece_1': 3,
    'robot0_eef_pos_rel_piece_2': 3, 'robot0_eef_quat_rel_base': 4, 'robot0_eef_quat_rel_piece_1': 4,
    'robot0_eef_quat_rel_piece_2': 4
}

MIMICGEN_FULL_OBS_DIM = sum(MIMICGEN_OBSTYPE_TO_DIM.values())

MIMICGEN_OBSTYPE_TO_STARTIDX = {
    'object': 0, 'robot0_eef_pos': 86, 'robot0_eef_pos_rel_pod': 89, 'robot0_eef_pos_rel_pod_holder': 92, 
    'robot0_eef_quat': 95, 'robot0_eef_quat_rel_pod': 99, 'robot0_eef_quat_rel_pod_holder': 103, 
    'robot0_eef_vel_ang': 107, 'robot0_eef_vel_lin': 110, 'robot0_gripper_qpos': 113, 'robot0_gripper_qvel': 115, 
    'robot0_joint_pos': 117, 'robot0_joint_pos_cos': 124, 'robot0_joint_pos_sin': 131, 'robot0_joint_vel': 138, 
    'robot0_contact': 145, 'robot0_eef_force_norm': 146, 'robot0_eef_pos_rel_base': 147,
    'robot0_eef_pos_rel_piece_1': 150, 'robot0_eef_pos_rel_piece_2': 153, 'robot0_eef_quat_rel_base': 156, 
    'robot0_eef_quat_rel_piece_1': 160, 'robot0_eef_quat_rel_piece_2': 164
}

MAIN_LOWDIM_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]

TASK_TO_HORIZON = {
    # extracted from: https://github.com/NVlabs/mimicgen/blob/ea0988523f468ccf7570475f1906023f854962e9/mimicgen/scripts/generate_core_training_configs.py#L134
    'CoffeePreparation_D0': 800, 
    'CoffeePreparation_D1': 800, 
    'Coffee_D0': 400,
    'Coffee_D1': 400, 
    'Coffee_D2': 400,
    'HammerCleanup_D0': 800, 
    'HammerCleanup_D1': 800, 
    'Kitchen_D0': 800,
    'Kitchen_D1': 800, 
    'MugCleanup_D0': 500,
    'MugCleanup_D1': 500,
    'NutAssembly_D0': 500,
    'PickPlace_D0': 1000,
    'Square_D0': 400, 
    'Square_D1': 400, 
    'Square_D2': 400, 
    'StackThree_D0': 400,
    "StackThree_D1": 400,
    'Stack_D0': 400, 
    "Stack_D1": 400, 
    'Threading_D0': 400,
    'Threading_D1': 400, 
    'Threading_D2': 400,
    'ThreePieceAssembly_D0': 500,
    'ThreePieceAssembly_D1': 500,
    'ThreePieceAssembly_D2': 500
}

TASK_TO_ROBOT = {
    # exctracted from metadata of data files
    'CoffeePreparation_D0': "Panda", 
    'CoffeePreparation_D1': "Panda", 
    'Coffee_D0': "Panda",
    'Coffee_D1': "Panda", 
    'Coffee_D2': "Panda",
    'HammerCleanup_D0': "Panda", 
    'HammerCleanup_D1': "Panda", 
    'Kitchen_D0': "Panda",
    'Kitchen_D1': "Panda", 
    'MugCleanup_D0': "Panda",
    'MugCleanup_D1': "Panda",
    'NutAssembly_D0': "Sawyer",
    'PickPlace_D0': "Sawyer",
    'Square_D0': "Panda", 
    'Square_D1': "Panda", 
    'Square_D2': "Panda", 
    'StackThree_D0': "Panda",
    "StackThree_D1": "Panda",
    'Stack_D0': "Panda", 
    "Stack_D1": "Panda", 
    'Threading_D0': "Panda",
    'Threading_D1': "Panda", 
    'Threading_D2': "Panda",
    'ThreePieceAssembly_D0': "Panda",
    'ThreePieceAssembly_D1': "Panda",
    'ThreePieceAssembly_D2': "Panda"
}


def collect_env_shapes_and_stats(): 
    env_names = sorted(get_mimicgen_envs())
    shapes, stats = {}, {}
    for name in tqdm(env_names): 
        env = get_mimicgen_constructor(name)()
        s = env.reset()
        a = env.action_space.sample()
        obs = extract_current_obs(env, name)
        shapes[name] = {"states": s.shape[-1], "actions": a.shape[-1], 
                        "obs": {k: v.shape[-1] for k, v in obs.items()}}
        
        states = np.stack([env.reset() for _ in range(10)])
        actions = np.stack([env.action_space.sample() for _ in range(10)])
        stats[name] = {
            "states": (float(states.min().round(2)), float(states.max().round(2)),
                       float(states.mean().round(2)), float(states.std().round(2))),
            "actions": (actions.min(0).round(2).tolist(), actions.max(0).round(2).tolist(),
                        actions.mean(0).round(2).tolist(), actions.std(0).round(2).tolist())
        }
    with open('../data/mimicgen/env_shapes.json', 'w') as f:
        json.dump(shapes, f, indent=4)
    with open('../data/mimicgen/env_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
        
        
def extract_obs_dims():
    obstype_to_dim = {}
    env_names = sorted(get_mimicgen_envs())
    for name in env_names:
        env = get_mimicgen_constructor(name)()
        _ = env.reset()
        obs = extract_current_obs(env, name)        
        for k, v in obs.items():
            v = np.array([v]) if np.isscalar(v) else v.ravel()
            obstype_to_dim[k] = max(obstype_to_dim.get(k, 0), v.shape[0])
    return obstype_to_dim 


def extract_obstype_to_startidx(obstype_to_dim):
    cum_dim = 0
    obstype_to_start_idx = {}
    for k, v in obstype_to_dim.items():
        obstype_to_start_idx[k] = cum_dim
        cum_dim += v
    return obstype_to_start_idx


def map_obs_to_full_space(obs):
    full_obs = np.zeros(MIMICGEN_FULL_OBS_DIM, dtype=np.float32)
    for k, v in obs.items():
        start_idx = MIMICGEN_OBSTYPE_TO_STARTIDX[k]
        v = np.array([v]) if np.isscalar(v) else v.ravel()
        full_obs[start_idx: start_idx + v.shape[0]] = v
    return full_obs


def map_flattened_obs_to_full_space(obs, obs_spec): 
    if not isinstance(obs, np.ndarray): 
        obs = np.array(obs)
    is_one_dim = len(obs.shape) == 1
    if is_one_dim: 
        obs = np.expand_dims(obs, axis=0)
    full_obs = np.zeros((*obs.shape[:-1], MIMICGEN_FULL_OBS_DIM))
    flat_start_idx = 0
    for k, v in obs_spec.items():
        dim = np.prod(v.shape) if len(v.shape) > 0 else 1
        full_start_idx = MIMICGEN_OBSTYPE_TO_STARTIDX[k]
        full_obs[..., full_start_idx: full_start_idx + dim] = obs[..., flat_start_idx: flat_start_idx + dim]
        flat_start_idx += dim
    if is_one_dim:
        full_obs = full_obs.ravel()
    return full_obs


class MimicgenGymWrapper(GymWrapper):
    
    def __init__(self, env, keys=None, to_full_space=False, img_key=None,
                 low_dim_keys=MAIN_LOWDIM_KEYS, terminate_on_success=True, sparse_reward=True):
        # Make sure joint position observations and eef vel observations are active
        self.to_full_space = to_full_space
        self.img_key = img_key
        self.low_dim_keys = low_dim_keys
        self.terminate_on_success = terminate_on_success
        self.sparse_reward = sparse_reward
        self.is_success = False
        if self.sparse_reward: 
            assert self.terminate_on_success, "Binary reward only makes sense for terminate_on_success."
        for ob_name in env.observation_names:
            if ("joint_pos" in ob_name) or ("eef_vel" in ob_name):
                env.modify_observable(observable_name=ob_name, attribute="active", modifier=True)
        super().__init__(env=env, keys=keys)
        if img_key is not None: 
            obs, _ = self.reset()
            self.observation_space = spaces.Box(low=0, high=255, shape=obs.shape, dtype=obs.dtype)
    
    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        self.is_success = False
        return super().reset(seed=seed, options=options)
    
    def step(self, action):
        """
        done returned by Mimicgen envs is incorrect, it never exits early even if ignore_done=False. 
        They have a is_done() function, that always returns False. Need to circumvent, by using native robosuite done
        condition.  
        """
        obs, reward, terminated, truncated, info = super().step(action)
        success = self._check_success()
        if success: 
            self.is_success = True
        info["success"] = success or self.is_success
        done = self.env.done or terminated
        if self.terminate_on_success: 
            done = done or success
        if self.sparse_reward:
            reward = float(success)
        return obs, reward, done, truncated, info

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenates the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        if self.img_key is not None: 
            return obs_dict[self.img_key]
        new_obs_dict = self._prepare_obs_dict(obs_dict)
        
        if self.to_full_space: 
            return map_obs_to_full_space(new_obs_dict).astype(np.float32)
        
        # return super()._flatten_obs(new_obs_dict, verbose=verbose)
        ob_lst = []
        for k in new_obs_dict:
            v = new_obs_dict[k]
            v = np.expand_dims(v, axis=-1) if v.ndim == 1 else v
            if verbose: 
                print("adding key: {}".format(k))
            ob_lst.append(v.flatten())
        return np.concatenate(ob_lst).astype(np.float32)
    
    def _prepare_obs_dict(self, obs_dict): 
        # remove attributes as in EnvRobosuite
        new_obs_dict = {}
        if self.low_dim_keys is not None:
            for k in self.low_dim_keys:
                if k == "object": 
                    new_obs_dict["object"] = np.array(obs_dict["object-state"])
                else: 
                    new_obs_dict[k] = np.array(obs_dict[k])
            return new_obs_dict
        new_obs_dict["object"] = np.array(obs_dict["object-state"])
        for robot in self.env.robots:
            # add all robot-arm-specific observations. Note the (k not in ret) check
            # ensures that we don't accidentally add robot wrist images a second time
            pf = robot.robot_model.naming_prefix
            for k in obs_dict:
                if k.startswith(pf) and (k not in new_obs_dict) and (not k.endswith("proprio-state")):
                    new_obs_dict[k] = np.array(obs_dict[k])
        return dict(sorted(new_obs_dict.items()))
    
    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        if attr == "env":
            return self.env
        orig_attr = getattr(self.env, attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                # NOTE: had to use "is" to prevent errors when returning numpy arrays from a wrapped method
                if result is self.env:
                    return self
                return result

            return hooked
        else:
            return orig_attr
        
    def __reduce__(self):
        return (self.__class__, (self.env, self.keys, self.to_full_space, self.img_key,
                                 self.low_dim_keys, self.terminate_on_success, self.sparse_reward))

    
def get_mimicgen_constructor(envid, env_kwargs=None):
    if "-" in envid:
        envid, robot = envid.split("-")
    else:
        robot = TASK_TO_ROBOT.get(envid, "Panda")
    env_kwargs = dict(env_kwargs) if env_kwargs is not None else {}
    render_mode = env_kwargs.pop("render_mode", None)
    horizon = env_kwargs.pop("horizon", TASK_TO_HORIZON[envid])
    img_key = env_kwargs.pop("img_key", None)
    
    def make():
        # register mimicgen envs
        import mimicgen
        env = robosuite.make(
            env_name=envid,
            horizon=horizon,
            robots=robot, 
            **STATE_OPTIONS if img_key is None else VISION_OPTIONS,
            **env_kwargs
        )
        env = MimicgenGymWrapper(env, img_key=img_key)        
        # make gymnasium env compatible with gym
        env = GymCompatibilityWrapper(env)
        
        if "hammer" in envid.lower() or "kitchen" in envid.lower(): 
            # envs from robosuite_task_zoo cannot handle timeouts
            # --> overwritten in step() function, by `done = self._check_success()` L498, hammer_place.py
            env = TimeLimit(env, max_episode_steps=horizon)
        
        # rename for easier metric tracking
        if render_mode is not None: 
            env.metadata.update({"render.modes": [render_mode]})
        return Monitor(env)
    return make


def get_mimicgen_constructors(envid, env_kwargs=None):
    if not isinstance(envid, list):
        envid = [envid] 
    return [get_mimicgen_constructor(eid, env_kwargs=env_kwargs) for eid in envid]


def make_mimicgen_envs(env_params, envid, make_eval_env=True):
    const_kwargs = {
        "envid": envid,
        "env_kwargs": env_params.get("mimicgen_env_kwargs", {}),
    }
    env = DummyVecEnv(get_mimicgen_constructors(**const_kwargs))
    eval_env = None
    if make_eval_env:
        eval_env = DummyVecEnv(get_mimicgen_constructors(**const_kwargs))
        eval_env.num_envs = 1
    env.num_envs = 1
    return env, eval_env
        

def get_mimicgen_envs():
    """
    Get mimicgen envs

    Returns:
        str: Chosen environment name
    """
    robosuite_envs = set(robosuite.ALL_ENVIRONMENTS)
    import mimicgen
    all_envs = set(robosuite.ALL_ENVIRONMENTS)

    only_mimicgen = sorted(all_envs - robosuite_envs)
    envs = [x for x in only_mimicgen if x[-1].isnumeric() and x[-2] != "O"]
    return envs


def extract_current_obs(env, name): 
    if "hammer" in name.lower() or "kitchen" in name.lower():
        obs = env.env.env.env._prepare_obs_dict(env.unwrapped._get_observations())
    else:
        obs = env.env.env._prepare_obs_dict(env.unwrapped._get_observations())
    obs = {k: np.expand_dims(v, axis=-1) if v.ndim == 1 else v for k, v in obs.items()}
    obs = {k: v.flatten() for k, v in obs.items()}
    return obs
