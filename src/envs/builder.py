import collections
from functools import partial
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder, VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from .env_names import ID_TO_NAMES, ID_TO_DOMAIN


def extract_full_env_names(env_names):
    """
    Extractst the actual full env names from given env_names. Env names may contain
    abbreviations such as "atari46" or "mt40", which need to be mapped to the actual environment 
    names for the 46 specified Atari games or the 40 Meta-world tasks, respetively.

    Args:
        env_names: Str or List. 

    Returns: List of env names
    
    """
    if not env_names:
        return None
    if not isinstance(env_names, (list, tuple)):
        env_names = [env_names]
    all_names = []
    for env_name in env_names: 
        names = ID_TO_NAMES.get(env_name, [env_name]) if isinstance(env_name, str) else list(env_name)
        all_names += names
    return all_names


def get_domains_for_env_names(env_names):
    """
    Extract domain names for each env names. 
    Constructs a dict that maps domains to env names. 

    Args:
        env_names: List of env names.

    Returns:
        Dict. Maps domains to env names.
        
    """
    domain_to_envs = collections.defaultdict(list)
    for name in env_names:
        domain = ID_TO_DOMAIN.get(name, "other")
        if "minihack" in name.lower():
            domain_to_envs["minihack"].append(name)
        elif "dummy" in name.lower():
            domain_to_envs["dummy"].append(name)
        else: 
            domain_to_envs[domain].append(name)
    return domain_to_envs


def make_eval_envs(eval_env_names, env_params):
    """
    Generate eval envs from env_names of each domain.

    Args:
        domain_to_envs: Dict. Contains domain-envname pairs. Each envname should be generated.   
        
    Returns: VecEnv
    
    """
    eval_envs = []
    if not isinstance(eval_env_names, (list, tuple)):
        eval_env_names = [eval_env_names]
    domain_to_envs = get_domains_for_env_names(eval_env_names)
    for domain, env_names in domain_to_envs.items(): 
        if domain == "atari": 
            from .env_utils import make_multi_atari_env
            env = make_multi_atari_env(env_names, seed=1, vec_env_cls=DummyVecEnv, 
                                       env_kwargs=env_params.get("atari_env_kwargs", {}),
                                       wrapper_kwargs=env_params.get("wrapper_kwargs", {}))
        elif domain == "other":
            from .env_utils import make_multi_vec_env
            env = make_multi_vec_env(env_names, seed=1, vec_env_cls=DummyVecEnv, 
                                     env_kwargs=env_params.get("env_kwargs", {}))
        elif domain == "mt50": 
            from .cw_utils import get_cw_env_constructors
            env = DummyVecEnv(
                get_cw_env_constructors(
                    env_names, randomization=env_params.randomization,
                    remove_task_ids=env_params.get("remove_task_ids", False),
                    add_task_ids=env_params.add_task_ids,
                    episodic=env_params.get("episodic", False),
                    time_limit=env_params.get("time_limit", None),
                    kind="test" if env_params.get("fixed_loc", False) else None,
                    render_mode=env_params.get("render_mode", None)
                )
            )
        elif domain == "dmcontrol":
            from .dmcontrol_utils import get_dmcontrol_constructor
            env = DummyVecEnv([get_dmcontrol_constructor(name, env_params.get("dmc_env_kwargs", {})) 
                               for name in env_names])
        elif domain == "minihack":
            from .minihack_utils import get_minihack_constructors           
            env_kwargs = env_params.get("env_kwargs", {})
            goal_pos, start_pos = env_params.get("eval_goal_pos", None), env_params.get("eval_start_pos", None)
            env = DummyVecEnv(
                get_minihack_constructors(env_params.envid, env_kwargs=env_kwargs, goal_pos=goal_pos, start_pos=start_pos)
            )
        elif domain == "composuite":
            from .composuite_utils import get_composuite_constructors
            env = DummyVecEnv(get_composuite_constructors(env_names, env_params.get("cs_env_kwargs", {})))
        elif domain == "mimicgen":
            from .mimicgen_utils import get_mimicgen_constructors
            env = DummyVecEnv(get_mimicgen_constructors(env_names, env_params.get("mimicgen_env_kwargs", {})))
        elif domain == "procgen": 
            from .procgen_utils import get_procgen_constructor, CustomDummyVecEnv
            env = CustomDummyVecEnv([
                get_procgen_constructor(name, env_params.distribution_mode, env_params.get("time_limit", None),
                                        env_params.get("eval_env_kwargs", {}))
                for name in env_names
            ])
        elif domain == "dummy": 
            from .dummy_env_utils import get_dummyenv_constructors
            env = DummyVecEnv(get_dummyenv_constructors(env_names, env_params.get("env_kwargs", {})))
        else: 
            raise NotImplementedError(f"Domain {domain} not implemented yet.")
        env.num_envs = 1
        eval_envs.append(env)
    
    if len(eval_envs) == 1: 
        return eval_envs[0]
    
    def make_env_fn(env):
        def _init():
            return env
        return _init
    
    # compose all envs to new DummyVecEnv. hacky, but necessary for sb3.
    eval_env_fns = []
    for env in eval_envs:
        eval_env_fns += [make_env_fn(e) for e in env.envs]
    eval_env = DummyVecEnv(eval_env_fns)
    eval_env.num_envs = 1 
    return eval_env


def wrap_envs(env, eval_env, env_params, eval_env_names, domain, logdir, train_eval_env=None): 
    wrappers = []
    if hasattr(env_params, "frame_stack"):
        wrappers.append(partial(VecFrameStack, n_stack=env_params.frame_stack))
    if domain in ["atari"] \
        or (domain == "minihack" and is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space)) \
        or (domain == "mimicgen" and is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space)):
        wrappers.append(VecTransposeImage)
    norm_reward = env_params.get("norm_reward", False)
    if env_params.norm_obs or norm_reward:
        wrappers.append(partial(VecNormalize, norm_obs=env_params.norm_obs, norm_reward=norm_reward, clip_reward=10.0))
    for wrapper in wrappers:
        env = wrapper(env)
        if not eval_env_names:
            eval_env = wrapper(eval_env) 
        if train_eval_env is not None: 
            train_eval_env = wrapper(train_eval_env)
    if env_params.record:
        env = VecVideoRecorder(env, f"{logdir}/videos",
                                record_video_trigger=lambda x: x % env_params.record_freq == 0,
                                video_length=env_params.record_length)
        if hasattr(env_params, "record_eval") and env_params.record_eval: 
            eval_env = VecVideoRecorder(eval_env, f"{logdir}/videos",
                                        record_video_trigger=lambda x: x % env_params.record_freq == 0,
                                        video_length=env_params.record_length)
            # make start_video_recorder a noop for train env - we recreate rec in evaluation.py
            env.start_video_recorder = lambda: True
        if train_eval_env is not None: 
            train_eval_env = VecVideoRecorder(train_eval_env, f"{logdir}/videos",
                                              record_video_trigger=lambda x: x % env_params.record_freq == 0,
                                              video_length=env_params.record_length)
            # make start_video_recorder a noop for train env - we recreate rec in evaluation.py
            env.start_video_recorder = lambda: True
    return env, eval_env, train_eval_env


def make_env(config, logdir):
    """
    Make train and eval environments. 
    Currently supports creating environments for Gym environments, Atari, Procgen, Meta-world,
    Continual-world and Minihack.
    
    Args:
        config: Hydra config.
        logdir: Str. Path to logdir for saving videos.

    Returns:
        env, eval_env both of type VecEnv.
        
    """
    env_params = config.env_params
    domain = ID_TO_DOMAIN.get(env_params.envid, "other")
    env_names = extract_full_env_names(env_params.envid)
    eval_env_names = extract_full_env_names(env_params.get("eval_env_names", None))
    
    train_eval_env = None
    if domain == "procgen":
        from .procgen_utils import get_procgen_constructor, CustomDummyVecEnv
        config.seed = None
        unwrap = env_params.get("unwrap", False)
        env = CustomDummyVecEnv([
            get_procgen_constructor(name, env_params.distribution_mode, env_params.get("time_limit", None),
                                    env_params.get("env_kwargs", {}))
            for name in env_names
        ])
        if unwrap: 
            env = env.envs[0]
        if not eval_env_names:         
            eval_env = CustomDummyVecEnv([
                get_procgen_constructor(name, env_params.distribution_mode, env_params.get("time_limit", None),
                                        env_params.get("eval_env_kwargs", {}))
                for name in env_names
            ])
        if env_params.get("train_eval_seeds", False): 
            # train env names, but eval env kwargs (seeds)
            train_eval_env = CustomDummyVecEnv([
                get_procgen_constructor(name, env_params.distribution_mode, env_params.get("time_limit", None),
                                        env_params.get("eval_env_kwargs", {}))
                for name in env_names
            ])
    elif "minihack" in env_params.envid.lower():
        from .minihack_utils import make_minihack_envs
        env, eval_env = make_minihack_envs(env_params, make_eval_env=not eval_env_names)
    elif domain == "composuite": 
        from .composuite_utils import make_composuite_envs
        env, eval_env = make_composuite_envs(env_params, envid=env_names, make_eval_env=not eval_env_names)
    elif domain == "mimicgen": 
        from .mimicgen_utils import make_mimicgen_envs
        env, eval_env = make_mimicgen_envs(env_params, envid=env_names, make_eval_env=not eval_env_names)
    elif domain == "dummy": 
        from .dummy_env_utils import make_dummyenv_envs
        env, eval_env = make_dummyenv_envs(env_params, envid=env_names, make_eval_env=not eval_env_names)
    elif domain == "dmcontrol": 
        from .dmcontrol_utils import get_dmcontrol_constructor
        env_kwargs = env_params.get("dmc_env_kwargs", {})
        env = DummyVecEnv([get_dmcontrol_constructor(name, env_kwargs)
                           for name in env_names])
        if not eval_env_names:         
            eval_env = DummyVecEnv([get_dmcontrol_constructor(name, env_kwargs)
                                    for name in env_names])   
    elif domain == "mt50":
        from .cw_utils import get_cw_env_constructors
        env = DummyVecEnv(
            get_cw_env_constructors(
                env_names, randomization=env_params.randomization,
                add_task_ids=env_params.add_task_ids,
                episodic=env_params.get("episodic", False),
                time_limit=env_params.get("time_limit", None),
                kind="train" if env_params.get("fixed_loc", False) else None,
                render_mode=env_params.get("render_mode", None)
            )
        )
        if not eval_env_names:
            eval_env = DummyVecEnv(
                get_cw_env_constructors(
                    env_names,
                    randomization=env_params.randomization,
                    add_task_ids=env_params.add_task_ids,
                    episodic=env_params.get("episodic", False),
                    time_limit=env_params.get("time_limit", None),
                    kind="test" if env_params.get("fixed_loc", False) else None,
                    render_mode=env_params.get("render_mode", None)
                )
            )
            eval_env.num_envs = 1
        env.num_envs = 1
    else:
        env_kwargs = env_params.get("env_kwargs", {})
        wrapper_kwargs = env_params.get("wrapper_kwargs", {})
        if domain == "atari": 
            env_kwargs = env_params.get("atari_env_kwargs", {})
            from .env_utils import make_atari_env_custom
            env_fn = make_atari_env_custom
        else: 
            from stable_baselines3.common.env_util import make_vec_env
            env_fn = make_vec_env
        env = env_fn(env_params.envid, n_envs=env_params.num_envs,
                     seed=1, vec_env_cls=DummyVecEnv, env_kwargs=env_kwargs, wrapper_kwargs=wrapper_kwargs)
        if not eval_env_names: 
            eval_env = env_fn(env_params.envid, n_envs=1, seed=1, vec_env_cls=DummyVecEnv,
                              env_kwargs=env_kwargs, wrapper_kwargs=wrapper_kwargs)
            eval_env.num_envs = 1
        
    # make eval envs for all given names (can be from multiple domains)
    if eval_env_names: 
        eval_env = make_eval_envs(eval_env_names, env_params)
    
    # wrap envs
    env, eval_env, train_eval_env = wrap_envs(env, eval_env, env_params, eval_env_names, 
                                              domain, logdir, train_eval_env=train_eval_env)

    return env, eval_env, train_eval_env
