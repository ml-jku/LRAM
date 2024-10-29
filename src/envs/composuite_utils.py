import composuite
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import DummyVecEnv
from composuite.env.gym_wrapper import GymWrapper
from .compatibility_wrapper import GymCompatibilityWrapper


class CustomGymWrapper(GymWrapper):
    """
    Overwrites original GymWrapper to allow for pickling. 
    """ 
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
        return (self.__class__, (self.env, self.keys))
        

def get_composuite_constructor(envid, env_kwargs=None):
    env_kwargs = dict(env_kwargs) if env_kwargs is not None else {}
    render_mode = env_kwargs.pop("render_mode", None) 
    use_task_id_obs = env_kwargs.pop("use_task_id_obs", True) 
    def make():
        robot_name, object_name, obstacle_name, objective_name = envid.split("_")
        env = composuite.make(robot_name, object_name, obstacle_name, objective_name,
                              use_task_id_obs=use_task_id_obs, ignore_done=False, **env_kwargs)
        # overwrite original GymWrapper to allow for env pickling. 
        env = CustomGymWrapper(env.unwrapped)
        # make gymnasium env compatible with gym
        env = GymCompatibilityWrapper(env)        
        # rename for easier metric tracking
        env.name = envid
        if render_mode is not None: 
            env.metadata.update({"render.modes": [render_mode]})
        return Monitor(env)
    return make


def get_composuite_constructors(envid, env_kwargs=None):
    if not isinstance(envid, list):
        envid = [envid] 
    return [get_composuite_constructor(eid, env_kwargs=env_kwargs) for eid in envid]


def make_composuite_envs(env_params, envid, make_eval_env=True):
    const_kwargs = {
        "envid": envid,
        "env_kwargs": env_params.get("cs_env_kwargs", {}),
    }
    env = DummyVecEnv(get_composuite_constructors(**const_kwargs))
    eval_env = None
    if make_eval_env:
        eval_env = DummyVecEnv(get_composuite_constructors(**const_kwargs))
        eval_env.num_envs = 1
    env.num_envs = 1
    return env, eval_env
