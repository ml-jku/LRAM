import os
import omegaconf
import numpy as np
from pathlib import Path
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from transformers import DecisionTransformerConfig
from . import get_model_class, get_agent_class, AGENT_CLASSES
from ..buffers import make_buffer_class


def make_agent(config, env, logdir):
    state_dim = get_obs_shape(env.observation_space)[0]
    act_dim = orig_act_dim = get_action_dim(env.action_space)
    agent_params_dict = omegaconf.OmegaConf.to_container(config.agent_params, resolve=True, throw_on_missing=True)
    agent_kind = agent_params_dict.pop("kind")
    agent_load_path = agent_params_dict.pop("load_path", None)
    agent_load_path = Path(agent_load_path["dir_path"]) / agent_load_path["file_name"] \
        if isinstance(agent_load_path, dict) else agent_load_path
    if agent_kind in AGENT_CLASSES.keys():
        if agent_kind in ["MDDT", "DDT", "MDDXLSTM", "MDDMamba"]: 
            # https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
            import torch.multiprocessing
            torch.multiprocessing.set_sharing_strategy('file_system')
        if config.get("ddp", False): 
            import torch.multiprocessing
            torch.multiprocessing.set_start_method('spawn', force=True)
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(f"cuda:{local_rank}")
        
        # prespecified state/action dims in case of mixed spaces
        max_state_dim, max_act_dim = config.agent_params.replay_buffer_kwargs.get("max_state_dim", None), \
            config.agent_params.replay_buffer_kwargs.get("max_act_dim", None)
        if max_state_dim is not None: 
            state_dim = max_state_dim
        if max_act_dim is not None:
            act_dim = max_act_dim
            
        # state/action projections for randomization
        s_proj_dim, a_proj_dim = config.agent_params.get("s_proj_dim", None), \
            config.agent_params.get("a_proj_dim", None)        
        if s_proj_dim is not None:
            state_dim = s_proj_dim
        if a_proj_dim is not None:
            act_dim = a_proj_dim
        
        # huggingface specific params
        agent_huggingface_params = agent_params_dict.pop("huggingface")
        config_class = DecisionTransformerConfig
        if "mamba" in agent_kind.lower():
            from .models.decision_mamba import MambaConfig
            config_class = MambaConfig
        elif "xlstm" in agent_kind.lower(): 
            from .models.decision_xlstm import xLSTMConfig
            config_class = xLSTMConfig
        dt_config = config_class(
            state_dim=state_dim,
            act_dim=act_dim,
            **agent_huggingface_params
        )

        # model specific params
        model_kwargs = agent_params_dict.pop("model_kwargs", {})
        if max_act_dim is not None:
            model_kwargs["max_act_dim"] = max_act_dim
        if a_proj_dim is not None: 
            model_kwargs["orig_act_dim"] = orig_act_dim
            
        # exploration specific params
        action_noise = make_action_noise(act_dim, agent_params_dict)

        # replay buffer class
        replay_buffer_class = make_buffer_class(agent_params_dict["replay_buffer_kwargs"].pop("kind", "default"))
        
        # compose additional agent kwargs
        target_return = config.env_params.target_return
        reward_scale = config.env_params.reward_scale
        add_agent_kwargs = {
            "device": config.device,
            "seed": config.seed,
            "action_noise": action_noise,
            "load_path": agent_load_path,
            "replay_buffer_class": replay_buffer_class,
            "ddp": config.get("ddp", False),
            "tensorboard_log": logdir if config.use_wandb else None,
            "target_return": target_return / reward_scale if isinstance(reward_scale, (int, float)) \
                and isinstance(target_return, (int, float)) else target_return,
            "reward_scale": reward_scale if isinstance(reward_scale, (int, float)) else dict(reward_scale),
        }
        
        # make DT model
        policy = get_model_class(agent_kind)(
            dt_config, env.observation_space, env.action_space,
            stochastic_policy=agent_params_dict["stochastic_policy"],
            **model_kwargs
        )
                
        # make DT agent
        agent = get_agent_class(agent_kind)(
            policy,
            env,
            **add_agent_kwargs,
            **agent_params_dict
        )
    elif agent_kind in ["SAC"]:
        from stable_baselines3 import SAC
        policy, policy_kwargs = agent_params_dict.pop("policy"), agent_params_dict.pop("policy_kwargs", {})
        extra_encoder = agent_params_dict.pop("extra_encoder")
        share_features_extractor = agent_params_dict.pop("share_features_extractor")
        features_extractor_arch = agent_params_dict.pop("features_extractor_arch")
        if extra_encoder:
            from src.algos.models.extractors import FlattenExtractorWithMLP
            policy_kwargs.update({"features_extractor_class": FlattenExtractorWithMLP,
                                  "share_features_extractor": share_features_extractor,
                                  "features_extractor_kwargs": {"net_arch": features_extractor_arch}})
        agent = SAC(policy=policy,
                            env=env,
                            device=config.device,
                            seed=config.seed,
                            tensorboard_log=logdir if config.use_wandb else None,
                            verbose=1,
                            policy_kwargs=policy_kwargs,
                            **agent_params_dict)
        print(agent.policy)
    elif agent_kind == "TD3":
        from stable_baselines3 import TD3
        policy = agent_params_dict.pop("policy")
        agent = TD3(policy=policy,
                    env=env,
                    device=config.device,
                    seed=config.seed,
                    tensorboard_log=logdir if config.use_wandb else None,
                    verbose=1,
                    action_noise=NormalActionNoise(mean=np.zeros(act_dim), sigma=0.1 * np.ones(act_dim)),
                    **agent_params_dict)
        print(agent.policy)
    elif agent_kind in ["PPO", "DQN"]:
        policy = agent_params_dict.pop("policy")
        if agent_kind == "PPO": 
            from .ppo_with_buffer import PPOWithBuffer
            agent_class = PPOWithBuffer
        elif agent_kind == "DQN":
            from stable_baselines3 import DQN
            agent_class = DQN
        agent = agent_class(policy=policy,
                            env=env,
                            device=config.device,
                            seed=config.seed,
                            tensorboard_log=logdir if config.use_wandb else None,
                            verbose=1,
                            **agent_params_dict)
        print(agent.policy)
    else:
        raise NotImplementedError
    return agent


def make_action_noise(act_dim, agent_params_dict):
    action_noise_std = agent_params_dict.pop("action_noise_std", None)
    ou_noise = agent_params_dict.pop("ou_noise", False)
    if ou_noise:
        return OrnsteinUhlenbeckActionNoise(mean=np.zeros(act_dim),
                                            sigma=action_noise_std * np.ones(act_dim)) \
            if action_noise_std is not None else None
    else:
        return NormalActionNoise(mean=np.zeros(act_dim), sigma=action_noise_std * np.ones(act_dim)) \
            if action_noise_std is not None else None
