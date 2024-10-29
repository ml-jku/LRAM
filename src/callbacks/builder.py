import os
import functools
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList
from ..algos import AGENT_CLASSES
from ..envs.env_names import ID_TO_NAMES, MT50_ENVS_v2, ATARI_ENVS, DM_CONTROL_ENVS, \
    MINIHACK_ENVS, GYM_ENVS, PROCGEN_ENVS, COMPOSUITE_ENVS, MIMICGEN_ENVS


class CustomWandbCallback(WandbCallback):

    def __init__(self, model_sync_wandb=False, **kwargs):
        super().__init__(**kwargs)
        self.model_sync_wandb = model_sync_wandb

    def save_model(self) -> None:
        print(f"Saving model checkpoint to {self.path}")
        self.model.save(self.path)
        if self.model_sync_wandb:
            wandb.save(self.path, base_path=self.model_save_path)


def make_callbacks(config, env=None, eval_env=None, logdir=None, train_eval_env=None):
    callbacks = []
    if config.use_wandb and logdir is not None:
        model_save_path = None
        if config.wandb_callback_params.model_save_path is not None:
            model_save_path = f"{logdir}/{config.wandb_callback_params.model_save_path}"
        if config.get("ddp", False):
            global_rank = int(os.environ["RANK"])
            model_save_path = model_save_path if global_rank == 0 else None
        callbacks.append(
            CustomWandbCallback(
                gradient_save_freq=config.wandb_callback_params.gradient_save_freq,
                verbose=config.wandb_callback_params.verbose,
                model_save_path=model_save_path,
                model_sync_wandb=config.wandb_callback_params.get("model_sync_wandb", False),
                model_save_freq=config.wandb_callback_params.get("model_save_freq", 0)
            )
        )
    if config.eval_params.use_eval_callback:
        if config.agent_params.kind in AGENT_CLASSES.keys():
            from .custom_eval_callback import CustomEvalCallback, MultiEnvEvalCallback
            if config.env_params.envid not in [*list(ID_TO_NAMES.keys()), *ATARI_ENVS, *MT50_ENVS_v2,
                                               *DM_CONTROL_ENVS, *MINIHACK_ENVS, *GYM_ENVS,
                                               *PROCGEN_ENVS, *COMPOSUITE_ENVS, *MIMICGEN_ENVS,
                                               "Dummy-v1"]:
                eval_callback_class = functools.partial(CustomEvalCallback, use_wandb=config.use_wandb)
            else:
                eval_callback_class = functools.partial(MultiEnvEvalCallback, use_wandb=config.use_wandb)
        else:
            from stable_baselines3.common.callbacks import EvalCallback
            eval_callback_class = EvalCallback
        if config.eval_params.max_no_improvement_evals > 0:
            from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
            stop_training_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=config.eval_params.max_no_improvement_evals, verbose=1)
        else:
            stop_training_callback = None
        eval_callback_kwargs = {
            "n_eval_episodes": config.eval_params.n_eval_episodes, "eval_freq": config.eval_params.eval_freq,
            "callback_after_eval": stop_training_callback, "deterministic": config.eval_params.deterministic,
            "first_step": config.eval_params.get("first_step", True),
            "log_eval_trj": config.eval_params.get("log_eval_trj", False),
            "n_jobs": config.eval_params.get("n_jobs", 0)
        }
        if config.eval_params.get("track_gpu_stats", False):
            eval_callback_kwargs["track_gpu_stats"] = config.eval_params.track_gpu_stats
        if config.eval_params.get("eval_on_train", False):
            train_eval_callback = eval_callback_class(eval_env=env, prefix="train_eval", **eval_callback_kwargs)
            callbacks.append(train_eval_callback)
        if train_eval_env is not None:
            train_eval_seeds_callback = eval_callback_class(eval_env=train_eval_env, prefix="train_eval_seeds", 
                                                            **eval_callback_kwargs)
            callbacks.append(train_eval_seeds_callback)
        
        eval_callback = eval_callback_class(eval_env=eval_env, **eval_callback_kwargs)
        callbacks.append(eval_callback)
    if hasattr(config.eval_params, "use_valid_callback") and config.eval_params.use_valid_callback: 
        from .validation_callback import ValidationCallback
        valid_callback_kwargs = {
            "eval_freq": config.eval_params.eval_freq,
            "first_step": config.eval_params.get("first_step", True),
            **config.eval_params.get("valid_kwargs", {})
        }
        valid_callback = ValidationCallback(**valid_callback_kwargs)
        callbacks.append(valid_callback)
    return CallbackList(callbacks)
