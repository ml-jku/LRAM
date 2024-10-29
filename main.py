import os
import traceback
import hydra
import wandb
import omegaconf
from datetime import timedelta
from pathlib import Path
from torch.distributed import init_process_group, destroy_process_group
from src.utils import maybe_split, safe_mean, multiply


def setup_logging(config):
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    config_dict["PID"] = os.getpid()
    print(f"PID: {os.getpid()}")
    # hydra changes working directories automatically
    logdir = str(Path.joinpath(Path(os.getcwd()), config.logdir))
    Path(logdir).mkdir(exist_ok=True, parents=True)
    print(f"Logdir: {logdir}")

    run = None
    if config.use_wandb:
        print("Setting up logging to Weights & Biases.")
        # make "wandb" path, otherwise WSL might block writing to dir
        wandb_path = Path.joinpath(Path(logdir), "wandb")
        wandb_path.mkdir(exist_ok=True, parents=True)
        wandb_params = omegaconf.OmegaConf.to_container(config.wandb_params, resolve=True, throw_on_missing=True)
        key, host = wandb_params.pop("key", None), wandb_params.pop("host", None)
        if key is not None and host is not None: 
            wandb.login(key=key, host=host)
            config.wandb_params.update({"key": None, "host": None})
        run = wandb.init(tags=[config.experiment_name],
                         config=config_dict, **wandb_params)
        print(f"Writing Weights & Biases logs to: {str(wandb_path)}")
        run.log_code(hydra.utils.get_original_cwd())
    return run, logdir


def setup_ddp():
    init_process_group(backend="nccl", timeout=timedelta(minutes=240))


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(config):
    ddp = config.get("ddp", False)
    if ddp: 
        setup_ddp()
        # make sure only global rank0 writes to wandb
        logdir = None
        global_rank = int(os.environ["RANK"])
        if global_rank == 0: 
            run, logdir = setup_logging(config)
    else: 
        run, logdir = setup_logging(config)
    print("Config: \n", omegaconf.OmegaConf.to_yaml(config, resolve=True, sort_keys=True))
    
    # imports after initializing ddp to avoid fork/spawn issues
    from src.envs import make_env
    from src.callbacks import make_callbacks
    from src.algos.builder import make_agent
    
    env, eval_env, train_eval_env = make_env(config, logdir)
    agent = make_agent(config, env, logdir)
    callbacks = make_callbacks(config, env=env, eval_env=eval_env, logdir=logdir, train_eval_env=train_eval_env)
    res, score = None, None
    try:
        res = agent.learn(
            **config.run_params,
            eval_env=eval_env,
            callback=callbacks
        )
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
    finally:
        print("Finalizing run...")
        if config.use_wandb:
            if config.env_params.record:
                env.video_recorder.close()
            if not ddp or (ddp and global_rank == 0): 
                run.finish()
                wandb.finish
        # return last avg reward for hparam optimization
        score = None if res is None else safe_mean([ep_info["r"] for ep_info in res.ep_info_buffer])
        if ddp: 
            destroy_process_group()
        if hasattr(agent, "cache"): 
            agent.cache.cleanup_cache()
    return score


if __name__ == "__main__":
    omegaconf.OmegaConf.register_new_resolver("maybe_split", maybe_split)
    omegaconf.OmegaConf.register_new_resolver("multiply", multiply)
    main()
