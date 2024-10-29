import hydra
import wandb
import omegaconf
from src.utils import maybe_split
from src.envs import make_env
from src.callbacks import make_callbacks
from src.algos.builder import make_agent
from main import setup_logging


def evaluate(agent, callbacks): 
    _, callback = agent._setup_learn(0, None, callbacks)
    agent.policy.eval()
    callback.on_training_start(locals(), globals())
    agent._dump_logs()


@hydra.main(config_path="configs", config_name="config")
def main(config):
    print("Config: \n", omegaconf.OmegaConf.to_yaml(config, resolve=True, sort_keys=True))
    config.agent_params.offline_steps = 0
    config.agent_params.data_paths = None
    run, logdir = setup_logging(config)
    env, eval_env, train_eval_env = make_env(config, logdir)
    agent = make_agent(config, env, logdir)
    callbacks = make_callbacks(config, env=env, eval_env=eval_env, logdir=logdir, train_eval_env=train_eval_env)
    try:
        # set training steps to 0 to avoid training
        agent.learn(
            total_timesteps=0,
            eval_env=eval_env,
            callback=callbacks
        )
    finally:
        print("Finalizing run...")
        if config.use_wandb:
            run.finish()
            wandb.finish()
        # return last avg reward for hparam optimization
        if hasattr(agent, "cache"): 
            agent.cache.cleanup_cache()


if __name__ == "__main__":
    omegaconf.OmegaConf.register_new_resolver("maybe_split", maybe_split)
    main()
