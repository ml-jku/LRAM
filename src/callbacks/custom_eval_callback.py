import os
import copy
import functools
import scipy.stats
import numpy as np
import torch
import wandb
from collections import defaultdict
from joblib import delayed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization, DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import HumanOutputFormat
from .evaluation import custom_evaluate_policy
from ..envs.env_utils import extract_env_name
from ..envs.hn_scores import get_human_normalized_score, ENVID_TO_HNS
from ..envs.dn_scores import get_data_normalized_score, ENVID_TO_DNS
from ..envs.env_names import PROCGEN_ENVS, ID_TO_DOMAIN
from ..utils.misc import gather_dict, ProgressParallel, with_initializer, get_gpu_ram_stats


def _init_worker(model):    
    # avoids unnecessary imports in worker processes - done once per worker
    global _worker_model
    _worker_model = model
    if hasattr(model.policy, "reinit_cuda_kernels"):
        if hasattr(model, "ddp") and model.ddp:
            torch.cuda.set_device(model.device)
        torch.cuda.empty_cache()
        model.policy.reinit_cuda_kernels()
    if model.compile:
        # compile models are not picklable, therefore need to recompile
        _worker_model.policy = torch.compile(_worker_model.policy, dynamic=False if model.ddp else None)


class CustomEvalCallback(EvalCallback):
    """
    We just want to swap evaluate_policy to custom_evalutate_policy, which works with DT.

    """
    def __init__(self, eval_env, first_step=True, log_eval_trj=False, prefix="eval", use_wandb=False, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.first_step = first_step
        self.log_eval_trj = log_eval_trj
        self.prefix = prefix
        # for some reason, steps in wandb need to be monotonically increasing: if we want to log "earlier" steps
        # define custom metric: https://github.com/wandb/wandb/issues/6554
        self.use_wandb = use_wandb
        self.wandb_is_defined = False
        if self.use_wandb and wandb.run is not None: 
            self.step_metric = f"{prefix}_step"
            wandb.define_metric(self.step_metric)

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []

            with torch.no_grad():
                episode_rewards, episode_lengths = custom_evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=self.warn,
                    callback=self._log_success_callback,
                )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record(f"{self.prefix}/mean_reward", float(mean_reward))
            self.logger.record(f"{self.prefix}/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record(f"{self.prefix}/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)
            
            # log human normalized scores
            env_name = extract_env_name(self.eval_env)
            if env_name in ENVID_TO_HNS: 
                self._log_normalized_scores(env_name, np.array(episode_rewards), "hns")
            elif env_name in ENVID_TO_DNS:
                self._log_normalized_scores(env_name, np.array(episode_rewards), "dns")

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()
            
            if self.log_eval_trj: 
                self._log_eval_trj(episode_rewards)
                
            self.wandb_is_defined = False
            if hasattr(self.model, "ddp") and self.model.ddp:
                torch.distributed.barrier()
        return continue_training

    def _on_training_start(self):
        if self.first_step:
            # Do an initial evaluation before training
            print("Initial evaluation...")
            self._on_step()
    
    def _log_normalized_scores(self, env_name, episode_rewards, score_type="hns"):
        score_fn = get_human_normalized_score if score_type == "hns" else get_data_normalized_score
        score = np.mean(score_fn(env_name, episode_rewards))
        if self.verbose > 0:
            print(f"{score_type}: {score:.2f}%")
        self.logger.record(f"{self.prefix}/{env_name}/{score_type}", score)
        
    def associate_wandb_metrics(self, metrics):
        # should only be done the very first time
        for metric in metrics:
            wandb.define_metric(metric, step_metric=self.step_metric)
            
    def _log_eval_trj(self, vals, key="reward_trj"):
        for i, ep_reward in enumerate(vals):
            if self.use_wandb and wandb.run is not None: 
                if not self.wandb_is_defined:
                    self.associate_wandb_metrics([f"{self.prefix}/{key}"])
                wandb.log({
                    f"{self.prefix}/{key}": ep_reward, self.step_metric: self.num_timesteps + i
                })
            else: 
                self.logger.record(f"{self.prefix}/{key}", ep_reward, exclude="stdout")
                self.logger.dump(self.num_timesteps + i)
        
                
class MultiEnvEvalCallback(EvalCallback):

    def __init__(self, eval_env, first_step=True, log_eval_trj=False, prefix="eval", 
                 use_wandb=False, n_jobs=0, track_gpu_stats=False, **kwargs):
        super().__init__(eval_env, **kwargs)
        # dicts for keeping track of performance immediately after task was trained on
        # and performance after all tasks for measuring forgetting
        self.task_to_last_scores = defaultdict(float)
        self.task_to_task_scores = defaultdict(float)
        self.first_step = first_step
        self.log_eval_trj = log_eval_trj
        self.prefix = prefix
        self.use_wandb = use_wandb
        self.n_jobs = n_jobs
        self.track_gpu_stats = track_gpu_stats
        self.wandb_is_defined = False  
        if self.use_wandb and wandb.run is not None: 
            self.step_metric = f"{prefix}_step"
            wandb.define_metric(self.step_metric)
        self._is_success_buffer = []

    def _init_callback(self):
        super()._init_callback()
        # configure Logger to display more than 36 characters --> this kills runs due to duplicate keys
        # increase max_length for now.
        for format in self.logger.output_formats:
            if isinstance(format, HumanOutputFormat):
                format.max_length = 96

    def _on_step(self) -> bool:
        continue_training = True
        eval_fn = custom_evaluate_policy
        is_ddp = hasattr(self.model, "ddp") and self.model.ddp and len(self.eval_env.envs) > 1
        is_rank0, world_size = False, None
        if is_ddp: 
            is_rank0, world_size = self.model.global_rank == 0, int(os.environ["WORLD_SIZE"]),
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # perform evaluation + individual env logging
            all_avg_stats, trj_stats_per_env, stats_per_domain, trj_stats_per_domain, ddp_store = self._on_step_eval(
                eval_fn, is_ddp=is_ddp, is_rank0=is_rank0, world_size=world_size
            )
                
            # handle distributed logging
            if is_ddp:
                out = self.gather_and_record_logs(
                    is_rank0, world_size, ddp_store, all_avg_stats, stats_per_domain, 
                    trj_stats_per_env, trj_stats_per_domain
                )
                if is_rank0:
                    ddp_store, all_avg_stats, stats_per_domain, trj_stats_per_env, trj_stats_per_domain = out
            self._log_forgetting_scores()

            # log aggregate scores
            if len(all_avg_stats.get("avg_successes", [])) > 1:
                avg_success_rate = np.mean(all_avg_stats["avg_successes"])
                iqm_success_rate = scipy.stats.trim_mean(all_avg_stats["avg_successes"], proportiontocut=0.25, axis=None)
                if self.verbose > 0:
                    print(f"Success rate: {100 * avg_success_rate:.2f}%")
                self.logger.record(f"{self.prefix}/avg_success_rate", avg_success_rate)
                self.logger.record(f"{self.prefix}/iqm_success_rate", iqm_success_rate)
                self.logger.dump(self.num_timesteps)
            if len(all_avg_stats["avg_rewards"]) > 1:
                self.logger.record(f"{self.prefix}/avg_rewards", np.mean(all_avg_stats["avg_rewards"]))
                self.logger.dump(self.num_timesteps)
            if len(all_avg_stats.get("avg_hns", [])) > 1:
                iqm_hns = scipy.stats.trim_mean(all_avg_stats["avg_hns"], proportiontocut=0.25, axis=None)
                self.logger.record(f"{self.prefix}/avg_hns", np.mean(all_avg_stats["avg_hns"]))
                self.logger.record(f"{self.prefix}/iqm_hns", iqm_hns)
                self.logger.dump(self.num_timesteps)
            if len(all_avg_stats.get("avg_dns", [])) > 1:
                iqm_dns = scipy.stats.trim_mean(all_avg_stats["avg_dns"], proportiontocut=0.25, axis=None)
                self.logger.record(f"{self.prefix}/avg_dns", np.mean(all_avg_stats["avg_dns"]))
                self.logger.record(f"{self.prefix}/iqm_dns", iqm_dns)
                self.logger.dump(self.num_timesteps)
            if len(stats_per_domain.keys()) > 1:
                metric_avgs = defaultdict(list)
                for domain, stats in stats_per_domain.items():
                    for key, values in stats.items():
                        avg = np.mean(values)
                        self.logger.record(f"{self.prefix}/{domain}/{key}", avg)
                        metric_avgs[key].append(avg)
                for key, values in metric_avgs.items():
                    self.logger.record(f"{self.prefix}/all/{key}", np.mean(values))
                self.logger.dump(self.num_timesteps)

            if self.log_eval_trj:
                avg_ep_reward_trj = np.stack(list(trj_stats_per_env["rewards"].values())).mean(0)
                all_key_vals = trj_stats_per_env["rewards"]
                if len(self._is_success_buffer) > 0:
                    trj_stats_per_env["successes"][f"{self.prefix}/avg_success_trj"] = np.stack(list(trj_stats_per_env["successes"].values())).mean(0)
                    all_key_vals = {**all_key_vals, **trj_stats_per_env["successes"]}
                if len(all_avg_stats.get("avg_gns", [])) > 0:
                    trj_stats_per_env["hns"][f"{self.prefix}/avg_hns_trj"] = np.stack(list(trj_stats_per_env["hns"].values())).mean(0)
                    all_key_vals = {**all_key_vals, **trj_stats_per_env["hns"]}
                if len(all_avg_stats.get("avg_dns", [])) > 0:
                    trj_stats_per_env["dns"][f"{self.prefix}/avg_dns_trj"] = np.stack(list(trj_stats_per_env["dns"].values())).mean(0)
                    all_key_vals = {**all_key_vals, **trj_stats_per_env["dns"]}
                for domain, trj_stats in trj_stats_per_domain.items():
                    for key, values in trj_stats.items():
                        all_key_vals = {**all_key_vals, **{f"{self.prefix}/{domain}/{key}": np.stack(values).mean(0)}}
                self._log_eval_trj(avg_ep_reward_trj, all_key_vals)
                
            self.wandb_is_defined = True
        if hasattr(self.model, "ddp") and self.model.ddp:
            torch.distributed.barrier()
        return continue_training

    def _on_step_eval(self, eval_fn, is_ddp=False, is_rank0=False, world_size=None): 
        ddp_store, all_stats = {}, []
        all_avg_stats, trj_stats_per_env = defaultdict(list), defaultdict(lambda: defaultdict(list))
        stats_per_domain, trj_stats_per_domain = defaultdict(lambda: defaultdict(list)), \
            defaultdict(lambda: defaultdict(list))
        num_envs = len(self.eval_env.envs)
        common_kwargs = {"eval_fn": eval_fn, "is_ddp": is_ddp, "world_size": world_size, "eval_env": None,
                         "n_eval_episodes": self.n_eval_episodes, "render": self.render, 
                         "deterministic": self.deterministic, "prefix": self.prefix}
        if self.n_jobs > 1 and num_envs > 1:
            from joblib.externals.loky import get_reusable_executor 
            # parallel eval
            all_results = [None] * num_envs
            # procgen envs are not picklable, remove from eval env, evaluate separately
            picklable_eval_env = self.eval_env
            is_procgen = [extract_env_name(env) in PROCGEN_ENVS for env in self.eval_env.envs] 
            if any(is_procgen):
                picklable_eval_env = copy.copy(self.eval_env)
                picklable_eval_env.envs = copy.copy(self.eval_env.envs)
                for idx, env in enumerate(self.eval_env.envs):
                    if is_procgen[idx]:
                        picklable_eval_env.envs[idx] = None
            
            # make picklabale model
            pickleable_model = self.make_pickleable_model(self.model)
            
            # run parallel eval
            fn = functools.partial(self._run_evaluation, model=None, env=None, log_success=False,
                                   verbose=True, **common_kwargs)
            initializer = functools.partial(_init_worker, model=pickleable_model)    
            if not all(is_procgen):
                all_results = with_initializer(ProgressParallel(n_jobs=self.n_jobs, total=num_envs, timeout=3000), initializer)(
                    delayed(fn)(idx=idx, env=env) for idx, env in enumerate(picklable_eval_env.envs)
                )
                # terminate processes: https://github.com/joblib/joblib/issues/945
                get_reusable_executor().shutdown(wait=True)
                global _worker_env, _worker_model
                _worker_env, _worker_model = None, None
            if any(is_procgen): 
                # evaluate procgen envs sequentially
                for idx, env in enumerate(self.eval_env.envs):
                    if is_procgen[idx]:
                        common_kwargs["eval_env"] = self.eval_env
                        all_results[idx] = self._run_evaluation(
                            model=self.model, env=env, idx=idx, log_success=False, verbose=True, **common_kwargs
                        )
            # collect results and log 
            episode_rewards, episode_lengths, is_success_buffer, episode_times = zip(*all_results)
            for idx, env in enumerate(self.eval_env.envs):
                _stats = self._on_step_logging(
                    self.model, self.logger, env, idx, episode_rewards[idx], episode_lengths[idx], is_success_buffer[idx],
                    num_envs, world_size, is_ddp=is_ddp, is_rank0=is_rank0, episode_times=episode_times[idx]
                )
                all_stats.append(_stats)
                self.is_success_buffer = is_success_buffer 
        else:
            # sequential eval
            for idx, env in enumerate(self.eval_env.envs):
                # evaluate
                common_kwargs["eval_env"] = self.eval_env
                episode_rewards, episode_lengths, is_success_buffer, episode_times = self._run_evaluation(
                    self.model, env, idx, **common_kwargs
                )
                # log
                _stats = self._on_step_logging(
                    self.model, self.logger, env, idx, episode_rewards, episode_lengths, is_success_buffer,
                    num_envs, world_size, is_ddp=is_ddp, is_rank0=is_rank0, episode_times=episode_times
                )
                all_stats.append(_stats)
                self.is_success_buffer = is_success_buffer
        
        # extract stats of individual envs and aggregate
        _all_avg_stats, _trj_stats_per_env, _stats_per_domain, _trj_stats_per_domain, _ddp_store = zip(*all_stats)
        for i in range(num_envs):
            for k, v in _all_avg_stats[i].items():
                all_avg_stats[k].extend(v)
            for domain, stats in _stats_per_domain[i].items():
                for k, v in stats.items():
                    stats_per_domain[domain][k].extend(v)
            for domain, stats in _trj_stats_per_domain[i].items():
                for k, v in stats.items():
                    trj_stats_per_domain[domain][k].extend(v)
            for k, v in _trj_stats_per_env[i].items():
                trj_stats_per_env[k].update(v)
            ddp_store = {**ddp_store, **_ddp_store[i]}
        return all_avg_stats, trj_stats_per_env, stats_per_domain, trj_stats_per_domain, ddp_store
    
    @staticmethod
    def _run_evaluation(model, env, idx, eval_fn, n_eval_episodes, render=False, deterministic=True, log_success=True,
                        is_ddp=False, verbose=False, world_size=None, eval_env=None, prefix="eval"):
        if model is None:
            model = _worker_model
        if env is None or (is_ddp and idx % world_size != model.global_rank): 
            return None, None, None, None
        is_success_buffer = []
        def log_success_callback(locals_, globals_) -> None:
            """
            In contrast, to the class method, this uses a global buffer, 
            which is compatible with pickling. 
            """
            info = locals_["info"]
            if isinstance(info, list):
                info = info[0]
            if locals_["done"]:
                maybe_is_success = info.get("is_success") or info.get("success")
                if maybe_is_success is not None:
                    is_success_buffer.append(maybe_is_success)                
        
        env_name = extract_env_name(env, idx)
        if verbose: 
            print(f"Evaluating idx={idx}, env_name={env_name}.")
        
        if not isinstance(env, (DummyVecEnv, SubprocVecEnv)) and env_name not in PROCGEN_ENVS:
            env = DummyVecEnv([lambda: env])
        if eval_env is not None and isinstance(eval_env, VecVideoRecorder):
            # wrap the env with same recorder
            env = VecVideoRecorder(env, video_folder=eval_env.video_folder, 
                                   record_video_trigger=eval_env.record_video_trigger,
                                   video_length=eval_env.video_length, 
                                   name_prefix=f"{prefix}_{env_name}")
            env.name = env_name
        
        # Reset success rate buffer
        with torch.no_grad():
            eval_out = eval_fn(
                model,
                env,
                n_eval_episodes=n_eval_episodes,
                render=render,
                deterministic=deterministic,
                return_episode_rewards=True,
                callback=log_success_callback if log_success else None,
                task_id=idx
            )
            episode_rewards, episode_lengths = eval_out[:2]
            episode_times = None if len(eval_out) < 3 else eval_out[2]
        if hasattr(env, "video_recorder") and env.video_recorder is not None:
            env.close_video_recorder()
            del env.video_recorder
        if verbose: 
            print(f"Done evaluating idx={idx}, env_name={env_name}.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return episode_rewards, episode_lengths, is_success_buffer, episode_times
    
    def _on_step_logging(self, model, logger, env, idx, episode_rewards, episode_lengths, is_success_buffer, 
                         num_envs, world_size, is_ddp=False, is_rank0=False, episode_times=None):
        ddp_store = {}
        all_avg_stats, trj_stats_per_env = defaultdict(list), defaultdict(lambda: defaultdict(list))
        stats_per_domain, trj_stats_per_domain = defaultdict(lambda: defaultdict(list)), \
            defaultdict(lambda: defaultdict(list))
        if is_ddp and idx % world_size != model.global_rank: 
            return all_avg_stats, trj_stats_per_env, stats_per_domain, trj_stats_per_domain, ddp_store
        env_name = extract_env_name(env, idx)
        env_domain = ID_TO_DOMAIN.get(env_name, "other")
        env_id = f"{env_name}_{idx}"

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        self.last_mean_reward = mean_reward

        if self.verbose > 0:
            print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        # Add to current Logger
        logger.record(f"{self.prefix}/{env_id}/mean_reward", float(mean_reward))
        logger.record(f"{self.prefix}/{env_id}/mean_ep_length", mean_ep_length)
        if self.track_gpu_stats:
            # this only makes sense in a single env setting.
            if self.model.device != "cpu":
                gpu_stats = get_gpu_ram_stats(self.model.device)
                for k, v in gpu_stats.items():
                    logger.record(f"{self.prefix}/{env_id}/{k}", v)
        
        if episode_times is not None:
            mean_ep_time = np.mean(episode_times)
            logger.record(f"{self.prefix}/{env_id}/mean_ep_time", mean_ep_time)
            logger.record(f"{self.prefix}/{env_id}/time_per_step", mean_ep_time / (mean_ep_length + 1e-8))
            logger.record(f"{self.prefix}/{env_id}/steps_per_second", mean_ep_length / (mean_ep_time + 1e-8))
            if hasattr(self.model.policy, "inf_dummy_batch_size") and self.model.policy.inf_dummy_batch_size is not None: 
                logger.record(f"{self.prefix}/{env_id}/total_steps_per_second", 
                              (mean_ep_length * self.model.policy.inf_dummy_batch_size) / (mean_ep_time + 1e-8))        
        if num_envs == 1:
            # redundant logging for easier aggregation
            logger.record(f"{self.prefix}/mean_reward", float(mean_reward))
            logger.record(f"{self.prefix}/mean_ep_length", mean_ep_length)
        elif idx == model.current_task_id:
            logger.record(f"{self.prefix}/cur_task_mean_reward", float(mean_reward))
        all_avg_stats["avg_rewards"].append(float(mean_reward))
        stats_per_domain[env_domain]["avg_rewards"].append(float(mean_reward))

        # log success rates
        if len(is_success_buffer) > 0:
            success_rate = np.mean(is_success_buffer)
            all_avg_stats["avg_successes"].append(success_rate)
            stats_per_domain[env_domain]["success_rates"].append(success_rate)
            self._log_success_rates(logger, env_id, success_rate, current_task_id=model.current_task_id,
                                    num_envs=num_envs, idx=idx)

        # log human normalized scores
        if env_name in ENVID_TO_HNS: 
            all_hns = get_human_normalized_score(env_name, np.array(episode_rewards))
            hns = np.mean(all_hns)
            all_avg_stats["avg_hns"].append(hns)
            stats_per_domain[env_domain]["avg_hns"].append(hns)
            self._log_normalized_scores(logger, env_id, hns, num_envs=num_envs, idx=idx,
                                        current_task_id=model.current_task_id, score_type="hns")
        if env_name in ENVID_TO_DNS: 
            all_dns = get_data_normalized_score(env_name, np.array(episode_rewards))
            dns = np.mean(all_dns)
            all_avg_stats["avg_dns"].append(dns)
            stats_per_domain[env_domain]["avg_dns"].append(dns)
            self._log_normalized_scores(logger, env_id, dns, num_envs=num_envs, idx=idx,
                                        current_task_id=model.current_task_id, score_type="dns")

        # Dump log so the evaluation results are printed with the correct timestep
        logger.record(f"time/{env_id}/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if is_ddp and not is_rank0:
            for k, v in logger.name_to_value.items():
                ddp_store[k] = v
        logger.dump(self.num_timesteps)
        
        if self.log_eval_trj: 
            trj_stats_per_env["rewards"][f"{self.prefix}/{env_id}/reward_trj"] = episode_rewards
            if len(is_success_buffer) > 0:
                trj_stats_per_env["succcesses"][f"{self.prefix}/{env_id}/success_trj"] = is_success_buffer
                trj_stats_per_domain[env_domain]["success_trj"].append(is_success_buffer)
            if env_name in ENVID_TO_HNS: 
                trj_stats_per_env["hns"][f"{self.prefix}/{env_id}/hns_trj"] = all_hns
                trj_stats_per_domain[env_domain]["hns_trj"].append(all_hns)
            if env_name in ENVID_TO_DNS: 
                trj_stats_per_env["dns"][f"{self.prefix}/{env_id}/dns_trj"] = all_dns
                trj_stats_per_domain[env_domain]["dns_trj"].append(all_dns)
            
        if mean_reward > self.best_mean_reward:
            if self.verbose > 0:
                print("New best mean reward!")
            if self.best_model_save_path is not None:
                model.save(os.path.join(self.best_model_save_path, "best_model"))
            self.best_mean_reward = mean_reward
            # Trigger callback on new best model, if needed
            if self.callback_on_new_best is not None:
                continue_training = self.callback_on_new_best.on_step()

        # Trigger callback after every evaluation, if needed
        if self.callback is not None:
            continue_training = continue_training and self._on_event()
        
        return all_avg_stats, trj_stats_per_env, stats_per_domain, trj_stats_per_domain, ddp_store

    def _on_training_end(self) -> None:
        self._log_forgetting_scores()
    
    def _on_training_start(self):
        if self.first_step:
            # Do an initial evaluation before training
            print("Initial evaluation...")
            self._on_step()

    def _log_forgetting_scores(self):
        if len(self.task_to_task_scores) > 1 and len(self.task_to_last_scores) > 1:
            forgetting_scores = []
            for task_id, task_score in self.task_to_task_scores.items():
                forgetting = task_score - self.task_to_last_scores[task_id]
                self.logger.record(f"{self.prefix}/{task_id}/forgetting", float(forgetting))
                forgetting_scores.append(forgetting)
            self.logger.record(f"{self.prefix}/forgetting", np.mean(forgetting_scores))
            self.logger.dump(self.num_timesteps)
            
    def _log_success_rates(self, logger, env_id, success_rate, current_task_id, num_envs=1, idx=0):
        if self.verbose > 0:
            print(f"Success rate: {100 * success_rate:.2f}%")
        logger.record(f"{self.prefix}/{env_id}/success_rate", success_rate)
        if num_envs == 1:
            # redundant logging for easier aggregation
            logger.record(f"{self.prefix}/success_rate", success_rate)
        else:
            self.task_to_last_scores[env_id] = success_rate
            if idx == current_task_id:
                logger.record(f"{self.prefix}/cur_task_success_rate", success_rate)
                self.task_to_task_scores[env_id] = success_rate

    def _log_normalized_scores(self, logger, env_id, score, num_envs, current_task_id, idx=0, score_type="hns"):
        if self.verbose > 0:
            print(f"{score_type}: {score:.2f}%")
        logger.record(f"{self.prefix}/{env_id}/{score_type}", score)
        if num_envs == 1:
            # redundant logging for easier aggregation
            logger.record(f"{self.prefix}/{score_type}", score)
        else:
            self.task_to_last_scores[env_id] = score
            if idx == current_task_id:
                logger.record(f"{self.prefix}/cur_task_{score_type}", score)
                self.task_to_task_scores[env_id] = score
        return score
    
    def _log_eval_trj(self, vals, key_val_dict, key="avg_reward_trj"):
        # dump
        self.logger.dump(self.num_timesteps)
        # for some tensorboard-specific reason, all trj rewards need to be written together. 
        for ep_idx, ep_reward in enumerate(vals):
            if self.use_wandb and wandb.run is not None: 
                # use different "step_metric" for tracking eval trj (wandb issues)
                if not self.wandb_is_defined:
                    self.associate_wandb_metrics(
                        [f"{self.prefix}/{key}"] + [k for k in key_val_dict.keys()]
                    )
                wandb.log({
                    f"{self.prefix}/{key}": ep_reward, 
                    **{k: v[ep_idx] for k, v in key_val_dict.items()},
                    self.step_metric: self.num_timesteps + ep_idx
                })
            else: 
                self.logger.record(f"{self.prefix}/{key}", ep_reward, exclude="stdout")
                # record individual env episode rewards
                for k, v in key_val_dict.items():
                    self.logger.record(k, v[ep_idx], exclude="stdout")
                self.logger.dump(self.num_timesteps + ep_idx)

    def associate_wandb_metrics(self, metrics):
        # should only be done the very first time
        for metric in metrics:
            wandb.define_metric(metric, step_metric=self.step_metric)
    
    def gather_and_record_logs(self, is_rank0, world_size, ddp_store, all_avg_stats, stats_per_domain, 
                               trj_stats_per_env, trj_stats_per_domain):
        ddp_store = gather_dict(is_rank0, world_size, ddp_store)
        if is_rank0: 
            for k, v in ddp_store.items():
                self.logger.record(k, v, exclude="tensorboard" if k.startswith("time") else None)
            self.logger.dump(self.num_timesteps)
        # convert to dict to avoid pickling error with lambda
        all_avg_stats.default_factory = None
        trj_stats_per_env.default_factory = None
        stats_per_domain.default_factory = None
        trj_stats_per_domain.default_factory = None
        all_avg_stats = gather_dict(is_rank0, world_size, all_avg_stats)
        trj_stats_per_env = gather_dict(is_rank0, world_size, trj_stats_per_env)
        stats_per_domain = gather_dict(is_rank0, world_size, stats_per_domain)
        trj_stats_per_domain = gather_dict(is_rank0, world_size, trj_stats_per_domain)
        return ddp_store, all_avg_stats, stats_per_domain, trj_stats_per_env, trj_stats_per_domain

    def make_pickleable_model(self, model): 
        pickleable_model = copy.copy(model)
        pickleable_model._logger = None
        pickleable_model.env = None
        pickleable_model.eval_env = None
        del pickleable_model.schedulers
        del pickleable_model.exploration_schedule
        del pickleable_model.lr_schedule
        del pickleable_model.target_return_dict
        del pickleable_model.optimizer
        del pickleable_model.loss_fn
        # keep replay buffer, but reset
        pickleable_model.replay_buffer = copy.copy(model.replay_buffer)
        pickleable_model.replay_buffer.reset(0, verbose=False)
        policy = self.model.policy
        if self.model.compile:
            # make exception for compiled model, use original - compiled models are not serializable
            policy = policy._orig_mod
        if self.model.ddp: 
            policy = policy.module
        pickleable_model.policy = copy.deepcopy(policy)
        pickleable_model.policy_class = None
        if hasattr(pickleable_model.policy, "make_pickleable"):
            pickleable_model.policy.make_pickleable()
        # validate if all parameters are matching
        new_params = {k: v for k, v in pickleable_model.policy.encoder.named_parameters()}
        for k, v in model.policy.encoder.named_parameters():
            assert torch.allclose(v, new_params[k]), f"Mismatch in encoder parameters for {k}"
        return pickleable_model
