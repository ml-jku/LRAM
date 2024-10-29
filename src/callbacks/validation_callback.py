import copy
import collections
import torch
from tqdm import tqdm
from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common.logger import HumanOutputFormat
from ..buffers.buffer_utils import filter_top_p_trajectories


class ValidationCallback(EventCallback):
    """
    Callback to compute loss metrics on validation set every n steps. 

    """
    def __init__(
        self, 
        eval_freq=10000,
        n_batches=2000,
        first_step=True,
        prefix="valid", 
        splits=["full", "top_50", "bottom_50"],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.first_step = first_step
        self.prefix = prefix
        self.eval_freq = eval_freq
        self.n_batch = n_batches
        self.splits = splits

    def init_callback(self, model) -> None:
        super().init_callback(model)
        if self.callback is not None:
            self.callback.init_callback(self.model)
        self._setup_validation_buffer()
        # configure Logger to display more than 36 characters --> this kills runs due to duplicate keys
        # increase max_length for now.
        for format in self.logger.output_formats:
            if isinstance(format, HumanOutputFormat):
                format.max_length = 96
        
    def _setup_validation_buffer(self): 
        buffer_class = self.model.replay_buffer_class
        original_buffer = self.model.replay_buffer
        self.validation_buffer = buffer_class(
            self.model.buffer_size,
            self.model.observation_space,
            self.model.action_space,
            **self.model.replay_buffer_kwargs,
        )
        assert original_buffer._valid_trajectories is not None
        # extract validation set from original buffer
        self.validation_buffer.init_from_existing_buffer(original_buffer, validation=True)
            
    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            print("Validating...")
            self.logger.dump(self.num_timesteps)
            # compute metrics for given splits --> e.g., all, top50%, bottom50%
            self.compute_metrics_for_all_splits()
            self.logger.dump(self.num_timesteps)
            if hasattr(self.model, "ddp") and self.model.ddp:
                torch.distributed.barrier()
        return continue_training

    def _on_training_start(self):
        if self.first_step:
            # Do an initial validation before training
            print("Initial validation...")
            self._on_step()
            
    def compute_metrics_for_all_splits(self):
        original_trjs = copy.deepcopy(self.validation_buffer._trajectories)
        if hasattr(self.validation_buffer, "domain_to_indices"):
            original_domain_to_indices = copy.deepcopy(self.validation_buffer.domain_to_indices)
        for split in self.splits: 
            n_batch = self.n_batch
            if split != "full":
                # no need to filter for full split
                top_bottom, p = split.split("_")
                p = int(p) / 100
                n_batch = round(n_batch * p)
                # for each only keep top/bottom p trajectories
                filtered_trjs, filtered_domain_to_indices = self.extract_filtered_trajectories(top_bottom, p)
                self.validation_buffer._trajectories = filtered_trjs
                self.reset_valid_buffer()
                if filtered_domain_to_indices is not None:
                    # for multidomain buffer
                    self.validation_buffer.domain_to_indices = filtered_domain_to_indices
            # compute metrics for the split
            self.compute_metrics_for_single_split(split, n_batch=n_batch)
        # reset trajectory split to original
        self.validation_buffer._trajectories = original_trjs
        self.reset_valid_buffer()        
        if hasattr(self.validation_buffer, "domain_to_indices"):
            self.validation_buffer.domain_to_indices = original_domain_to_indices
    
    def extract_filtered_trajectories(self, top_bottom, p):
        filtered_trjs = collections.deque(maxlen=self.validation_buffer.buffer_size)
        filtered_domain_to_indices = None
        if hasattr(self.validation_buffer, "domain_to_indices"):
            filtered_domain_to_indices = collections.defaultdict(list)
        for task_id, trjs in self.validation_buffer.task_to_trj.items():
            task_trjs = filter_top_p_trajectories(trjs, top_p=p, 
                                                  epname_to_return=self.validation_buffer.trj_to_return,
                                                  bottom=top_bottom == "bottom")
            filtered_trjs += task_trjs
            if hasattr(self.validation_buffer, "domain_to_indices"): 
                # multidomain buffer
                domain = self.validation_buffer.task_to_domain[task_id]
                start_idx = filtered_domain_to_indices[domain][-1] + 1 \
                    if len(filtered_domain_to_indices[domain]) > 0 else 0
                domain_indices = list(range(start_idx, start_idx + len(task_trjs)))
                filtered_domain_to_indices[domain] += domain_indices
        return filtered_trjs, filtered_domain_to_indices
    
    def compute_metrics_for_single_split(self, split_prefix, n_batch): 
        for _ in tqdm(range(n_batch), desc=f"Validating {split_prefix}"):
            metrics = self._compute_metrics()
            for k, v in metrics.items():
                self.logger.record_mean(f"{self.prefix}/{split_prefix}/{k}", v)  
                if self.model.accumulation_steps > 1 and hasattr(self.validation_buffer, "domain_names") \
                    and self.validation_buffer.domain_names is not None: 
                    domain_name = self.validation_buffer.get_current_domain_name(self.model.accumulation_steps)
                    self.logger.record_mean(f"{self.prefix}/{split_prefix}/{domain_name}/{k}", v)  

    @torch.no_grad()     
    def _compute_metrics(self):
        # get batch
        observations, actions, next_observations, rewards, rewards_to_go, timesteps, attention_mask, \
                dones, task_ids, trj_ids, action_targets, action_mask, prompt, _, trj_seeds = self.model.sample_batch(
                    self.model.batch_size, buffer=self.validation_buffer
        )
        with torch.autocast(device_type='cuda', dtype=self.model.amp_dtype, enabled=self.model.use_amp):
            # compute model output
            policy_output = self.model.policy(
                states=observations,
                actions=actions,
                rewards=rewards,
                returns_to_go=rewards_to_go,
                timesteps=timesteps.long(),
                attention_mask=attention_mask,
                return_dict=True,
                with_log_probs=self.model.stochastic_policy,
                deterministic=False,
                prompt=prompt,
                task_id=self.model.current_task_id_tensor,
                ddp_kwargs=self.model.ddp_kwargs,
            )
         # compute loss
            _, loss_dict = self.model.compute_policy_loss(
                policy_output, action_targets, attention_mask, 0,
                ent_tuning=False, return_targets=rewards_to_go,
                reward_targets=rewards, state_targets=observations, dones=dones, 
                timesteps=timesteps, next_states=next_observations, action_mask=action_mask
            )
        return loss_dict
    
    def reset_valid_buffer(self): 
        self.validation_buffer.trj_ds_has_changed = True
        self.validation_buffer.trj_loader = None
        self.validation_buffer.trj_dataset = None
        self.validation_buffer.num_sampled_batches = 0
