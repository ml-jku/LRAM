import gc
import collections
import pickle
import json
import torch
import numpy as np
import heapq
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import WeightedRandomSampler
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from typing import NamedTuple
from .trajectory import Trajectory
from .trajectory_dataset import TrajectoryDataset
from .buffer_utils import filter_top_p_trajectories, filter_trajectories_uniform, \
    filter_trajectories_first, filter_trajectories_last, load_hdf5, split_train_valid
from .samplers import DistributedSamplerWrapper, MixedBatchRandomSampler
from .dataloaders import MultiEpochsDataLoader
from ..envs.env_names import ENVID_TO_NAME
from ..augmentations import make_augmentations
from ..utils.misc import get_ram_stats


class TrajectoryReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    rewards_to_go: torch.Tensor
    timesteps: torch.Tensor
    attention_mask: torch.Tensor
    dones: torch.Tensor
    task_ids: torch.Tensor
    trj_ids: torch.Tensor
    action_mask: torch.Tensor
    total_returns: torch.Tensor
    trj_seeds: torch.Tensor


def make_default_dict_list():
    # neccessary for pickling 
    return collections.defaultdict(list)


class TrajectoryReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="cpu",
        n_envs=1,
        max_len=1000,
        context_len=20,
        action_pad=0,
        num_workers=0,
        init_top_p=1,
        init_p=1,
        init_first_p=1,
        init_last_p=1,
        seqs_per_sample=1,
        prefetch_factor=2,
        p_mask=0.0,
        p_rand_trj=0.0,
        p_valid=0.0,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
        as_heap=False,
        relative_pos_embds=False,
        pin_memory=False,
        last_seq_only=False,
        store_state_stats=False,
        shuffle=False,
        from_disk=True,
        to_rgb=False,
        ddp=False,
        use_next_obs=False,
        last_trj_mask=False,
        sample_full_seqs_only=False,
        full_context_trjs=False,
        var_context_trjs=False,
        use_global_trj_ids=True, 
        episodic=False,
        s_proj_raw=False,
        task_weighted_per_batch=False,
        img_is_encoded=False,
        seed_as_task=False,
        obs_to_float16=False,
        sample_start=False,
        seq_sample_kind="random",
        max_len_type="trajectory",
        max_state_dim=None,
        max_act_dim=None,
        target_return=None,
        augment_params=None,
        n_ep_later=None,
        p_skip_context=None,
        max_trj_len=None,
    ):
        # do not pass real buffer size, as we don't need to initialize buffers
        super().__init__(
            1, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination
        )
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self._obs_shape = None
        self._action_dim = None
        self.n_envs = n_envs
        self.max_len = max_len
        self.max_trj_len = max_trj_len
        self.context_len = context_len
        self.action_pad = action_pad
        self.target_return = target_return
        self.max_len_type = max_len_type
        self.as_heap = as_heap
        self.relative_pos_embds = relative_pos_embds
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.prefetch_factor = prefetch_factor
        self.init_p = init_p
        self.init_top_p = init_top_p
        self.init_first_p = init_first_p
        self.init_last_p = init_last_p
        self.seqs_per_sample = seqs_per_sample
        self.last_seq_only = last_seq_only
        self.store_state_stats = store_state_stats
        self.from_disk = from_disk
        self.to_rgb = to_rgb
        self.ddp = ddp
        self.max_state_dim = max_state_dim
        self.max_act_dim = max_act_dim
        self.use_next_obs = use_next_obs
        self.seq_sample_kind = seq_sample_kind
        self.last_trj_mask = last_trj_mask
        self.sample_full_seqs_only = sample_full_seqs_only
        self.full_context_trjs = full_context_trjs
        self.p_mask = p_mask
        self.var_context_trjs = var_context_trjs
        self.p_rand_trj = p_rand_trj
        self.use_global_trj_ids = use_global_trj_ids
        self.n_ep_later = n_ep_later
        self.p_skip_context = p_skip_context
        self.episodic = episodic
        self.s_proj_raw = s_proj_raw
        self.task_weighted_per_batch = task_weighted_per_batch
        self.img_is_encoded = img_is_encoded
        self.seed_as_task = seed_as_task
        self.obs_to_float16 = obs_to_float16
        self.sample_start = sample_start
        self.p_valid = p_valid
        self.transforms = None
        if augment_params is not None:
            self.transforms = make_augmentations(augment_params)
        self.buffer_stats = {}
        if self.as_heap:
            self._trajectories = []
            self.trajectories_heap = []
        else:
            self._trajectories = collections.deque(maxlen=buffer_size)
        self._trajectory_lengths = {}
        self._valid_trajectories = collections.deque(maxlen=buffer_size)
        self.current_trajectory = None
        self.full = False
        self.pos = 0
        self.max_return_so_far = -float("inf")
        self.state_mean = None
        self.state_std = None
        self.trj_dataset = None
        self.trj_loader = None
        self.trj_iterator = None
        self.trj_ds_has_changed = False
        # keep track of number of sampled batches with current loader
        self.num_sampled_batches = 0
        self.trajectory_probs = None
        self.total_transitions = 0
        # track task/trj_ids to enable sampling inter-task subsequences
        self.task_id = 0
        self.trj_id = 0
        # trj_id != global_trj_id 
        # --> when filtering, trj_id does not preserve ids of all trjs, global_trj_id does
        self.global_trj_id = 0
        self.task_to_trj = collections.defaultdict(list)
        self.trj_to_task = collections.defaultdict(int)
        self.trj_to_seed = collections.defaultdict(int)
        self.task_seed_to_trj = collections.defaultdict(make_default_dict_list)
        self.trj_to_return = {}
        self.trj_to_trjid = {}
        self.dataset_class = TrajectoryDataset
        self.dataloader_class = MultiEpochsDataLoader
        self.samples_class = TrajectoryReplayBufferSamples
        self.trj_sample_kwargs = {"relative_pos_embds": self.relative_pos_embds, "last_seq_only": self.last_seq_only,
                                  "handle_timeout_termination": self.handle_timeout_termination}
        
        # remove unnecessary variables
        del self.actions, self.observations, self.next_observations, self.rewards, self.dones, self.timeouts

    @property
    def trajectories(self):
        return self._trajectories

    @property
    def trajectory_lengths(self):
        return self._trajectory_lengths

    @property
    def obs_shape(self):
        if self._obs_shape is None:
            self._obs_shape = get_obs_shape(self.observation_space)
        return self._obs_shape

    @obs_shape.setter
    def obs_shape(self, value):
        self._obs_shape = value

    @property
    def action_dim(self):
        if self._action_dim is None:
            self._action_dim = get_action_dim(self.action_space)
        return self._action_dim

    @action_dim.setter
    def action_dim(self, value):
        self._action_dim = value

    def add(self, obs, next_obs, action, reward, done, infos=None):
        if self.current_trajectory is None:
            self.current_trajectory = Trajectory(
                self.obs_shape, self.action_dim, self.max_len,
                relative_pos_embds=self.relative_pos_embds, handle_timeout_termination=self.handle_timeout_termination,
                last_seq_only=self.last_seq_only
            )
        is_full = self.current_trajectory.add(obs, next_obs, action, reward, done, infos)
        if done or is_full:
            if not done and is_full:
                self.current_trajectory.add_dones()
            self.add_trajectory(self.obs_shape, self.action_dim)
            self.trj_ds_has_changed = True

    def is_full(self):
        if self.max_len_type == "transition":
            if self.total_transitions > self.buffer_size:
                return True
            return False
        if len(self.trajectories) > self.buffer_size:
            return True
        return False

    def is_empty(self):
        return len(self.trajectories) == 0

    def add_trajectory(self, obs_shape, action_dim, init_trj_buffers=True):
        # could make single call for these
        self.current_trajectory.setup_final_trj(target_return=self.target_return)
        self.max_return_so_far = max(self.max_return_so_far, self.current_trajectory.total_return)

        if self.as_heap:
            if self.is_full():
                _ = heapq.heappushpop(self.trajectories_heap, (self.current_trajectory.total_return, self.current_trajectory))
            else:
                heapq.heappush(self.trajectories_heap, (self.current_trajectory.total_return, self.current_trajectory))
            self._trajectories = [trj[1] for trj in self.trajectories_heap]
        else:
            if self.max_len_type == "transition":
                if self.is_full():
                    _ = self.trajectories.pop()
            self.trajectories.append(self.current_trajectory)
        # count transitions (do it here, to avoid recomputing anew when adding trajectories)
        self.total_transitions += len(self.current_trajectory)
        # init new trj
        self.current_trajectory = Trajectory(
            obs_shape, action_dim, self.max_len,
            relative_pos_embds=self.relative_pos_embds, handle_timeout_termination=self.handle_timeout_termination,
            last_seq_only=self.last_seq_only, init_trj_buffers=init_trj_buffers, 
            sample_full_seqs_only=self.sample_full_seqs_only, 
            episodic=self.episodic
        )
        if self.is_full():
            self.full = True

    def sample(self, batch_size=32, env=None, top_k=5, weight_by="len"):
        trajectory_probs = self.compute_trajectory_probs(top_k, weight_by)
        if self.trj_ds_has_changed or self.trj_loader is None:
            if self.trj_dataset is None:
                # only create this the very first time we sample.
                self.trj_dataset = self.make_dataset(env)
                self.trj_loader = self.make_dataloader(
                    self.trj_dataset, trajectory_probs=trajectory_probs, batch_size=batch_size
                )
                self.trj_iterator = iter(self.trj_loader)
            self.trj_ds_has_changed = False
        try:
            samples = next(self.trj_iterator)
        except StopIteration:
            print("Dataloader empty. Recreating... ")
            # required in case DataLoader is fully iterated.
            self.trj_iterator = iter(self.trj_loader)
            samples = next(self.trj_iterator)
            self.num_sampled_batches = 0
        self.num_sampled_batches += 1

        # remove next_observations to avoid unnecessary transfer transfer
        if not self.use_next_obs:
            samples[2] = None
            
        # data transfer is costly, use non_blocking to make sure loading happens asynchronously
        return self.samples_class(
            *[t.to(self.device, non_blocking=True) if t is not None else t for t in samples]
        )

    def make_dataset(self, env):
        if self.ddp: 
            # share objects
            from torch.multiprocessing import Manager
            manager = Manager()
            self._trajectories = manager.list([str(trj) if isinstance(trj, Path) else trj for trj in self._trajectories])
            self._trajectory_lengths = manager.dict(self._trajectory_lengths)
            self.task_to_trj = manager.dict(dict(self.task_to_trj))
            self.trj_to_task = manager.dict(dict(self.trj_to_task))
            self.trj_to_seed = manager.dict(dict(self.trj_to_seed))
            self.task_seed_to_trj = manager.dict(dict(self.task_seed_to_trj))
        return self.dataset_class(
            # do not pass env, as this requires lots of RAM
            self.trajectories, None, self.context_len, self.action_pad,
            trj_lengths=self.trajectory_lengths, trj_sample_kwargs=self.trj_sample_kwargs,
            max_state_dim=self.max_state_dim, max_act_dim=self.max_act_dim, to_rgb=self.to_rgb,
            transforms=self.transforms, task_to_trj=self.task_to_trj, trj_to_task=self.trj_to_task,
            seqs_per_sample=self.seqs_per_sample, seq_sample_kind=self.seq_sample_kind, 
            last_trj_mask=self.last_trj_mask, p_mask=self.p_mask, full_context_trjs=self.full_context_trjs,
            var_context_trjs=self.var_context_trjs, p_rand_trj=self.p_rand_trj, n_ep_later=self.n_ep_later,
            p_skip_context=self.p_skip_context, s_proj_raw=self.s_proj_raw,
            img_is_encoded=self.img_is_encoded, trj_to_seed=self.trj_to_seed, task_seed_to_trj=self.task_seed_to_trj,
            sample_start=self.sample_start
        )

    def make_dataloader(self, dataset, trajectory_probs, batch_size):
        sampler = self.make_sampler(dataset, trajectory_probs, batch_size)
        if self.ddp: 
            torch.distributed.barrier()
        get_ram_stats()
        trj_loader = self.dataloader_class(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=self.shuffle, 
            prefetch_factor=self.prefetch_factor
        )
        if self.ddp: 
            torch.distributed.barrier()
        return trj_loader

    def make_sampler(self, dataset, trajectory_probs, batch_size):
        # produce more samples such that dataloader doesn't need to be reconstructed after consumption
        mult = 100
        if self.task_weighted_per_batch:            
            # mix batches in proportion to tasks
            total_samples_per_domain = {t: len(v) for t, v in self.task_to_trj.items()}
            total_samples = sum(total_samples_per_domain.values())
            self.domain_weights = {i: total_samples_per_domain[i] / total_samples for i in total_samples_per_domain}
            sampler = MixedBatchRandomSampler(weights=trajectory_probs, domain_weights=self.domain_weights, 
                                              batch_size=batch_size, num_samples=len(dataset) * mult, replacement=True)
        else: 
            sampler = WeightedRandomSampler(weights=trajectory_probs, replacement=True, num_samples=len(dataset) * mult)
        if self.ddp:
            return DistributedSamplerWrapper(sampler)
        return sampler

    def compute_trajectory_probs(self, top_k=5, weight_by="len"):
        if self.trj_ds_has_changed or self.trajectory_probs is None:
            if weight_by == "return":
                if self.task_weighted_per_batch: 
                    # compute weights separately for each task
                    return_vals = {task: np.array([self.trajectories[t].total_return for t in trjs]) 
                                   for task, trjs in self.task_to_trj.items()}
                    return_vals = {task: (vals - vals.min()) / ((vals.max() - vals.min()) + 1e-8) 
                                   if vals.max() > vals.min() else np.ones_like(vals) 
                                   for task, vals in return_vals.items()}
                    self.trajectory_probs = {task: vals / (vals.sum() + 1e-8) for task, vals in return_vals.items()}
                else: 
                    return_vals = np.array([t.total_return for t in self.trajectories])
                    # min-max-norm to get weights
                    return_vals = (return_vals - return_vals.min()) / (return_vals.max() - return_vals.min())
                    self.trajectory_probs = return_vals / return_vals.sum()
            elif weight_by == "return_corrected":
                # additionally subtracts the length of the trajectory from reward
                # useful if e.g., alive_bonus is present in env
                return_vals = np.array([t.total_return - len(t) for t in self.trajectories])
                # min-max-norm to get weightsx
                return_vals = (return_vals - return_vals.min()) / (return_vals.max() - return_vals.min())
                self.trajectory_probs = return_vals / return_vals.sum()
            elif weight_by == "top_k":
                upper_bound = len(self)
                top_k = top_k if top_k < upper_bound else upper_bound
                return_vals = np.array([t.total_return for t in self.trajectories])
                top_k_inds = np.argpartition(return_vals, -top_k)[-top_k:]
                self.trajectory_probs = np.zeros_like(return_vals)
                self.trajectory_probs[top_k_inds] = 1 / top_k
            elif weight_by == "len":
                if self.task_weighted_per_batch:
                    # compute weights separately for each task
                    trj_lens = {task: [len(self.trajectories[t]) if isinstance(self.trajectories[t], Trajectory) 
                                    else self.trajectory_lengths[str(self.trajectories[t])] for t in trjs]
                                for task, trjs in self.task_to_trj.items()}
                    total_samples_per_domain = {task: sum(lens) for task, lens in trj_lens.items()}
                    self.trajectory_probs = {task: [l / total_samples_per_domain[task] for l in lens] 
                                            for task, lens in trj_lens.items()}
                else: 
                    trj_lens = [len(t) if isinstance(t, Trajectory) else self.trajectory_lengths[str(t)] 
                                for t in self.trajectories]
                    total_samples = sum(trj_lens)
                    self.trajectory_probs = [l / total_samples for l in trj_lens]
            elif weight_by == "uniform":
                # i.e., uniform weight for all
                num_trjs = len(self)
                self.trajectory_probs = [1 / num_trjs] * num_trjs
            elif weight_by == "reward_std":
                std_vals = np.array([t.std_reward for t in self.trajectories])
                # min-max-norm to get weights
                std_vals = (std_vals - std_vals.min()) / (std_vals.max() - std_vals.min())
                self.trajectory_probs = std_vals / std_vals.sum()
            else:
                raise NotImplementedError()
        return self.trajectory_probs

    def _get_buffer_stats(self, prefix="buffer", midfix=""):
        # midfix required in case of duplicate subkeys
        stats = {}
        if self.trj_ds_has_changed or not self.buffer_stats:
            if self.is_empty():
                return stats
            stats[f"{prefix}/{midfix}buffer_size"] = len(self.trajectories)
            trj_lengths = [len(trj) for trj in self.trajectories if isinstance(trj, Trajectory)]
            if trj_lengths:
                stats[f"{prefix}/{midfix}n_transitions_in_buffer"] = np.sum(trj_lengths)
                stats[f"{prefix}/{midfix}trj_length_mean"] = np.mean(trj_lengths)
                stats[f"{prefix}/{midfix}trj_length_std"] = np.std(trj_lengths)
                stats[f"{prefix}/{midfix}trj_length_min"] = np.min(trj_lengths)
                stats[f"{prefix}/{midfix}trj_length_max"] = np.max(trj_lengths)
            trj_rewards = [trj.rewards.sum() for trj in self.trajectories if isinstance(trj, Trajectory)]
            if trj_rewards:
                stats[f"{prefix}/{midfix}trj_rewards_mean"] = np.mean(trj_rewards)
                stats[f"{prefix}/{midfix}trj_rewards_std"] = np.std(trj_rewards)
                stats[f"{prefix}/{midfix}trj_rewards_min"] = np.min(trj_rewards)
                stats[f"{prefix}/{midfix}trj_rewards_max"] = np.max(trj_rewards)
                stats[f"{prefix}/{midfix}trj_rewards_max_so_far"] = self.max_return_so_far
            self.buffer_stats = stats
        else:
            stats = self.buffer_stats
        return stats

    def _get_max_return(self):
        return self.max_return_so_far

    def _get_mean_topk_return(self, k=50):
        topk_returns = np.array(sorted([t.total_return for t in self.trajectories], reverse=True)[:k])
        return np.random.uniform(topk_returns.mean(), topk_returns.mean() + topk_returns.std())

    def _get_quantile_return(self, q=0.75):
        all_returns = [t.total_return for t in self.trajectories]
        return np.quantile(all_returns, q=q)

    def _get_topk_trajectories(self, top_k=5):
        upper_bound = len(self.trajectories)
        top_k = top_k if top_k < upper_bound else upper_bound
        return_vals = np.array([t.total_return for t in self.trajectories])
        top_k_inds = np.argpartition(return_vals, -top_k)[-top_k:]
        return [self.trajectories[idx] for idx in top_k_inds]

    def reset(self, top_percent=1, verbose=True):
        self.total_transitions = 0
        if top_percent > 0:
            if verbose: 
                print(f"Reinitializing buffer with top {top_percent * 100}% of trajectories...")
            self._trajectories = collections.deque(filter_top_p_trajectories(self.trajectories, top_percent))
        else:
            if verbose: 
                print("Reinitializing buffer...")
            self._trajectories = collections.deque(maxlen=self.buffer_size)
            self.max_return_so_far = -float("inf")

        self.pos = len(self)
        self.trj_dataset = None
        self.trj_loader = None
        self.trj_iterator = None
        self.trj_ds_has_changed = True
        self.trajectory_probs = None
        self.current_trajectory = None
        self._trajectory_lengths = {}

        if not self.is_full():
            self.full = False

    def init_buffer_from_dataset(self, paths):
        print("Intitializing buffer from data paths.")
        assert "base" in paths and "names" in paths
        base_path, names = paths["base"], paths["names"]
        if isinstance(names, str):
            names = [names]
        paths = [Path(base_path) / ENVID_TO_NAME.get(name, name) for name in names]
        states_store, states_mean = [], []
        task_id = self.task_id
        for path in paths:
            print(f"Loading trajectories from: {path}")
            self.set_task_id(task_id)
            trajectories = self.load_trajectory_dataset(path)
            if hasattr(self, "task_to_domain"):
                # only for multidomain buffer
                self.task_to_domain[self.task_id] = self.domain_id
            if trajectories is None:
                print(f"No trajectories loaded from: {path}")
                task_id += 1
                continue
            init_top_p = self.init_top_p if not isinstance(self.init_top_p, list) else self.init_top_p[task_id]
            if init_top_p < 1:
                trajectories = filter_top_p_trajectories(trajectories, top_p=init_top_p)
            if self.init_p < 1:
                trajectories = filter_trajectories_uniform(trajectories, p=self.init_p)
            if self.init_first_p < 1:
                trajectories = filter_trajectories_first(trajectories, p=self.init_first_p)
            if self.init_last_p < 1:
                trajectories = filter_trajectories_last(trajectories, p=self.init_last_p)
            if self.p_valid > 1:
                raise NotImplementedError("Validation set currently only supported when loading from disk.")
            
            for trj in tqdm(trajectories, desc="Storing trajectories"):
                if self.full:
                    break
                observations, next_observations, actions, rewards, dones, trj_id, trj_seed = trj["observations"],\
                    trj["next_observations"], trj["actions"], trj["rewards"],\
                    trj["terminals"], trj.get("trj_id", self.trj_id), trj.get("trj_seed", 0)

                if self.current_trajectory is None:
                    self.current_trajectory = Trajectory(
                        None, None, self.max_len,
                        relative_pos_embds=self.relative_pos_embds, task_id=task_id,
                        handle_timeout_termination=self.handle_timeout_termination, trj_id=trj_id,
                        last_seq_only=self.last_seq_only, init_trj_buffers=False,
                        sample_full_seqs_only=self.sample_full_seqs_only,
                        episodic=self.episodic, trj_seed=trj_seed
                    )
                    
                observations = np.vstack(observations) if isinstance(observations, list) else observations
                next_observations = np.vstack(next_observations) if next_observations is not None else None
                actions, rewards, dones = np.vstack(actions), np.stack(rewards).reshape(-1), np.stack(dones).reshape(-1)
                if self.max_trj_len is not None: 
                    observations, next_observations, actions, rewards, dones = observations[:self.max_trj_len],\
                        next_observations[:self.max_trj_len] if next_observations is not None else None, \
                        actions[:self.max_trj_len], rewards[:self.max_trj_len],\
                        dones[:self.max_trj_len]
                self.current_trajectory.add_full_trj(
                    observations,
                    next_observations if self.use_next_obs else None,
                    actions,
                    rewards,
                    dones,
                    task_id=task_id if not self.seed_as_task else (task_id + 1) * trj_seed,
                    trj_id=trj_id,
                    trj_seed=trj_seed
                )
                self.add_trajectory(None, None, init_trj_buffers=False)
                if self.store_state_stats:
                    obs_stacked = np.vstack(observations)
                    if len(paths) == 1:
                        states_store.append(obs_stacked)
                    else:
                        states_mean.append(obs_stacked.mean(axis=0))

                self.task_to_trj[task_id].append(self.trj_id)
                self.trj_to_task[self.trj_id] = task_id
                self.trj_to_seed[self.trj_id] = trj_seed
                self.task_seed_to_trj[task_id][trj_seed].append(self.trj_id)
                self.trj_id += 1
            
            task_id += 1
            # free ram
            for t in trajectories: 
                del t
            del trajectories
            gc.collect()

        if self.store_state_stats:
            # don't use normalization for a replay buffer consisting of mixture of datasets
            if len(paths) == 1:
                self.state_mean = np.vstack(states_store).mean(axis=0)
                # add very small epsilon to ensure it's not 0
                self.state_std = np.vstack(states_store).std(axis=0) + 1e-8
            else:
                self.state_mean = np.vstack(states_mean).mean(axis=0)
                self.state_std = np.vstack(states_mean).std(axis=0) + 1e-8

        self.set_task_id(0)

    def load_trajectory_dataset(self, path):
        assert isinstance(path, Path), "Path must be a Path object."
        if path.suffix == ".pkl":
            with open(str(path), "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, ReplayBuffer):
                trajectories = self.extract_trajectories_from_buffer(obj)
            else:
                trajectories = obj
        elif path.suffix == ".npz" or path.suffix == ".npy":
            obj = np.load(str(path))
            trajectories = self.extract_trajectories_from_npz(obj)
        elif path.is_dir():
            if self.from_disk:
                trj_files = sorted([p for p in path.glob("*.npz")])
                trj_files += sorted([p for p in path.glob("*.hdf5")])
                trj_files += sorted([p for p in path.glob("*.pkl")])
                
                # sort according to trjid order of files (important for procgen)
                trj_id_path = path / "episode_trjids.json"
                if trj_id_path.exists():
                    with open(trj_id_path, "r") as f:
                        epname_to_trjid = json.load(f)
                    trj_files = sorted(trj_files, key=lambda x: epname_to_trjid[x.stem])
                    if self.init_first_p < 1:
                        trj_files = filter_trajectories_first(trj_files, p=self.init_first_p)
                    if self.init_last_p < 1:
                        trj_files = filter_trajectories_last(trj_files, p=self.init_last_p)

                # filter trajectories
                init_top_p = self.init_top_p if not isinstance(self.init_top_p, list) else self.init_top_p[self.task_id]
                if init_top_p < 1 or self.p_valid > 0:
                    returns_path = path / "episode_returns.json"
                    with open(returns_path, "r") as f:
                        name_to_return = json.load(f)
                    for p in trj_files:
                        self.trj_to_return[str(p)] = name_to_return[p.stem]
                    if init_top_p < 1:
                        trj_files = filter_top_p_trajectories(trj_files, top_p=init_top_p, epname_to_return=name_to_return)
                
                # extract episode lengths
                lengths_path = path / "episode_lengths.json"
                if lengths_path.exists():
                    with open(lengths_path, "r") as f:
                        name_to_len = json.load(f)
                    for p in trj_files:
                        self._trajectory_lengths[str(p)] = name_to_len[p.stem]
                trajectories = None

                # track task/trj ids
                for trj_id in range(self.trj_id, self.trj_id + len(trj_files)):
                    self.task_to_trj[self.task_id].append(trj_id)
                    self.trj_to_task[trj_id] = self.task_id
                    self.trj_to_trjid[str(trj_files[trj_id - self.trj_id])] = trj_id
                self.trj_id += len(trj_files)
                
                if self.p_valid > 0:
                    trj_files, valid_trj_files = split_train_valid(trj_files, p=self.p_valid, 
                                                                   trj_len_dict=self._trajectory_lengths)
                    assert set(trj_files).isdisjoint(valid_trj_files), "Train and valid sets must be disjoint."
                    self._valid_trajectories += valid_trj_files
                self._trajectories += trj_files
            else:
                trajectories = self.extract_trajectories_from_dir(path)
        else:
            raise NotImplementedError("Unsupported file type.")
        return trajectories

    def extract_trajectories_from_buffer(self, obj):
        pos = obj.pos if not obj.full else len(obj.observations)
        observations, next_observations, actions, rewards, dones = obj.observations[:pos], obj.next_observations[:pos], \
            obj.actions[:pos], obj.rewards[:pos], obj.dones[:pos]
        trajectories = self.extract_trajectories(observations, next_observations, actions, rewards, dones)
        return trajectories

    def extract_trajectories_from_npz(self, obj):
        observations, next_observations, actions, rewards, dones = obj["observations"], obj["next_observations"],\
            obj["actions"], obj["rewards"], obj["dones"]
        trajectories = self.extract_trajectories(observations, next_observations, actions, rewards, dones)
        return trajectories

    def extract_trajectories_from_dir(self, path):
        trajectories = []
        # directory contains multiple .npz files or .hdf5 files
        paths = sorted([p for p in Path(path).glob("**/*.npz")])
        if not paths:
            paths = sorted([p for p in Path(path).glob("**/*.hdf5")])
        # load trj seeds if exists
        epname_to_seed = None
        if (path / "episode_seeds.json").exists():
            with open(path / "episode_seeds.json", "r") as f:
                epname_to_seed = json.load(f)
        # sort according to trjid order of files (important for procgen)
        if (path / "episode_trjids.json").exists():
            with open(path / "episode_trjids.json", "r") as f:
                epname_to_trjid = json.load(f)
            paths = sorted(paths, key=lambda x: epname_to_trjid[x.stem])
        for i, p in enumerate(tqdm(paths, total=len(paths), desc="Extracting trajectories")):
            if p.suffix == ".hdf5":
                observations, actions, rewards, dones, _ = load_hdf5(p, img_is_encoded=self.img_is_encoded)
            else:
                trj = np.load(str(p))
                observations, actions, rewards = trj["states"], list(trj["actions"]), list(trj["rewards"])
                dones = np.array([trj["dones"]])
            if self.obs_to_float16:
                observations = observations.astype(np.float16) 
            # make a dict
            trajectories.append({
                "observations": observations,
                "next_observations": None,
                "actions": actions,
                "rewards": rewards,
                "terminals": dones,
                "trj_id": i if not self.use_global_trj_ids else self.global_trj_id,
                "trj_seed": epname_to_seed[p.stem] if epname_to_seed is not None else 0
            })
            # only increase global_trjid, rest will be added above
            self.global_trj_id += 1
            
        return trajectories

    def extract_trajectories(self, observations, next_observations, actions, rewards, dones):
        trajectories = []
        trj_id = 0
        current_trj = collections.defaultdict(list)
        for s, s1, a, r, done in tqdm(zip(observations, next_observations,
                                          actions, rewards, dones),
                                      total=len(observations), desc="Extracting trajectories"):
            nans = [np.isnan(s).any(), np.isnan(s1).any(), np.isnan(a).any(), np.isnan(r)]
            if any(nans):
                print("NaNs found:", nans)
            s = s.astype(np.float32)
            s1 = s1.astype(np.float32)
            current_trj["observations"].append(s)
            current_trj["next_observations"].append(s1)
            current_trj["actions"].append(a)
            current_trj["rewards"].append(r)
            current_trj["terminals"].append(done)
            if done:
                current_trj["trj_id"] = trj_id if not self.use_global_trj_ids else self.global_trj_id
                trajectories.append(current_trj)
                current_trj = collections.defaultdict(list)
                trj_id += 1
                self.global_trj_id += 1
        
        return trajectories

    def size(self):
        return len(self.trajectories)

    def __len__(self):
        return len(self.trajectories)

    def get_state_mean_std(self):
        return self.state_mean, self.state_std

    def set_task_id(self, task_id):
        self.task_id = task_id
        self.trj_loader = None
        self.trj_dataset = None
        self.trj_ds_has_changed = True
        
    def init_from_existing_buffer(self, buffer, validation=False):
        if validation: 
            assert self.p_valid > 0, "Validation buffer requires p_valid > 0."
        if self.init_p < 1:
            # subsample each task trjs uniformly according to trj lens
            for task, trj_ids in buffer.task_to_trj.items():
                # task trj lens 
                trj_lens = [len(buffer.trajectories[t]) if isinstance(buffer.trajectories[t], Trajectory) 
                            else self.trajectory_lengths[t]
                            for t in trj_ids]
                trj_probs = np.array(trj_lens) / sum(trj_lens)
                sampled_trj_ids = sorted(np.random.choice(trj_ids, int(self.init_p * len(trj_ids)), 
                                                          p=trj_probs, replace=False))
                for tid in sampled_trj_ids:
                    self._trajectories.append(buffer.trajectories[tid])
        else: 
            self._trajectories = buffer._trajectories if not validation else buffer._valid_trajectories
        self._trajectory_lengths = buffer._trajectory_lengths
        self.task_id = buffer.task_id
        self.trj_id = buffer.trj_id
        self.global_trj_id = buffer.global_trj_id
        self.task_to_trj = buffer.task_to_trj
        self.trj_to_task = buffer.trj_to_task
        self.trj_to_return = buffer.trj_to_return
        self.trj_to_trjid = buffer.trj_to_trjid
        self.state_mean = buffer.state_mean
        self.state_st = buffer.state_std
        self.max_return_so_far = buffer.max_return_so_far
        if validation: 
            self.task_to_trj = collections.defaultdict(list)
            for trj in self._trajectories:
                trjid = self.trj_to_trjid[str(trj)]
                self.task_to_trj[self.trj_to_task[trjid]].append(trj)
