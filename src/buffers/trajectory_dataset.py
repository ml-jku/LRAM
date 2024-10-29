import traceback
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from .buffer_utils import load_hdf5, load_npz, load_pkl
from ..algos.agent_utils import make_random_proj_matrix


class TrajectoryDataset(Dataset):

    def __init__(
        self,
        trajectories, 
        env, 
        context_len, 
        action_pad, 
        seqs_per_sample=1, 
        p_mask=0.0, 
        p_rand_trj=0.0,
        to_rgb=False, 
        last_trj_mask=True, 
        full_context_trjs=False, 
        var_context_trjs=False, 
        s_proj_raw=False, 
        img_is_encoded=False, 
        sample_start=False,
        trj_lengths=None, 
        trj_sample_kwargs=None, 
        max_state_dim=None, 
        max_act_dim=None, 
        transforms=None, 
        task_to_trj=None, 
        trj_to_task=None,
        task_seed_to_trj=None,
        trj_to_seed=None,
        n_ep_later=None, 
        p_skip_context=None, 
        seq_sample_kind="random", 
        n_proj_mat=100
    ):
        """
        Args:
            trajectories: List of Trajectory or Path objects.
            context_len: Int.
            action_pad: Int.
            to_rgb: Bool. Whether to convert gray-scale images to RGB format, by repeating the gray-scale channel.
            trj_lengths: Dict. Maps trj paths to respective lengths.
            max_state_dim: None or Int. For continuous observations (e.g., metaworld, dmcontrol) determines 
                if the obs sequence should be padded to max_state_dim. If None, no padding is done.
            max_act_dim: None or Int. For continous actions (e.g., metaworld, dmcontrol) determines if the
                action sequence should be padded to max_act_dim. If None, no padding is done.
            transforms: None or callable. If callable, it is applied to each state sample.
            seqs_per_sample: Int. Number of subsequences (from same task) to compose into a single sample.
            task_to_trj: None or Dict. Maps task ids to list of trj ids. Required for seqs_per_samples > 1.
            trj_to_task: None or Dict. Maps trj ids to task ids. Required for seqs_per_samples > 1.
            
        """
        self.trajectories = trajectories
        self.context_len = context_len
        self.action_pad = action_pad
        self.env = env
        self.max_state_dim = max_state_dim
        self.max_act_dim = max_act_dim
        self.to_rgb = to_rgb
        self.transforms = transforms
        self.trj_lengths = trj_lengths if trj_lengths is not None else {}
        self.trj_sample_kwargs = trj_sample_kwargs if trj_sample_kwargs is not None else {}
        self.seqs_per_sample = seqs_per_sample
        self.seq_sample_kind = seq_sample_kind
        self.task_to_trj = task_to_trj
        self.trj_to_task = trj_to_task
        self.task_seed_to_trj = task_seed_to_trj
        self.trj_to_seed = trj_to_seed
        self.last_trj_mask = last_trj_mask
        self.full_context_trjs = full_context_trjs
        self.p_mask = p_mask
        self.var_context_trjs = var_context_trjs
        self.p_rand_trj = p_rand_trj
        self.n_ep_later = n_ep_later
        self.p_skip_context = p_skip_context
        self.s_proj_raw = s_proj_raw
        self.n_proj_mat = n_proj_mat
        self.img_is_encoded = img_is_encoded
        self.sample_start = sample_start
        self.s_proj_mat = dict()
        if self.seqs_per_sample > 1:
            assert self.task_to_trj is not None and self.trj_to_task is not None

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        trj = self.trajectories[idx]
        if isinstance(trj, (str, Path)):
            # load from disk
            path = str(trj)
            s, s1, a, r, togo, t, done, task_id, trj_id, action_mask, trj_seed = self.get_sample_from_disk(path, idx)
            total_return = 0
        else:
            # samples stored in memory, load from there
            s, s1, a, r, togo, t, done, task_id, trj_id, action_mask, trj_seed = self.get_sample_from_memory(trj, idx)
            total_return = trj.total_return
        # postprocess states, actions
        if len(s.shape) == 4 and self.to_rgb:
            # convert to "RGB" by repeating the gray-scale channel
            s = np.repeat(s, 3, axis=1)
            s1 = np.repeat(s1, 3, axis=1)
        if self.env is not None:
            s = self.env.normalize_obs(s)
            s1 = self.env.normalize_obs(s1)
        padding = (self.context_len * self.seqs_per_sample) - s.shape[0]
        mask = self.make_attention_mask(padding, s.shape[0])        
        action_mask = np.ones_like(a, dtype=np.int32) if action_mask is None else action_mask
        if self.max_act_dim is not None and a.dtype.kind == "f": 
            a, action_mask = self.pad_actions(a)
        if self.max_state_dim is not None and len(s.shape) == 2:
            if self.s_proj_raw: 
                s, s1 = self.project_states(s), self.project_states(s1)
            else: 
                s, s1 = self.pad_states(s, s1)
        if padding:
            s, s1, a, r, togo, t, done, action_mask = self.pad_sequences(s, s1, a, r, togo, t, 
                                                                         done, action_mask, padding)
        if len(s.shape) == 4 and self.transforms is not None:
            # perform image augmentations
            s = self.transforms(torch.from_numpy(s).float())
            s1 = self.transforms(torch.from_numpy(s1).float())
        
        return s, a, s1, np.expand_dims(r, axis=1), np.expand_dims(togo, axis=1), \
               t, mask, done, task_id, trj_id, action_mask, total_return, trj_seed
               
    def get_sample_from_memory(self, trj, idx): 
        if self.seqs_per_sample > 1:
            if self.p_skip_context is not None and np.random.random() < self.p_skip_context:
                return self.get_single_sample_from_memory(trj, idx=idx)
            s, s1, a, r, togo, t, done, task_id, trj_id, action_mask, trj_seed = [], [], [], [], [], [],\
                [], [], [], None, []
            context_trj_ids = self.get_context_trj_ids(idx)
            # context trjs should not contain same trj
            all_trjs = [self.trajectories[trj_id] for trj_id in context_trj_ids if trj_id != trj] + [trj]            
            for i, at in enumerate(all_trjs):
                # if not last trajectory, sample full trajectory (if full_context_trjs)
                _s, _s1, _a, _r, _togo, _t, _done, _task_id, _trj_id, _, _trj_seed = self.get_single_sample_from_memory(
                    trj=at, 
                    full_trj=self.full_context_trjs if i < len(all_trjs) - 1 else False,
                    idx=idx,
                )
                s.append(_s)
                s1.append(_s1)
                a.append(_a)
                r.append(_r)
                togo.append(_togo)
                t.append(_t)
                done.append(_done)
                task_id.append(_task_id)
                trj_id.append(_trj_id)
                trj_seed.append(_trj_seed)
            
            # compose sample
            s, s1, a, r, togo, t, done = np.concatenate(s, axis=0), np.concatenate(s1, axis=0), \
                np.concatenate(a, axis=0), np.concatenate(r, axis=0), np.concatenate(togo, axis=0),  \
                np.concatenate(t, axis=0), np.concatenate(done, axis=0)
            
            if self.max_act_dim is not None and a.dtype.kind == "f": 
                a, action_mask = self.pad_actions(a)
            if self.max_state_dim is not None and len(s.shape) == 2:
                if self.s_proj_raw: 
                    s, s1 = self.project_states(s), self.project_states(s1)
                else: 
                    s, s1 = self.pad_states(s, s1)
            
            # only compute loss on last trj actions
            if self.last_trj_mask: 
                action_mask = np.zeros_like(a, dtype=np.int32)
                action_mask[-len(_s):] = 1

            return s, s1, a, r, togo, t, done, task_id[-1], trj_id[-1], action_mask, trj_seed[-1]
                        
        return self.get_single_sample_from_memory(trj, idx=idx)
        
    def get_single_sample_from_memory(self, trj, full_trj=False, idx=None):
        s, s1, a, r, togo, t, done, task_id, trj_id, trj_seed = trj.sample(self.context_len, full_trj=full_trj,
                                                                           sample_start=self.sample_start)
        return s, s1, a, r, togo, t, done, task_id, trj_id, None, trj_seed
    
    def get_sample_from_disk(self, path, idx):
        if self.seqs_per_sample > 1: 
            pass
        return self.get_single_sample_from_disk(path, idx)
    
    def get_single_sample_from_disk(self, path, idx): 
        # directly load subset of trajectory from disk making use of trj_lengths
        upper_bound = self.trj_lengths[path]
        # sample end_idx first
        end_idx = np.random.randint(1, upper_bound, size=1)[0] if upper_bound > 1 else 1
        if self.sample_start: 
            start_idx = np.random.randint(max(end_idx - self.context_len, 0), end_idx, size=1)[0]
        else: 
            start_idx = max(end_idx - self.context_len, 0)
        s, a, r, done_flag, togo = self.load_trj(path, start_idx=start_idx, end_idx=end_idx)
        assert togo is not None, "RTGs must be stored in trj file."
        r = r.astype(np.float32)
        if len(a.shape) == 1: 
            a = np.expand_dims(a, -1)
        if isinstance(done_flag, (list, tuple, np.ndarray)):
            done_flag = done_flag[..., -1]
        done = np.zeros(len(s))
        done[-1] = done_flag
        s1 = np.zeros_like(s)
        t = np.arange(start_idx, end_idx)
        task_id, trj_id, trj_seed = 0, idx, 0
        return s, s1, a, r, togo, t, done, task_id, trj_id, None, trj_seed
    
    def load_trj(self, path, start_idx=None, end_idx=None): 
        if path.endswith('.npz'):
            observations, actions, rewards, dones, returns_to_go = load_npz(path, start_idx=start_idx, end_idx=end_idx)
        elif path.endswith('.hdf5'):
            try: 
                observations, actions, rewards, dones, returns_to_go = load_hdf5(
                    path, start_idx=start_idx, end_idx=end_idx, img_is_encoded=self.img_is_encoded
                )
            except Exception as e:
                raise ValueError(f"Error loading {path}: {e}\n{traceback.format_exc()}")
        elif path.endswith('.pkl'):
            observations, actions, rewards, dones, returns_to_go = load_pkl(path, start_idx=start_idx, end_idx=end_idx)
        else: 
            raise ValueError("Only .npz and .hdf5 files are supported.")
        return observations, actions, rewards, dones, returns_to_go
        
    def pad_sequences(self, s, s1, a, r, togo, t, done, action_mask, padding):
        # obs is either 4 dimensional (image) or 2 dimensional (state input) 
        # first dimension is the sequence length
        obs_shape, act_dim = s.shape[1:], a.shape[-1]
        s = np.concatenate([np.zeros((padding, *obs_shape), dtype=s.dtype), s], axis=0)
        s1 = np.concatenate([np.zeros((padding, *obs_shape), dtype=s1.dtype), s1], axis=0)
        a = np.concatenate([np.ones((padding, act_dim), dtype=a.dtype) * self.action_pad, a], axis=0)
        r = np.concatenate([np.zeros((padding), dtype=r.dtype), r], axis=0)
        togo = np.concatenate([np.zeros((padding), dtype=togo.dtype), togo], axis=0)
        t = np.concatenate([np.zeros((padding), dtype=t.dtype), t], axis=0)
        done = np.concatenate([np.zeros((padding), dtype=done.dtype), done], axis=0)
        action_mask = np.concatenate([np.zeros((padding, act_dim), dtype=action_mask.dtype), action_mask], axis=0)
        return s, s1, a, r, togo, t, done, action_mask
    
    def pad_states(self, s, s1):
        # pad state input to max_state_dim in case of continuous state only
        padding = self.max_state_dim - s.shape[-1]
        s = np.concatenate([s, np.zeros((s.shape[0], padding), dtype=s.dtype)], axis=-1)
        s1 = np.concatenate([s1, np.zeros((s.shape[0], padding), dtype=s1.dtype)], axis=-1)
        return s, s1

    def pad_actions(self, a):
        # pad action to max_act_dim in case of continuous actions only
        padding = self.max_act_dim - a.shape[-1]
        action_mask = np.concatenate([np.ones_like(a), np.zeros((a.shape[0], padding))], axis=-1)
        a = np.concatenate([a, np.ones((a.shape[0], padding), dtype=a.dtype) * self.action_pad], axis=-1)
        return a, action_mask
    
    def get_context_trj_ids(self, idx):
        seqs_per_sample = self.seqs_per_sample if not self.var_context_trjs else np.random.randint(1, self.seqs_per_sample)
        if self.seq_sample_kind == "random": 
            # get subsequences from any task in any order    
            context_trj_ids = np.random.randint(0, len(self.trajectories), size=seqs_per_sample - 1)
        elif "task" in self.seq_sample_kind: 
            # get subsequences from same task either sorted by return or any order
            task_trj_ids = self.task_to_trj[self.trj_to_task[idx]]
            context_trj_ids = np.random.choice(task_trj_ids, seqs_per_sample - 1, replace=False)
            if self.seq_sample_kind == "task_ordered":
                context_trj_ids = sorted(context_trj_ids, key=lambda i: self.trajectories[i].total_return, reverse=True)
        elif self.seq_sample_kind == "sequential":
            min_task_idx = self.task_to_trj[self.trj_to_task[idx]][0]
            if self.n_ep_later is None: 
                context_trj_ids = list(range(max(idx - seqs_per_sample + 1, min_task_idx), idx))
            else: 
                start = max(idx - self.n_ep_later * (self.seqs_per_sample - 1), min_task_idx)
                interval = max((idx - start) // (self.seqs_per_sample - 1), 1)
                context_trj_ids = [start + i * interval for i in range(self.seqs_per_sample - 1)]
                context_trj_ids = [i for i in context_trj_ids if i < idx]
        else: 
            raise NotImplementedError(f"{self.seq_sample_kind} is not a valid seq_sample_kind.")
        # replace with random context trjs from any task
        if self.p_rand_trj > 0: 
            for i in range(len(context_trj_ids)): 
                if np.random.random() < self.p_rand_trj: 
                    context_trj_ids[i] = np.random.randint(0, len(self.trajectories))
        return context_trj_ids

    def make_attention_mask(self, padding, seq_len):
        seq_mask = np.random.binomial(1, 1.0 - self.p_mask, size=seq_len) if self.p_mask > 0.0 else np.ones(seq_len)
        return np.concatenate([np.zeros(padding), seq_mask], axis=0)

    def project_states(self, states, mat_idx=None): 
        # states is seq_len x obs_dim or seq_len x num_channels x h x w
        states = np.expand_dims(states, 0)
        in_dim = states.shape[-1]
        if in_dim not in self.s_proj_mat: 
            # create n_proj_mat projection matrices on first call
            self.s_proj_mat[in_dim] = make_random_proj_matrix(
                in_dim, self.max_state_dim, seed=42, batch_size=self.n_proj_mat
            )
        if mat_idx is None: 
            mat_idx = np.random.randint(0, self.n_proj_mat, states.shape[0])
        return np.matmul(states.squeeze(0), self.s_proj_mat[in_dim][mat_idx].squeeze(0).T)
