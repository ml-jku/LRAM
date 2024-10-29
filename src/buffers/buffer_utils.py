import h5py
import pickle
import numpy as np
import torch


def discount_cumsum(x, gamma):
    new_x = np.zeros_like(x)
    new_x[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        new_x[t] = x[t] + gamma * new_x[t + 1]
    return new_x

def discount_cumsum_np(x, gamma):
    # much faster version of the above
    new_x = np.zeros_like(x)
    rev_cumsum = np.cumsum(np.flip(x, 0)) 
    new_x = np.flip(rev_cumsum * gamma ** np.arange(0, x.shape[0]), 0)
    new_x = np.ascontiguousarray(new_x).astype(np.float32)
    return new_x


def discount_cumsum_torch(x, gamma):
    new_x = torch.zeros_like(x)
    rev_cumsum = torch.cumsum(torch.flip(x, [0]), 0)
    new_x = torch.flip(rev_cumsum * gamma ** torch.arange(0, x.shape[0], device=x.device), [0])
    new_x = new_x.contiguous().to(dtype=torch.float32)
    return new_x


def compute_rtg_from_target(x, target_return):
    new_x = np.zeros_like(x)
    new_x[0] = target_return
    for i in range(1, x.shape[0]):
        new_x[i] = min(new_x[i - 1] - x[i - 1], target_return)
    return new_x


def split_train_valid(trajectories, p=1, trj_len_dict=None, seed=1):
    # split trajectories into train and validation sets, by trj len weights
    if trj_len_dict is not None: 
        trj_lens = [trj_len_dict[str(t)] for t in trajectories] 
    else: 
        trj_lens = [len(t["observations"]) for t in trajectories]
    p_train = 1 - p
    total_samples = sum(trj_lens)    
    trajectory_probs = [l / total_samples for l in trj_lens]
    # always performs same split via random state
    random_state = np.random.RandomState(seed=seed)
    idx = random_state.choice(len(trajectories), size=int(len(trajectories) * p_train),
                           p=trajectory_probs, replace=False)
    train_trjs = [trajectories[i] for i in idx]
    idx = set(idx)
    valid_trjs = [trajectories[i] for i in range(len(trajectories)) if i not in idx]
    return train_trjs, valid_trjs


def filter_top_p_trajectories(trajectories, top_p=1, epname_to_return=None, bottom=False):
    start = len(trajectories) - int(len(trajectories) * top_p)
    if epname_to_return is None: 
        if hasattr(trajectories[0], "rewards"):
            def sort_fn(x): return np.array(x.rewards).sum()
        else: 
            def sort_fn(x): return np.array(x.get("rewards")).sum()
    else:
        def sort_fn(x): return epname_to_return[str(x)]
    sorted_trajectories = sorted(trajectories, key=sort_fn, reverse=bottom)
    return sorted_trajectories[start:]


def filter_trajectories_uniform(trajectories, p=1):
    # sample uniformly with trj len weights
    trj_lens = [len(t["observations"]) for t in trajectories]
    total_samples = sum(trj_lens)
    trajectory_probs = [l / total_samples for l in trj_lens]
    idx = np.random.choice(len(trajectories), size=int(len(trajectories) * p), p=trajectory_probs, replace=False)
    return [trajectories[i] for i in idx]


def filter_trajectories_first(trajectories, p=1):
    return trajectories[:int(len(trajectories) * p)]


def filter_trajectories_last(trajectories, p=1):
    return trajectories[int(len(trajectories) * p): ]


def load_npz(path, start_idx=None, end_idx=None): 
    returns_to_go = None
    with np.load(path, mmap_mode="r" if start_idx and end_idx else None) as trj: 
        if start_idx is not None and end_idx is not None:
            # subtrajectory only
            observations, actions, rewards = trj["states"][start_idx: end_idx].astype(np.float32), \
                trj["actions"][start_idx: end_idx].astype(np.float32), trj["rewards"][start_idx: end_idx].astype(np.float32)
            if "returns_to_go" in trj:
                returns_to_go = trj["returns_to_go"][start_idx: end_idx].astype(np.float32)
        else: 
            # fully trajectory
            observations, actions, rewards = trj["states"], trj["actions"], trj["rewards"], 
            if "returns_to_go" in trj:
                returns_to_go = trj["returns_to_go"].astype(np.float32)
        dones = np.array([trj["dones"]])
    return observations, actions, rewards, dones, returns_to_go


def load_hdf5(path, start_idx=None, end_idx=None, img_is_encoded=False):
    returns_to_go, dones = None, None
    with h5py.File(path, "r") as f:
        if start_idx is not None and end_idx is not None:
            # subtrajectory only
            if img_is_encoded:
                observations = f['states_encoded'][start_idx: end_idx]
            else: 
                observations = f['states'][start_idx: end_idx]
            actions = f['actions'][start_idx: end_idx]
            rewards = f['rewards'][start_idx: end_idx]
            if "returns_to_go" in f:
                returns_to_go = f["returns_to_go"][start_idx: end_idx]
            if "dones" in f: 
                try:
                    dones = f['dones'][start_idx: end_idx]
                except Exception as e: 
                    pass
        else: 
            # fully trajectory
            if img_is_encoded:
                observations = f['states_encoded'][:]
            else: 
                observations = f['states'][:]
            actions = f['actions'][:]
            rewards = f['rewards'][:]
            if "returns_to_go" in f:
                returns_to_go = f["returns_to_go"][:]
            if "dones" in f:
                try:
                    dones = f['dones'][:]
                except Exception as e: 
                    pass
        if dones is None: 
            dones = np.array([f['dones'][()]])
    return observations, actions, rewards, dones, returns_to_go

    
def append_to_hdf5(path, new_vals, compress_kwargs=None):
    compress_kwargs = {"compression": "gzip", "compression_opts": 1} if compress_kwargs is None \
        else compress_kwargs
    # open in append mode, add new vals
    with h5py.File(str(path), 'a') as f:
        for k, v in new_vals.items():
            if k in f:
                del f[k]
            f.create_dataset(k, data=v, **compress_kwargs)


def load_pkl(path, start_idx=None, end_idx=None): 
    returns_to_go = None
    with open(path, "rb") as f:
        trj = pickle.load(f)
    if start_idx is not None and end_idx is not None:
        # subtrajectory only
        observations, actions, rewards = trj["states"][start_idx: end_idx], \
            trj["actions"][start_idx: end_idx], trj["rewards"][start_idx: end_idx]
        if "returns_to_go" in trj:
            returns_to_go = trj["returns_to_go"][start_idx: end_idx]
    else: 
        # fully trajectory
        observations, actions, rewards = trj["states"], trj["actions"], trj["rewards"], 
        if "returns_to_go" in trj:
            returns_to_go = trj["returns_to_go"]
    dones = np.array([trj["dones"]])    
    return observations, actions, rewards, dones, returns_to_go


def compute_start_end_context_idx(idx, seq_len, cache_len, future_cache_len, full_context_len=True, dynamic_len=False):
    start = max(0, idx - cache_len)
    end = min(seq_len, idx + future_cache_len)
    if dynamic_len: 
        start = np.random.randint(start, idx + 1)
        end = np.random.randint(idx, end + 1)
    elif full_context_len: 
        total_cache_len = cache_len + future_cache_len
        if end - start < total_cache_len:
            if start > 0:
                start -= total_cache_len - (end - start)
            else:
                end += total_cache_len - (end - start)
            start = max(0, start)
            end = min(seq_len, end)
    return start, end
