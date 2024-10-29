import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import psutil
import torch
from tqdm import tqdm


def maybe_split(dir_name: str) -> str:
    """
    Recursively splits a given dir_name at half, once it exceeds max folder size of 255.
    """
    if len(dir_name) > 255:
        half = len(dir_name) // 2
        dir_name = maybe_split(dir_name[:half]) + "/" + maybe_split(dir_name[half:])
    return dir_name


def multiply(value, mult):
    return value * mult


def safe_mean(arr): 
    return np.nan if len(arr) == 0 else np.mean(arr)


def load_layer_stats(path, layer_idx=0, masked=False):
    """
    Load layer stats from a given path. Assumes the file is a .json file.
    Args:
        path: Str. Path to stats file.
        layer_idx: Int. Index of layer stats to load.

    """
    with open(path, "r") as f:
        stats = json.load(f)
    if layer_idx < 0:
        layer_idx = list(stats.keys())[layer_idx]
    # in json, keys need to be string
    layer_idx = str(layer_idx)
    if masked:
        mean, std = stats[layer_idx]["mean"], stats[layer_idx]["std"]
    else:
        mean, std = stats[layer_idx]["mean_masked"], stats[layer_idx]["std_masked"]
    return mean, std


def set_frozen_to_eval(module):
    requires_grad = []
    for p in module.parameters():
        requires_grad.append(p.requires_grad)
    if not any(requires_grad):
        module.eval()


def load_layer_stats_per_task(path, layer_idx=0):
    with open(path, "r") as f:
        stats = json.load(f)
    if layer_idx < 0:
        layer_idx = stats.keys()[layer_idx]
    # in json, keys need to be string
    layer_idx = str(layer_idx)
    # iterate tasks, build up mean and std
    means, stds = [], []
    for task in stats.keys():
        means.append(stats[task][layer_idx]["mean"])
        stds.append(stats[task][layer_idx]["std"])
    return means, stds


def make_attention_maps(attention_scores, step, lower_triu=True, vmin=None, vmax=None):
    """
    attention_scores: Tuple of `torch.FloatTensor` (one for each layer) of shape
        `(batch_size, num_heads, sequence_length,sequence_length)`.
    step: Int. Current timestep

    """
    figures = {}
    mask = None
    for i, scores in enumerate(attention_scores):
        # first attention head
        if scores is None:
            print(f"Attention scores for layer {i} are None. Skipping")
            continue
        scores = scores.float().detach().cpu().numpy()
        h0_scores = scores[-1, 0]
        fig, ax = plt.subplots()
        if lower_triu:
            mask = np.triu(np.ones_like(h0_scores, dtype=bool))
            np.fill_diagonal(mask, False)
        sns.heatmap(h0_scores, cmap="rocket_r", mask=mask, ax=ax, vmin=vmin, vmax=vmax)
        ax.set_title(f"Timestep: {step}, Layer: {i}, Head: 0")
        figures[f"layer{i}_head0"] = fig
        # avg over all heads
        avg_scores = scores[-1].mean(0)
        fig, ax = plt.subplots()
        if lower_triu:
            mask = np.triu(np.ones_like(avg_scores, dtype=bool))
            np.fill_diagonal(mask, False)
        sns.heatmap(avg_scores, cmap="rocket_r", mask=mask, ax=ax, vmin=vmin, vmax=vmax)
        ax.set_title(f"Timestep: {step}, Layer: {i}, Head: all")
        figures[f"layer{i}_allheads"] = fig
    return figures


def get_ram_stats():
    """
    Retrieves memory statistics using psutil.

    Returns:
        dict: A dictionary containing memory usage details in megabytes (MB):
            - 'total': Total physical memory.
            - 'available': Memory available for new processes without swapping.
            - 'used': Memory currently in use.
            - 'free': Memory not being used at all.
            - 'percent': Percentage of memory used.
    """
    mem = psutil.virtual_memory()
    stats = {
        'total': round(mem.total / (1024 ** 2), 2),
        'available': round(mem.available / (1024 ** 2), 2),
        'used': round(mem.used / (1024 ** 2), 2),
        'free': round(mem.free / (1024 ** 2), 2),
        'percent': round(mem.percent, 2),
    }
    print("RAM stats: ", stats)
    return stats


def get_gpu_ram_stats(device):
    """
    Retrieves GPU memory statistics using PyTorch.

    Returns:
        dict: A dictionary containing GPU memory usage details in megabytes (MB):
            - 'total': Total GPU memory.
            - 'allocated': GPU memory currently allocated by PyTorch.
            - 'cached': GPU memory currently reserved by the caching allocator.
            - 'free': Approximate free GPU memory.
            - 'percent': Percentage of total memory allocated.
    """
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    allocated_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
    cached_mem = torch.cuda.memory_reserved(device) / (1024 ** 2)
    free_mem = total_mem - allocated_mem

    stats = {
        'gpu_ram_total': round(total_mem, 2),
        'gpu_ram_allocated': round(allocated_mem, 2),
        'gpu_ram_cached': round(cached_mem, 2),
        'gpu_ram_free': round(free_mem, 2),
        'gpu_ram_percent': round((allocated_mem / total_mem) * 100, 2),
    }
    return stats


def gather_dict(is_rank0, world_size, log_dict):
    gathered = [None for _ in range(world_size)] if is_rank0 else None
    torch.distributed.gather_object(log_dict, object_gather_list=gathered, dst=0)
    gathered_dict = None
    if gathered is not None: 
        gathered_dict = {}
        for d in gathered:
            for k, v in d.items():
                if k in gathered_dict: 
                    if isinstance(v, np.ndarray):
                        gathered_dict[k] = np.concatenate([gathered_dict[k], v])
                    elif isinstance(v, list):
                        gathered_dict[k].extend(v)
                    elif isinstance(v, dict): 
                        for kk, vv in v.items():
                            if kk in gathered_dict[k]: 
                                if isinstance(vv, np.ndarray):
                                    gathered_dict[k][kk] = np.concatenate([gathered_dict[k][kk], vv])
                                elif isinstance(vv, list):
                                    gathered_dict[k][kk].extend(vv)
                            else: 
                                gathered_dict[k][kk] = vv
                else: 
                    gathered_dict[k] = v
    return gathered_dict


def gather_list(is_rank0, world_size, log_array): 
    gathered = [None for _ in range(world_size)] if is_rank0 else None
    torch.distributed.gather_object(log_array, object_gather_list=gathered, dst=0)
    if gathered is not None:
        gathered = list(np.concatenate(gathered))
    return gathered


class ProgressParallel(joblib.Parallel):
    # from: https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def with_initializer(self, f_init):
    """
    Copied from: https://stackoverflow.com/questions/55424095/error-pickling-a-matlab-object-in-joblib-parallel-context
    Overwrite initializer hook in the Loky ProcessPoolExecutor
    - https://github.com/tomMoral/loky/blob/f4739e123acb711781e46581d5ed31ed8201c7a9/loky/process_executor.py#L850
    
    """
    hasattr(self._backend, '_workers') or self.__enter__()
    origin_init = self._backend._workers._initializer
    def new_init():
        origin_init()
        f_init()
    self._backend._workers._initializer = new_init if callable(origin_init) else f_init
    return self
