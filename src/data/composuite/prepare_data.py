import argparse
import collections
import json
import pathlib
import pickle
import h5py
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from joblib import delayed


def extract_array_stats(vals, prefix="", round=4):
    prefix = prefix + "_" if prefix else ""
    stats = {
        f"{prefix}min": np.min(vals).round(round),
        f"{prefix}max": np.max(vals).round(round),
        f"{prefix}mean": np.mean(vals).round(round),
        f"{prefix}std": np.std(vals).round(round),
        f"{prefix}q25": np.quantile(vals, 0.25).round(round),
        f"{prefix}q50": np.quantile(vals, 0.5).round(round),
        f"{prefix}q75": np.quantile(vals, 0.75).round(round),
        f"{prefix}q90": np.quantile(vals, 0.9).round(round),
        f"{prefix}q99": np.quantile(vals, 0.99).round(round),
    }
    return stats


def discount_cumsum_np(x, gamma):
    # much faster version of the above
    new_x = np.zeros_like(x)
    rev_cumsum = np.cumsum(np.flip(x, 0)) 
    new_x = np.flip(rev_cumsum * gamma ** np.arange(0, x.shape[0]), 0)
    new_x = np.ascontiguousarray(new_x).astype(np.float32)
    return new_x


def load_hdf5(path):
    with h5py.File(path, "r") as f:
        # fully trajectory
        observations = f['observations'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        terminals = f['terminals'][:]
        timeouts = f['timeouts'][:]
    dones = terminals | timeouts
    return observations, actions, rewards, dones


def extract_trajectories(observations, actions, rewards, dones, add_rtgs=False):
    trajectories = []
    trj_id = 0
    current_trj = collections.defaultdict(list)
    for s, a, r, done in tqdm(zip(observations, actions, rewards, dones),
                                    total=len(observations), desc="Extracting trajectories"):
        nans = [np.isnan(s).any(), np.isnan(a).any(), np.isnan(r)]
        if any(nans):
            print("NaNs found:", nans)
        s = s.astype(np.float32)
        current_trj["states"].append(s)
        current_trj["actions"].append(a)
        current_trj["rewards"].append(r)
        current_trj["dones"].append(done)
        if done:
            for k, v in current_trj.items():
                current_trj[k] = np.stack(v)
            if add_rtgs: 
                current_trj["returns_to_go"] = discount_cumsum_np(current_trj["rewards"], 1)
            current_trj["trj_id"] = trj_id
            trajectories.append(current_trj)
            current_trj = collections.defaultdict(list)
            trj_id += 1
    return trajectories
        

def save_episode(to_save, save_path, save_format="hdf5", compress=False):    
    if save_format == "hdf5":
        compress_kwargs = {"compression": "gzip", "compression_opts": 1} if compress else {}
        # compress_kwargs = compress_kwargs if compress_kwargs is not None else {}
        with h5py.File(save_path + ".hdf5", "w") as f:
            for k, v in to_save.items():
                if isinstance(v, (int, float, str, bool)):
                    # no compression
                    f.create_dataset(k, data=v)
                else: 
                    f.create_dataset(k, data=v, **compress_kwargs)
    elif save_format == "npzc": 
        np.savez_compressed(save_path, **to_save)
    elif save_format == "pkl": 
        with open(save_path + ".pkl", "wb") as f:
            pickle.dump(to_save, f)
    else: 
        np.savez(save_path, **to_save)
        
        
def save_json_stats(epname_to_len, epname_to_total_returns, epname_to_trjid, save_dir): 
    # store episode lengths 
    ep_lens = [v for v in epname_to_len.values()]
    ep_returns = [v for v in epname_to_total_returns.values()]
    # compute and dumpy episode stats
    stats = {
        "episodes": len(epname_to_len.keys()), 
        "transitions": sum(ep_lens),
        **extract_array_stats(ep_lens, prefix="episode_len"),
        **extract_array_stats(ep_returns, prefix="episode_return"),
    }
    print(" | ".join([f"{k}: {v}" for k, v in stats.items()]))
    with open(save_dir / "stats.json", "w") as f:
        json.dump(stats, f)
    with open(save_dir / "episode_lengths.json", "w") as f:
        json.dump(epname_to_len, f)
    with open(save_dir / "episode_returns.json", "w") as f:
        json.dump(epname_to_total_returns, f)
    with open(save_dir / "episode_trjids.json", "w") as f:
        json.dump(epname_to_trjid, f)
    return stats


def prepare_episodes_single(path, save_dir, save_format="hdf5", add_rtgs=False, compress=False):
    """
    Prepares a single composuite dataset for a given environment.

    Files are saved as follows:
    ```
    environment family (e.g. compusuite)
    - environment name 
    -- one hdf5 file per episode with fields: states, actions, rewards, returns_to_go, dones
    -- episode_lengths.json: dict with episode names as keys and episode lengths as values
    -- episode_returns.json: dict with episode names as keys and episode returns as values
    -- stats.json: dict with stats about the dataset
    ```

    Args:
        path: Str. Path to load episodes from.
        save_dir: Str. Path to save episodes to.
        add_rtgs: Bool. Whether to add returns-to-go to files.
        save_format: Str. File format to save episodes in.
        compress: Bool. Whether to apply compression or not.
    """
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)
    if not isinstance(save_dir, pathlib.Path):
        save_dir = pathlib.Path(save_dir)
    save_dir = save_dir / path.parent.stem
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # load episodes 
    observations, actions, rewards, dones = load_hdf5(path)
    episodes = extract_trajectories(observations, actions, rewards, dones, add_rtgs=add_rtgs)
    
    # save invidiual episodes 
    epname_to_len, epname_to_total_returns, epname_to_trjid = {}, {}, {}
    num_collected_transitions = 0
    for ep_idx, episode in enumerate(tqdm(episodes, desc="Saving episodes")):
        # record stats
        assert ep_idx == episode["trj_id"], f"Episode index {ep_idx} does not match trj_id {episode['trj_id']}."
        file_name = str(ep_idx)
        ep_len, ep_total_return = len(episode["states"]), episode["rewards"].sum()
        epname_to_len[file_name] = float(ep_len)
        epname_to_total_returns[file_name] = float(ep_total_return)
        epname_to_trjid[file_name] = ep_idx
        
        # save episode
        save_episode(episode, str(save_dir / file_name), save_format=save_format, compress=compress)
        num_collected_transitions += len(episode["states"])
        
    # extract and save stats
    stats = save_json_stats(epname_to_len, epname_to_total_returns, epname_to_trjid, save_dir)
    return stats
        
        
def prepare_episodes(data_dir, save_dir, save_format="hdf5", add_rtgs=False, compress=False): 
    """
    Prepares composuite datasets for all given paths. Write a single .hdf5 file per trajectory
    and otherwise keeps the trajectory structure the same.  

    Args:
        data_dir: Str. 
        save_dir: Str. Path to save episodes to.
        add_rtgs: Bool. Whether to add returns-to-go to files.
        save_format: Str. File format to save episodes in.
        compress: Bool. Whether to apply compression or not.

    """
    all_stats = {}
    if not isinstance(save_dir, pathlib.Path):
        save_dir = pathlib.Path(save_dir)
    # collect paths
    paths = [p for p in pathlib.Path(data_dir).rglob("**/*.hdf5")]
    for p in paths:
        print(f"Preparing episodes for {p}.")
        stats = prepare_episodes_single(
            p,
            save_dir=save_dir, 
            save_format=save_format,
            add_rtgs=add_rtgs,
            compress=compress,
        )
        all_stats[p.parent.stem] = stats
    pd.DataFrame(all_stats).round(4).T.to_csv(save_dir / "all_stats.csv")


def extract_shapes(data_dir):
    paths = sorted([p for p in pathlib.Path(data_dir).rglob("**/0.hdf5")])
    all_shapes = {}
    for p in paths: 
        with h5py.File(p, "r") as f:
            # extract shapes
            states, actions = f["states"][:], f["actions"][:]
            all_shapes[p.parent.stem] = {"states": states.shape[-1], "actions": actions.shape[-1]}
    with open('data_shapes.json', 'w') as f:
        json.dump(all_shapes, f, indent=4)


class ProgressParallel(joblib.Parallel):
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


def validate_fields(save_dir):
    paths = sorted([p for p in pathlib.Path(save_dir).rglob("**/*.hdf5")])
    print(f"Validating {len(paths)} files.")
    
    def validate_file(path):
        with h5py.File(path, "r") as f:
            for key in ['actions', 'dones', 'returns_to_go', 'rewards', 'states', 'trj_id']: 
                if key not in f.keys():
                    print(f"Invalid path: {path} missing key {key}.")
                    return path
        return None
    
    invalid_paths = ProgressParallel(n_jobs=128, total=len(paths))(delayed(validate_file)(p) for p in paths)
    invalid_paths = set([p for p in invalid_paths if p is not None])        
    print(f"Found {len(invalid_paths)} invalid paths: {invalid_paths}")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='./data/composuite/raw')
    parser.add_argument('--save_dir', type=str, 
                        default='./data/composuite/processed')
    parser.add_argument('--save_format', type=str, default="hdf5", help="File format to save episodes in.")
    parser.add_argument('--add_rtgs', action="store_true", help="Whether to precompute and add return-to-gos to files.")
    parser.add_argument('--compress', action="store_true", help="Whether to apply compression or not.")
    parser.add_argument('--shapes_only', action="store_true", help="Conduct shape.")
    parser.add_argument('--validate_fields_only', action="store_true", help="Conduct tests on processed hdf5 datasets.")
    args = parser.parse_args()
    if args.shapes_only: 
        extract_shapes(args.data_dir)
    elif args.validate_fields_only:
        validate_fields(args.save_dir)
    else: 
        prepare_episodes(args.data_dir, args.save_dir, save_format=args.save_format, add_rtgs=args.add_rtgs, 
                        compress=args.compress)
    print("Done.")