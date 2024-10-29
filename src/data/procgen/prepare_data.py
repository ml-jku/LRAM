import argparse
import collections
import json
import pathlib
import pickle
import h5py
import numpy as np
import pandas as pd
import tarfile
import gzip
import functools
from tqdm import tqdm
from joblib import delayed
# required for loading the trj buffers.
import sys
sys.path.insert(0, "../../..")
from src.utils.misc import ProgressParallel


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


def extract_props_from_file_name(file_name): 
    """
    Extracts properties from the file name.
    """
    datetime, trj_id, trj_len, seed, total_return = file_name.split("_")
    return {"datetime": datetime, "trj_id": int(trj_id), "seed": int(seed), 
            "trj_len": int(trj_len), "total_return": float(total_return)}


def discount_cumsum_np(x, gamma):
    # much faster version of the above
    new_x = np.zeros_like(x)
    rev_cumsum = np.cumsum(np.flip(x, 0)) 
    new_x = np.flip(rev_cumsum * gamma ** np.arange(0, x.shape[0]), 0)
    new_x = np.ascontiguousarray(new_x).astype(np.float32)
    return new_x


def load_episode(path, add_rtg=False, to_255=False):
    assert path.suffix == ".npy", "Only .npy files are supported for raw procgen."
    obj = np.load(path, allow_pickle=True).item()
    # remove reset state in the end of the episode
    episode = {"states": obj["observations"][:-1], "actions": obj["actions"],
               "rewards": obj["rewards"].astype(np.float32).reshape(-1), "dones": obj["dones"]}
    if add_rtg:
        episode["returns_to_go"] = discount_cumsum_np(episode["rewards"], 1)
    if to_255:
        episode["states"] = (episode["states"] * 255.0)
    # convert states to uint8 --> storage efficiency
    episode["states"] = episode["states"].astype(np.uint8)
    return episode


def load_buffer_split(split_path, n_envs=None):
    assert split_path.suffix == ".gz", "Only .gz files are supported for buffer splits."
    print("Loading", split_path)
    with gzip.open(str(split_path), "rb") as f:
        obj = pickle.load(f)
    observations, next_observations, actions, rewards, dones, seeds = obj.observations, obj.next_observations, \
        obj.actions, obj.rewards, obj.dones, obj.seeds
    if n_envs is not None: 
        observations = observations[:, :n_envs]
        next_observations = next_observations[:, :n_envs] if next_observations is not None else None
        actions = actions[:, :n_envs]
        rewards = rewards[:, :n_envs]
        dones = dones[:, :n_envs]
        seeds = seeds[:, :n_envs]
    return observations, next_observations, actions, rewards, dones, seeds
        

def save_episode(to_save, save_path, save_format="hdf5", compress=False):    
    if save_format == "hdf5":
        compress_kwargs = {"compression": "gzip", "compression_opts": 3} if compress else {}
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
        
        
def save_json_stats(epname_to_len, epname_to_total_returns, epname_to_trjid, epname_to_seed, save_dir): 
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
    with open(save_dir / "episode_seeds.json", "w") as f:
        json.dump(epname_to_seed, f)
    return stats
            

def extract_trajectories_from_single_split(observations, next_observations, actions, rewards, dones, seeds, 
                                           trj_id=0, current_trj=None, add_rtgs=True):
    trajectories = []
    assert len(observations.shape) == 5, "Expected 5D observations containing observations from n parallel envs."
    if current_trj is None: 
        # reinitialize current_trj --> first split 
        current_trj = [collections.defaultdict(list) for _ in range(observations.shape[1])]
    for s, a, r, done, seed in tqdm(zip(observations, actions, rewards, dones, seeds),
                                    total=len(observations), desc="Iterating transitions"):
        nans = [np.isnan(s).any(), np.isnan(a).any(), np.isnan(r).any()]
        if any(nans):
            print("NaNs found:", nans)
        # iterate n_envs
        for i in range(observations.shape[1]):
            current_trj[i]["states"].append(s[i])
            current_trj[i]["actions"].append(a[i])
            current_trj[i]["rewards"].append(r[i])
            current_trj[i]["dones"].append(done[i])
            if done[i]:
                # stack trj
                current_trj[i]["states"] = np.stack(current_trj[i]["states"]).astype(np.uint8)
                current_trj[i]["actions"] = np.stack(current_trj[i]["actions"]).astype(np.uint8)
                current_trj[i]["rewards"] = np.stack(current_trj[i]["rewards"])
                current_trj[i]["dones"] = np.stack(current_trj[i]["dones"]).astype(np.uint8)
                current_trj[i]["trj_id"] = trj_id
                current_trj[i]["seed"] = seed[i]           
                if add_rtgs: 
                    current_trj[i]["returns_to_go"] = discount_cumsum_np(current_trj[i]["rewards"], 1)
                # append, clear
                trajectories.append(current_trj[i])
                current_trj[i] = collections.defaultdict(list)
                trj_id += 1
    
    return trajectories, current_trj, trj_id


def prepare_trajectories_from_buffer_splits(env_name, paths, save_dir, save_format="hdf5", add_rtgs=False, 
                                            max_episodes=None, max_transitions=None, compress=False, n_envs=None):
    """
    Prepares a single procgen dataset for a given environment.

    Trainin data is saved in multiple .gz files containing different trajectory splits over all 25M transitions.
    Trajectories are saved like: 
    ```
    environment family (e.g. procgen)
    - environment name (e.g. bigfish)
    -- one hdf5 file per episode with fields: states, actions, rewards, returns_to_go, dones
    -- episode_lengths.json: dict with episode names as keys and episode lengths as values
    -- episode_returns.json: dict with episode names as keys and episode returns as values
    -- stats.json: dict with stats about the dataset
    ```

    Args:
        env_name: Str. Name of Procgen game.
        max_episodes: Int or None.
        max_transitions: Int or None.
        add_rtgs: Bool. Whether to add returns-to-go to files.
        save_format: Str. File format to save episodes in.

    """
    save_dir = save_dir / env_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # for keeping track current trj ids
    trj_id, num_collected_transitions = 0, 0    
    epname_to_len, epname_to_total_returns, epname_to_trjid, epname_to_seed = {}, {}, {}, {}
    
    # initially None, after first episodes contains remainder of splits, if exists
    current_trj = None
    
    for path in paths: 
        # load split
        try: 
            observations, next_observations, actions, rewards, dones, seeds = load_buffer_split(path, n_envs=n_envs)
        except Exception as e:
            print(f"CAUTION: Error loading {path}.")
            print(e)
            print("Skipping...")
            continue
        
        # extract trajectories from split
        trajectories, current_trj, trj_id = extract_trajectories_from_single_split(
            observations, next_observations, actions, rewards, dones, seeds, trj_id=trj_id, current_trj=current_trj,
            add_rtgs=add_rtgs
        )
        # free ram
        del observations, next_observations, actions, rewards, dones, seeds
        
        # save invidiual episodes 
        for trajectory in tqdm(trajectories, desc=f"Saving split {path.stem}"):

            if max_episodes is not None and trajectory["trj_id"] >= max_episodes:
                print(f"Max episodes reached for {env_name}.")
                break
            if max_transitions is not None and num_collected_transitions + len(trajectory["states"]) >= max_transitions:
                print(f"Max transitions reached for {env_name}.")
                break
            
            # record stats
            file_name = str(trajectory["trj_id"])
            ep_len, ep_total_return = len(trajectory["states"]), trajectory["rewards"].sum()            
            epname_to_len[file_name] = float(ep_len)
            epname_to_total_returns[file_name] = float(ep_total_return)
            epname_to_trjid[file_name] = int(trajectory.pop("trj_id"))
            epname_to_seed[file_name] = int(trajectory.pop("seed"))
            
            # save episode
            save_episode(trajectory, str(save_dir / file_name), save_format=save_format, compress=compress)
            num_collected_transitions += len(trajectory["states"])
    
    stats = save_json_stats(epname_to_len, epname_to_total_returns, epname_to_trjid, epname_to_seed, save_dir)
    return stats


def prepare_procgen_episodes_single(env_name, paths, save_dir, save_format="hdf5", add_rtgs=False, 
                                    max_episodes=None, max_transitions=None, compress=False, to_255=False):
    """
    Prepares a single procgen dataset for a given environment.

    Files are saved as follows:
    ```
    environment family (e.g. procgen)
    - environment name (e.g. bigfish)
    -- one hdf5 file per episode with fields: states, actions, rewards, returns_to_go, dones
    -- episode_lengths.json: dict with episode names as keys and episode lengths as values
    -- episode_returns.json: dict with episode names as keys and episode returns as values
    -- stats.json: dict with stats about the dataset
    ```

    Args:
        env_name: Str. Name of Procgen game.
        max_episodes: Int or None.
        max_transitions: Int or None.
        add_rtgs: Bool. Whether to add returns-to-go to files.
        save_format: Str. File format to save episodes in.

    """
    save_dir = save_dir / env_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # save invidiual episodes 
    epname_to_len, epname_to_total_returns, epname_to_trjid, epname_to_seed = {}, {}, {}, {}
    num_collected_transitions = 0
    for ep_idx, path in enumerate(tqdm(paths, desc="Saving episodes")):
        # load episodes
        episode = load_episode(path, add_rtg=add_rtgs, to_255=to_255)
        if max_episodes is not None and ep_idx >= max_episodes:
            print(f"Max episodes reached for {env_name}.")
            break
        if max_transitions is not None and num_collected_transitions + len(episode["states"]) >= max_transitions:
            print(f"Max transitions reached for {env_name}.")
            break
        
        # record stats
        file_name = path.stem
        ep_len, ep_total_return = len(episode["states"]), episode["rewards"].sum()
        file_name_props = extract_props_from_file_name(file_name)
        if ep_len != file_name_props["trj_len"]:
            print(f"Episode length mismatch {file_name}: {ep_len} vs {file_name_props['trj_len']}.")
        if round(ep_total_return, 2) != file_name_props["total_return"]:
            print(f"Episode return mismatch {file_name}: {ep_total_return} vs {file_name_props['total_return']}.")
        epname_to_len[file_name] = float(ep_len)
        epname_to_total_returns[file_name] = float(ep_total_return)
        epname_to_trjid[file_name] = file_name_props["trj_id"]
        epname_to_seed[file_name] = file_name_props["seed"]
        
        # save episode
        episode.update({"trj_id": file_name_props["trj_id"], "seed": file_name_props["seed"]})
        file_name = path.stem
        save_episode(episode, str(save_dir / file_name), save_format=save_format, compress=compress)
        num_collected_transitions += len(episode["states"])
        
    # extract and save stats
    stats = save_json_stats(epname_to_len, epname_to_total_returns, epname_to_trjid, epname_to_seed, save_dir)
    return stats
        
        
def prepare_procgen_episodes(paths, save_dir, save_format="hdf5", add_rtgs=False, compress=False, 
                             max_episodes=None, max_transitions=None, to_255=False,
                             from_buffer=False, n_envs=None, n_jobs=1): 
    """
    Prepares procgen datasets for all given paths. 

    Args:
        env_name: Str. Dict containing game_name-episode_paths pairs.
        max_episodes: Int or None.
        max_transitions: Int or None.
        add_rtgs: Bool. Whether to add returns-to-go to files.
        save_format: Str. File format to save episodes in.

    """
    all_stats = {}
    if not isinstance(save_dir, pathlib.Path):
        save_dir = pathlib.Path(save_dir)
    
    fn_kwargs = {
        "save_dir": save_dir,
        "max_episodes": max_episodes,
        "max_transitions": max_transitions,
        "save_format": save_format,
        "add_rtgs": add_rtgs,
        "compress": compress,
    }
    if from_buffer:
        fn_kwargs["n_envs"] = n_envs
    else: 
        fn_kwargs["to_255"] = to_255
    prepare_fn = functools.partial(prepare_trajectories_from_buffer_splits, **fn_kwargs) if from_buffer \
        else functools.partial(prepare_procgen_episodes_single, **fn_kwargs)
        
    def call_fn(k, v):
        if len(v) == 0:
            print(f"No episodes found for {k}.")
            return {}
        print(f"Preparing episodes for {k}.")
        if from_buffer: 
            # sort according to buffer split id
            v = sorted(v, key=lambda p: int(str(p.stem).split("_")[1]))
            # remove split 25, as saved twice
            # v = v[:-1] 
        return prepare_fn(k, v)
    
    if n_jobs > 1: 
        ProgressParallel(n_jobs=n_jobs, total=len(paths), timeout=50000)(delayed(call_fn)(k, v) for k, v in paths.items())
    else:
        for k, v in paths.items():
            stats = call_fn(k, v)
            all_stats[k] = stats
        pd.DataFrame(all_stats).round(4).T.to_csv(save_dir / "all_stats.csv")


def make_tarfile(target_dir):
    with tarfile.open(str(target_dir) + ".tar.gz", "w:gz") as f:
        f.add(str(target_dir), arcname=str(target_dir.stem))


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='./procgen/raw')
    parser.add_argument('--save_dir', type=str, 
                        default='./processed')
    parser.add_argument('--max_episodes', type=int, help="Max episodes to use per game.")
    parser.add_argument('--max_transitions', type=int, help="Max transitions to use per game.")
    parser.add_argument('--save_format', type=str, default="hdf5", help="File format to save episodes in.")
    parser.add_argument('--add_rtgs', action="store_true", help="Whether to precompute and add return-to-gos to files.")
    parser.add_argument('--sanity_check', action="store_true", help="Conduct data sanity check.")
    parser.add_argument('--compress', action="store_true", help="Whether to apply compression or not.")
    parser.add_argument('--to_255', action="store_true", help="Whether to convert to pixels values.")
    parser.add_argument('--from_buffer', action="store_true", 
                        help="Whether to construct trajectories from replay buffer.")
    parser.add_argument('--n_envs', type=int, 
        help="Data from first n_envs are extracted from buffer splits. By default stored with 5 parallel envs (2M).")
    parser.add_argument("--make_tar", action="store_true", help="Whether to make a tar file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs.")
    args = parser.parse_args()
    data_dir = pathlib.Path(args.data_dir)
    
    # collect paths
    all_games = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", "fruitbot",
                 "heist", "jumper", "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]
    if args.from_buffer: 
        paths = {game: [p for p in data_dir.rglob(f"**/{game}_*.gz")] for game in all_games}
    else: 
        paths = {game: [p for p in (data_dir / game).rglob("**/*.npy")] for game in all_games}
    
    if args.sanity_check: 
        for game, game_paths in paths.items():
            print(f"Checking {game}.")
            trj_ids = set()
            for p in game_paths: 
                ep_props = extract_props_from_file_name(p.stem)
                if ep_props["trj_id"] in trj_ids: 
                    print(f"Duplicate trajectory id: {p.stem}")
                trj_ids.add(ep_props["trj_id"])
            print(sorted(trj_ids))
    else: 
        # prepare datasets
        prepare_procgen_episodes(paths, args.save_dir, save_format=args.save_format, add_rtgs=args.add_rtgs, 
                                 max_episodes=args.max_episodes, max_transitions=args.max_transitions,
                                 compress=args.compress, to_255=args.to_255, from_buffer=args.from_buffer,
                                 n_envs=args.n_envs, n_jobs=args.n_jobs)
        if args.make_tar:
            print("Constructing .tar file...")
            make_tarfile(pathlib.Path(args.save_dir))
        