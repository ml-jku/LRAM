import pickle
import collections
import numpy as np
import argparse
import json
import hydra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from stable_baselines3.common.buffers import ReplayBuffer
import sys
sys.path.append("../../")


def extract_trajectories_from_buffer(buffer):
    """

    From trajectory_buffer.py
    Args:
        buffer: ReplayBuffer object.

    Returns: list of individual trajectories.

    """
    trajectories = []
    current_trj = collections.defaultdict(list)
    pos = buffer.pos if not buffer.full else len(buffer.observations)
    for s, s1, a, r, done in tqdm(zip(buffer.observations[:pos], buffer.next_observations[:pos],
                                      buffer.actions[:pos], buffer.rewards[:pos], buffer.dones[:pos]),
                                  total=pos, desc="Extracting trajectories"):
        nans = [np.isnan(s).any(), np.isnan(s1).any(), np.isnan(a).any(), np.isnan(r)]
        if any(nans):
            print("NaNs found:", nans)
        current_trj["observations"].append(s)
        current_trj["next_observations"].append(s1)
        current_trj["actions"].append(a)
        current_trj["rewards"].append(r)
        current_trj["terminals"].append(done)
        if done:
            trajectories.append(current_trj)
            current_trj = collections.defaultdict(list)
    return trajectories


def extract_trajectories_from_npz(obj):
    observations, next_observations, actions, rewards, dones = obj["observations"], obj["next_observations"],\
        obj["actions"], obj["rewards"], obj["dones"]
    trajectories = extract_trajectories(observations, next_observations, actions, rewards, dones)
    return trajectories


def extract_trajectories(observations, next_observations, actions, rewards, dones):
    trajectories = []
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
            trajectories.append(current_trj)
            current_trj = collections.defaultdict(list)
    return trajectories


def extract_returns(trajectories):
    return [np.array(trj["rewards"]).sum().item() for trj in trajectories]


def extract_array_stats(trajectories, kind="actions"):
    if kind == "len": 
        vals = np.array([len(trj["observations"]) for trj in trajectories])
    else: 
        vals = np.concatenate([trj[kind] for trj in trajectories])
    stats = {
        "min": np.min(vals),
        "max": np.max(vals),
        "mean": np.mean(vals),
        "std": np.std(vals),
        "q25": np.quantile(vals, 0.25),
        "q50": np.quantile(vals, 0.5),
        "q75": np.quantile(vals, 0.75),
        "q90": np.quantile(vals, 0.9),
        "q99": np.quantile(vals, 0.99),
    }
    return stats


def plot_distribution(values, kind="hist", task_name="test", xlabel="Returns", save_dir=None, size=None, palette=None, orient=None,
                      fontsize=None, alpha=None, edgecolor=None, plot_kwargs=None, fname=None):
    sns.set_style("whitegrid")
    if save_dir and not isinstance(save_dir, Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    if palette is not None:
        sns.set_palette(palette)
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    if kind == "hist":
        plot_kwargs.update({"kde": True})
        plot_fn = sns.histplot
    elif kind == "count":
        plot_fn = sns.countplot
    else:
        raise ValueError(f"Unknown kind: {kind}")
    ax = plot_fn(values, **plot_kwargs)
    plt.title(task_name, fontsize=fontsize)
    plt.xlabel(xlabel)
    if size is not None:
        plt.gcf().set_size_inches(size)
    if orient is not None:
        ax.set_orientation(orient)
    if alpha is not None:
        for container in ax.containers:
            plt.setp(container, alpha=alpha)
    if edgecolor is not None:
        for container in ax.containers:
            plt.setp(container, edgecolor=edgecolor)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{task_name if fname is None else fname}.png", bbox_inches='tight')
    else: 
        plt.show()
    plt.close()


def compute_trj_quality_stats(returns):
    returns = np.array(sorted(returns))
    min_return, max_return = returns.min(), returns.max()
    trj_qualities = (returns - min_return) / ((max_return - min_return) + 1e-8)
    stats = pd.DataFrame({"trj_quality": trj_qualities}).describe().to_dict()
    stats = pd.json_normalize(stats).to_dict(orient="records")[0]
    return stats


def extract_return_stats(trajectories, target_multiplier, save_dir=None, task_name=None, dataset_name=None):
    rewards = np.concatenate([trj["rewards"] for trj in trajectories]).reshape(-1)
    returns = extract_returns(trajectories)
    # compute stats
    reward_stats = pd.DataFrame({"rewards": rewards}).describe()
    return_stats = pd.DataFrame({"returns": returns}).describe()
    stats = pd.concat([reward_stats, return_stats], axis=1).to_dict()
    stats = pd.json_normalize(stats).to_dict(orient="records")[0]
    # compute max return/reward
    max_return, max_reward = max(returns), max(rewards)
    max_return = max_return * target_multiplier if max_return > 0 else max_return / target_multiplier
    max_reward = max_reward * target_multiplier if max_reward > 0 else max_reward / target_multiplier
    # compute trajectory quality
    stats.update(compute_trj_quality_stats(returns))
    
    # plot distributions
    if save_dir is not None: 
        save_dir = Path(save_dir) / dataset_name /  "distributions"        
        # plot_distribution(rewards, kind="hist", task_name=task_name, save_dir=save_dir / "rewards")
        plot_distribution(returns, kind="hist", task_name=task_name, save_dir=save_dir / "returns")
    return stats, max_return, max_reward, rewards, returns


def extract_stats(paths, target_multiplier=1, first_p_trjs=1, stats_per_dim=False, save_dir=None, name=None):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    all_r_stats, max_return_per_task, max_reward_per_task = {}, {}, {}
    all_a_stats, all_s_stats, all_len_stats = {}, {}, {}
    all_rewards, all_returns = [], []
    all_s_stats_per_dim = collections.defaultdict(dict)
    all_a_stats_per_dim = collections.defaultdict(dict)
    dataset_name = Path(paths[0]).parts[-2]
    for path in paths:
        print(f"Loading trajectories from: {path}")
        path = Path(path)
        task_name = path.stem
        if path.suffix == ".pkl":
            with open(str(path), "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, ReplayBuffer):
                trajectories = extract_trajectories_from_buffer(obj)
            else:
                trajectories = obj
        elif path.suffix == ".npz" or path.suffix == ".npy":
            obj = np.load(str(path))
            trajectories = extract_trajectories_from_npz(obj)
        elif path.is_dir():
            print("test")
            
        if first_p_trjs < 1:
            trajectories = trajectories[:int(first_p_trjs * len(trajectories))]

        # extract rewards, returns stats from trjs
        r_stats, max_return, max_reward, rewards, returns = extract_return_stats(
            trajectories, target_multiplier, save_dir=save_dir, task_name=task_name, dataset_name=dataset_name
        )
        all_rewards.append(rewards)
        all_returns.append(returns)
        all_r_stats[task_name] = r_stats
        max_return_per_task[task_name] = max_return
        max_reward_per_task[task_name] = max_reward

        # extract state/action stats from trjs
        all_s_stats[task_name] = extract_array_stats(trajectories, "observations")
        all_a_stats[task_name] = extract_array_stats(trajectories, "actions")
        all_len_stats[task_name] = extract_array_stats(trajectories, "len")
        if stats_per_dim: 
            all_obs = np.vstack([np.vstack(trj["observations"]) for trj in trajectories])
            all_acts = np.vstack([np.vstack(trj["actions"]) for trj in trajectories])
            all_s_stats_per_dim[task_name]["mean"] = list(all_obs.mean(0))
            all_s_stats_per_dim[task_name]["std"] = list(all_obs.std(0))
            all_a_stats_per_dim[task_name]["mean"] = list(all_acts.mean(0))
            all_a_stats_per_dim[task_name]["std"] = list(all_acts.std(0))
            
    if save_dir is not None:
        save_dir = Path(save_dir) / dataset_name
        if first_p_trjs < 1:
            save_dir = Path(save_dir) / f"first_p_trjs_{first_p_trjs}"
        save_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_r_stats).round(4).T.to_csv(save_dir / "r_stats.csv")
        pd.DataFrame(all_a_stats).round(4).T.to_csv(save_dir / "a_stats.csv")
        pd.DataFrame(all_s_stats).round(4).T.to_csv(save_dir / "s_stats.csv")
        pd.DataFrame(all_len_stats).round(4).T.to_csv(save_dir / "len_stats.csv")
        if stats_per_dim: 
            pd.DataFrame(dict(all_s_stats_per_dim)).round(4).T.to_csv(save_dir / "s_stats_per_dim.csv")
            pd.DataFrame(dict(all_a_stats_per_dim)).round(4).T.to_csv(save_dir / "a_stats_per_dim.csv")
            print("Stats per dim:")
            print(dict(all_s_stats_per_dim))
            print(dict(all_a_stats_per_dim))
            
        with open(save_dir / "max_returns.json", "w") as f:
            json.dump(max_return_per_task, f, indent=4, sort_keys=False)
        with open(save_dir / "max_rewards.json", "w") as f:
            json.dump(max_reward_per_task, f, indent=4, sort_keys=False)
            
        all_rewards = np.concatenate(all_rewards)
        all_returns = np.concatenate(all_returns)
        # plot_distribution(all_rewards, kind="hist", task_name="Rewards", save_dir=save_dir)
        plot_distribution(all_returns, kind="hist", 
                          task_name="Returns" if name is None else name,
                          fname="Returns",
                          save_dir=save_dir)

    return max_return_per_task, max_reward_per_task


def extract_stats_from_dirs(paths, save_dir=None, name=None):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    dataset_name = Path(paths[0]).parts[-2] if name is None else name
    df_stats = dict()
    for path in paths:
        print(f"Loading trajectories from: {path}")
        path = Path(path)
        task_name = path.stem
        assert path.is_dir(), "Only directories are supported for now."
        with open(path / "stats.json", 'r') as f:
            stats = json.load(f)
        df_stats[task_name] = stats
    df_stats = pd.DataFrame(df_stats).round(1).T          
    if save_dir is not None:
        save_dir = Path(save_dir) / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        df_stats.to_csv(save_dir / "stats.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_paths", default='dmcontrol11_icl.yaml')
    parser.add_argument("--save_dir", default='../../postprocessing/data_stats')
    parser.add_argument("--target_multiplier", default=1, type=float)
    parser.add_argument("--first_p_trjs", default=1, type=float)
    parser.add_argument("--stats_per_dim", action="store_true")
    parser.add_argument("--name", type=str)
    parser.add_argument("--is_dir", action="store_true")
    args = parser.parse_args()
    hydra.initialize(config_path="../../configs")
    conf = hydra.compose(config_name="config",
                         overrides=["agent_params=cdt_clusterembeds",
                                    f"agent_params/data_paths={args.data_paths}"])
    base_path, names = conf.agent_params.data_paths["base"], conf.agent_params.data_paths["names"]
    names = list(names) if not isinstance(names, list) else names
    paths = [str(Path(base_path) / name) for name in names]
    
    if args.is_dir: 
        extract_stats_from_dirs(paths, save_dir=args.save_dir, name=args.name)
    else: 
        max_return_per_task, max_reward_per_task = extract_stats(
            paths, target_multiplier=args.target_multiplier, save_dir=args.save_dir,
            first_p_trjs=args.first_p_trjs, name=args.name, stats_per_dim=args.stats_per_dim
        )
        print(max_return_per_task)
        print(max_reward_per_task)
