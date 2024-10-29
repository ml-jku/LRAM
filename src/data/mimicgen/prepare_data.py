import argparse
import json
import pathlib
import pickle
import h5py
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


MIMICGEN_OBSTYPE_TO_DIM = {
    'object': 86, 'robot0_eef_pos': 3, 'robot0_eef_pos_rel_pod': 3, 'robot0_eef_pos_rel_pod_holder': 3,
    'robot0_eef_quat': 4, 'robot0_eef_quat_rel_pod': 4, 'robot0_eef_quat_rel_pod_holder': 4, 'robot0_eef_vel_ang': 3,
    'robot0_eef_vel_lin': 3, 'robot0_gripper_qpos': 2, 'robot0_gripper_qvel': 2, 'robot0_joint_pos': 7, 
    'robot0_joint_pos_cos': 7, 'robot0_joint_pos_sin': 7, 'robot0_joint_vel': 7, 'robot0_contact': 1, 
    'robot0_eef_force_norm': 1, 'robot0_eef_pos_rel_base': 3, 'robot0_eef_pos_rel_piece_1': 3,
    'robot0_eef_pos_rel_piece_2': 3, 'robot0_eef_quat_rel_base': 4, 'robot0_eef_quat_rel_piece_1': 4,
    'robot0_eef_quat_rel_piece_2': 4
}

MIMICGEN_FULL_OBS_DIM = sum(MIMICGEN_OBSTYPE_TO_DIM.values())

MIMICGEN_OBSTYPE_TO_STARTIDX = {
    'object': 0, 'robot0_eef_pos': 86, 'robot0_eef_pos_rel_pod': 89, 'robot0_eef_pos_rel_pod_holder': 92, 
    'robot0_eef_quat': 95, 'robot0_eef_quat_rel_pod': 99, 'robot0_eef_quat_rel_pod_holder': 103, 
    'robot0_eef_vel_ang': 107, 'robot0_eef_vel_lin': 110, 'robot0_gripper_qpos': 113, 'robot0_gripper_qvel': 115, 
    'robot0_joint_pos': 117, 'robot0_joint_pos_cos': 124, 'robot0_joint_pos_sin': 131, 'robot0_joint_vel': 138, 
    'robot0_contact': 145, 'robot0_eef_force_norm': 146, 'robot0_eef_pos_rel_base': 147,
    'robot0_eef_pos_rel_piece_1': 150, 'robot0_eef_pos_rel_piece_2': 153, 'robot0_eef_quat_rel_base': 156, 
    'robot0_eef_quat_rel_piece_1': 160, 'robot0_eef_quat_rel_piece_2': 164
}

MAIN_LOWDIM_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]


def map_obs_to_full_space(obs):
    batch_size = next(iter(obs.values())).shape[0]
    full_obs_batch = np.zeros((batch_size, MIMICGEN_FULL_OBS_DIM), dtype=np.float32)
    for k, v in obs.items():
        start_idx = MIMICGEN_OBSTYPE_TO_STARTIDX[k]
        v = np.array(v)
        if np.isscalar(v):
            full_obs_batch[:, start_idx:start_idx + 1] = v[:, None]
        else:
            full_obs_batch[:, start_idx:start_idx + v.shape[1]] = v

    return full_obs_batch


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


def extract_properties(demo, compute_rtgs=False, to_full_space=False,
                       low_dim_keys=False, sparse_reward=False, img_key=None, is_failed=False):
    returns_to_go = None
    actions, rewards = demo["actions"][:].astype(np.float32), demo["rewards"][:].astype(np.float32)
    if "dones" in demo.keys():
        dones = demo["dones"][:]
    else: 
        dones = np.zeros_like(rewards)
        dones[-1] = 1
    obs = demo["obs"]
    if sparse_reward:
        rewards = np.zeros_like(rewards)
        if not is_failed: 
            # all are successful demonstrations. 
            rewards[-1] = 1
    if compute_rtgs: 
        returns_to_go = discount_cumsum_np(rewards, 1)
    if img_key is not None: 
        states = obs[img_key][:]
    else: 
        states = extract_states(obs, to_full_space=to_full_space, low_dim_keys=low_dim_keys)
    return {"states": states, "actions": actions, "rewards": rewards, "dones": dones, "returns_to_go": returns_to_go}


def extract_states(obs, return_shapes=False, to_full_space=False, low_dim_keys=False): 
    # sort dict by keys, filter out 
    states_dict = {k: obs[k][:] for k in sorted(obs.keys()) if not len(obs[k].shape) > 2}
    states_dict = {k: np.expand_dims(v, axis=-1) if v.ndim == 1 else v for k, v in states_dict.items()}
    if low_dim_keys:
        states_dict = {k: states_dict[k] for k in MAIN_LOWDIM_KEYS}
    if to_full_space: 
        states = map_obs_to_full_space(states_dict)
    else: 
        states = np.concatenate(list(states_dict.values()), axis=1).astype(np.float32)
    if return_shapes: 
        shapes = {k: v.shape for k, v in states_dict.items()}
        return states, shapes
    return states


def resize_img_episode(frames: np.ndarray, dim: int = 64) -> np.ndarray:
    # cv2 only supports processing single frame - loop over frames
    processed_frames = []
    for i in range(frames.shape[0]):
        frame = frames[i].astype(np.uint8)
        frame = cv2.resize(frame, (dim, dim), interpolation=cv2.INTER_AREA)
        frame = frame.transpose(2, 0, 1)
        processed_frames.append(frame)
    processed_frames = np.stack(processed_frames)
    return processed_frames


def prepare_episodes_single(path, save_dir, save_format="hdf5", add_rtgs=False, 
                            compress=False, to_full_space=False, low_dim_keys=False, sparse_reward=False,
                            img_key=None, crop_dim=84, max_failed=100):
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
    dir_name = path.stem
    is_failed = dir_name == "demo_failed"
    if dir_name == "demo" or dir_name == "demo_failed":
        # prepare path for self generated data
        dir_name = path.parent.stem
        dir_name = dir_name.replace("demo_src_", "").replace("_task", "")
        if is_failed: 
            dir_name = f"{str(dir_name)}_failed"
    save_dir = save_dir / dir_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_collected_transitions, trj_id = 0, 0
    epname_to_len, epname_to_total_returns, epname_to_trjid = {}, {}, {}

    # .hdf5s contain {"data": {"demo0": {}, "demo1": {}, ...}}
    with h5py.File(path, "r") as f:
        # save invidiual episodes 
        if len(f["data"]) == 0:
            print(f"Skipping {path} as it contains no episodes.")
            return {}
        for k in tqdm(f["data"], desc="Saving episodes"):
            demo = f["data"][k]
            to_save = extract_properties(demo, compute_rtgs=add_rtgs, to_full_space=to_full_space, 
                                         img_key=img_key, low_dim_keys=low_dim_keys, sparse_reward=sparse_reward,
                                         is_failed=is_failed)
            
            if img_key is not None and crop_dim != 84:
                to_save["states"] = resize_img_episode(to_save["states"], dim=crop_dim)
            
            # record stats
            file_name = str(trj_id)
            ep_len, ep_total_return = len(to_save["states"]), to_save["rewards"].sum()
            epname_to_len[file_name] = float(ep_len)
            epname_to_total_returns[file_name] = float(ep_total_return)
            epname_to_trjid[file_name] = trj_id
            
            # save episode
            save_episode(to_save, str(save_dir / file_name), save_format=save_format, compress=compress)
            num_collected_transitions += len(to_save["states"])
            trj_id += 1
            
    # extract and save stats
    stats = save_json_stats(epname_to_len, epname_to_total_returns, epname_to_trjid, save_dir)
    return stats


def prepare_episodes(data_dir, save_dir, save_format="hdf5", add_rtgs=False, compress=False,
                     to_full_space=False, low_dim_keys=False, sparse_reward=False, img_key=None, crop_dim=84,
                     max_failed=100, search_pattern=None, exclude_gripper=False): 
    """
    Prepares mimicgen datasets for all given paths. Write a single .hdf5 file per trajectory
    and otherwise keeps the trajectory structure the same.  

    Args:
        data_dir: Str. 
        save_dir: Str. Path to save episodes to.
        add_rtgs: Bool. Whether to add returns-to-go to files.
        save_format: Str. File format to save episodes in.
        compress: Bool. Whether to apply compression or not.
        img_key: Str. Key to extract images from dataset. Default is None.
        to_full_space: Bool. Whether to convert to full/unified state space.
    """
    all_stats = {}
    if not isinstance(save_dir, pathlib.Path):
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    if search_pattern is None:
        search_pattern = "**/*.hdf5"
    # collect paths
    paths = [p for p in pathlib.Path(data_dir).rglob(search_pattern)]
    if exclude_gripper:
        print("Excluding trajectories with specified gripper.")
        paths = [p for p in paths if "_gripper_" not in str(p)]
    print(paths)
    for p in paths:
        print(f"Preparing episodes for {p}.")
        stats = prepare_episodes_single(
            p,
            save_dir=save_dir, 
            save_format=save_format,
            add_rtgs=add_rtgs,
            compress=compress,
            to_full_space=to_full_space, 
            img_key=img_key,
            crop_dim=crop_dim,
            low_dim_keys=low_dim_keys,
            sparse_reward=sparse_reward,
            max_failed=max_failed
        )
        dir_name = p.stem
        if dir_name == "demo" or dir_name == "demo_failed":
            # prepare path for self generated data
            dir_name = p.parent.stem
            dir_name = dir_name.replace("demo_src_", "").replace("_task", "")
            if dir_name == "demo_failed":
                dir_name = f"{str(dir_name)}_failed"
        all_stats[dir_name] = stats
    pd.DataFrame(all_stats).round(4).T.to_csv(save_dir / "all_stats.csv")
    
    
def extract_shapes(data_dir):
    paths = sorted([p for p in pathlib.Path(data_dir).rglob("**/*.hdf5")])
    all_shapes = {}
    for p in paths: 
        with h5py.File(p, "r") as f:
            # extract shapes
            demo = f["data"]["demo_0"]
            states, shapes = extract_states(demo["obs"], return_shapes=True)
            shapes = {k: v[-1] for k, v in shapes.items()}
            all_shapes[p.stem] = {"states": states.shape[-1], 
                                  "actions": demo["actions"][:].shape[-1], "obs": shapes}
    with open('data_shapes.json', 'w') as f:
        json.dump(all_shapes, f, indent=4)
    

def extract_shapes_and_stats_from_processed(save_dir):
    paths = sorted([p for p in pathlib.Path(save_dir).rglob("**/0.hdf5")])
    all_shapes = {}
    all_stats = {}
    for p in paths: 
        with h5py.File(p, "r") as f:
            states, actions = f["states"][:], f["actions"][:]
            all_shapes[p.parent.stem] = {"states":  states.shape[-1], "actions": actions.shape[-1]}
            all_stats[p.parent.stem] = {
                "states": (float(states.min()), float(states.max()), float(states.mean()), float(states.std())),
                "actions": (actions.min(0), actions.max(0), actions.mean(0), actions.std(0))
            }
    with open('data_processed_shapes.json', 'w') as f:
        json.dump(all_shapes, f, indent=4)
    for k, v in all_stats.items():
        all_stats[k]["actions"] = [v2.round(2).tolist() for v2 in v["actions"]]
    with open('data_processed_stats.json', 'w') as f:
        json.dump(all_stats, f, indent=4)
        
        
def extract_metadata(data_dir): 
    from robomimic.utils.file_utils import get_env_metadata_from_dataset
    paths = sorted([p for p in pathlib.Path(data_dir).rglob("**/*.hdf5")])
    meta = {}
    for p in paths: 
        env_meta = get_env_metadata_from_dataset(p)
        meta[p.stem] = env_meta
    with open('metadata.json', 'w') as f:
        json.dump(meta, f, indent=4)
        

def visualize_trajectories(data_dir, save_dir="./figures"): 
    paths = sorted([p for p in pathlib.Path(data_dir).rglob("**/*.hdf5")])
    for p in paths: 
        with h5py.File(p, "r") as f:
            obs = f["data"]["demo_0"]["obs"]
            for key in obs.keys():
                if len(obs[key].shape) == 4: 
                    # is image
                    images = obs[key]
                    img_save_dir = Path(save_dir) / p.stem / key
                    img_save_dir.mkdir(parents=True, exist_ok=True)
                    for i, img in enumerate(images):
                        if i % 20 == 0 or i == len(images) - 1:  
                            plt.imshow(img)
                            plt.savefig(img_save_dir / f"{i}.png")
                            plt.close()
            print(f"Rewards for {p.stem}: {f['data']['demo_0']['rewards'][:]}")
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='./data/mimicgen/core')
    parser.add_argument('--save_dir', type=str, 
                        default='./data/mimicgen/core_processed')
    parser.add_argument('--save_format', type=str, default="hdf5", help="File format to save episodes in.")
    parser.add_argument('--hdf5_pattern', type=str, help="Str. Search pattern for hdf5 files.")
    parser.add_argument('--img_key', type=str, help="Img key to extract from dataset. Default is None.")
    parser.add_argument('--crop_dim', type=int, default=84, help="Crop dimension for state-images. Default is 84x84.")
    parser.add_argument('--max_failed', type=int, default=25, help="Max num of failed trjs to include.")
    parser.add_argument('--add_rtgs', action="store_true", help="Whether to precompute and add return-to-gos to files.")
    parser.add_argument('--compress', action="store_true", help="Whether to apply compression or not.")
    parser.add_argument('--to_full_space', action="store_true", help="Covert to full/unified state space.")
    parser.add_argument('--low_dim_keys', action="store_true", help="Use only low-dim keys for observations.")
    parser.add_argument('--sparse_reward', action="store_true", help="Use binary reward; success/fail. Here all are 1s.")
    parser.add_argument('--exclude_gripper', action="store_true")
    parser.add_argument('--shapes_only', action="store_true", help="Conduct shape.")
    parser.add_argument('--shapes_and_stats_only', action="store_true")
    parser.add_argument('--metadata_only', action="store_true")
    parser.add_argument('--vis_trjs', action="store_true")
    args = parser.parse_args()
    
    if args.shapes_only:
        extract_shapes(args.data_dir)
    elif args.shapes_and_stats_only:
        extract_shapes_and_stats_from_processed(args.save_dir)
    elif args.metadata_only:
        extract_metadata(args.data_dir)
    elif args.vis_trjs: 
        visualize_trajectories(args.data_dir)
    else: 
        prepare_episodes(args.data_dir, args.save_dir, save_format=args.save_format, add_rtgs=args.add_rtgs, 
                         compress=args.compress, to_full_space=args.to_full_space, img_key=args.img_key, 
                         crop_dim=args.crop_dim, low_dim_keys=args.low_dim_keys, sparse_reward=args.sparse_reward,
                         search_pattern=args.hdf5_pattern, max_failed=args.max_failed, exclude_gripper=args.exclude_gripper)
