"""
The DQN Replay Dataset was produced with the reduced Atari action set for each game. 
To train the agent across all games, the action space needs to be unified. 

We adjust the code from: 
- https://github.com/google-research/google-research/blob/master/multi_game_dt/Multi_game_decision_transformers_public_colab.ipynb

for mapping from the reduced action space to the full action space.

"""
import random
import pathlib
import argparse
import json
import cv2
import numpy as np
import d4rl_atari
import gym
import d3rlpy
import numpy as np
import h5py
import pickle
import functools
import joblib
from joblib import delayed
from tqdm import tqdm



ATARI_GAMES = [
    'air-raid', 'alien', 'amidar', 'assault', 'asterix',
    'asteroids', 'atlantis', 'bank-heist', 'battle-zone', 'beam-rider',
    'berzerk', 'bowling', 'boxing', 'breakout', 'carnival', 'centipede',
    'chopper-command', 'crazy-climber', 'demon-attack',
    'double-dunk', 'elevator-action', 'enduro', 'fishing-derby', 'freeway',
    'frostbite', 'gopher', 'gravitar', 'hero', 'ice-hockey', 'jamesbond',
    'journey-escape', 'kangaroo', 'krull', 'kung-fu-master',
    'montezuma-revenge', 'ms-pacman', 'name-this-game', 'phoenix',
    'pitfall', 'pong', 'pooyan', 'private-eye', 'qbert', 'riverraid',
    'road-runner', 'robotank', 'seaquest', 'skiing', 'solaris',
    'space-invaders', 'star-gunner', 'tennis', 'time-pilot', 'tutankham',
    'up-n-down', 'venture', 'video-pinball', 'wizard-of-wor',
    'yars-revenge', 'zaxxon'
]


_FULL_ACTION_SET = [
    'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
    'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
    'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
]

_LIMITED_ACTION_SET = {
    "air-raid": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "alien": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "amidar": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE"
    ],
    "assault": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "asterix": [
        "NOOP",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT"
    ],
    "asteroids": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE"
    ],
    "atlantis": [
        "NOOP",
        "FIRE",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "bank-heist": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "battle-zone": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "beam-rider": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "UPRIGHT",
        "UPLEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "berzerk": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "bowling": [
        "NOOP",
        "FIRE",
        "UP",
        "DOWN",
        "UPFIRE",
        "DOWNFIRE"
    ],
    "boxing": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "breakout": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT"
    ],
    "carnival": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "centipede": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "chopper-command": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "crazy-climber": [
        "NOOP",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT"
    ],
    "demon-attack": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "double-dunk": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "elevator-action": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "enduro": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "DOWN",
        "DOWNRIGHT",
        "DOWNLEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "fishing-derby": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "freeway": [
        "NOOP",
        "UP",
        "DOWN"
    ],
    "frostbite": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "gopher": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "gravitar": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "hero": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "ice-hockey": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "jamesbond": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "journey-escape": [
        "NOOP",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "kangaroo": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "krull": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "kung-fu-master": [
        "NOOP",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "DOWNRIGHT",
        "DOWNLEFT",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "montezuma-revenge": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "ms-pacman": [
        "NOOP",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT"
    ],
    "name-this-game": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "phoenix": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "DOWN",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE"
    ],
    "pitfall": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "pong": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "pooyan": [
        "NOOP",
        "FIRE",
        "UP",
        "DOWN",
        "UPFIRE",
        "DOWNFIRE"
    ],
    "private-eye": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "qbert": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN"
    ],
    "riverraid": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "road-runner": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "robotank": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "seaquest": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "skiing": [
        "NOOP",
        "RIGHT",
        "LEFT"
    ],
    "solaris": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "space-invaders": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "star-gunner": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "tennis": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "time-pilot": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE"
    ],
    "tutankham": [
        "NOOP",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "up-n-down": [
        "NOOP",
        "FIRE",
        "UP",
        "DOWN",
        "UPFIRE",
        "DOWNFIRE"
    ],
    "venture": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "video-pinball": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "wizard-of-wor": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE"
    ],
    "yars-revenge": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "zaxxon": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ]
}

# An array that Converts an action from a game-specific to full action set.
LIMITED_ACTION_TO_FULL_ACTION = {
    game_name: np.array(
        [_FULL_ACTION_SET.index(i) for i in _LIMITED_ACTION_SET[game_name]])
    for game_name in ATARI_GAMES
}

# An array that Converts an action from a full action set to a game-specific
# action set (Setting 0=NOOP if no game-specific action exists).
FULL_ACTION_TO_LIMITED_ACTION = {
    game_name: np.array([(_LIMITED_ACTION_SET[game_name].index(i)
                          if i in _LIMITED_ACTION_SET[game_name] else 0)
                         for i in _FULL_ACTION_SET]) for game_name in ATARI_GAMES
}


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


def discount_cumsum_np(x, gamma):
    # much faster version of the above
    new_x = np.zeros_like(x)
    rev_cumsum = np.cumsum(np.flip(x, 0)) 
    new_x = np.flip(rev_cumsum * gamma ** np.arange(0, x.shape[0]), 0)
    new_x = np.ascontiguousarray(new_x).astype(np.float32)
    return new_x


def resize_img_episode(frames: np.ndarray, dim: int = 64) -> np.ndarray:
    # cv2 only supports processing single frame - loop over frames
    processed_frames = []
    for i in range(frames.shape[0]):
        frame = frames[i, 0].astype(np.uint8)
        frame = cv2.resize(frame, (dim, dim), interpolation=cv2.INTER_AREA)
        frame = frame[None, :, :]
        processed_frames.append(frame)
    processed_frames = np.stack(processed_frames)
    return processed_frames


def save_episodes(episodes, save_dir, start_idx=0, save_format="hdf5", a_to_full_space=False, add_rtgs=False,
                  to_rgb=False, max_episodes=None, max_transitions=None, num_collected_transitions=None,
                  env_name=None, crop_dim=84):
    """
    Saves episodes and optionally given stats dict to desired save_dir.
    Args:
        episodes: List of d3rlpy.dataset.Episode objects.
        save_dir: pathlib.Path. Desired location to save individual episodes.
        stats: None or dict.
        save_format: Str.
        a_to_full_space: Bool.

    """    
    print(f"Saving episodes to {str(save_dir)}.")
    ep_lengths = {}
    ep_total_rewards = {}
    for i, episode in enumerate(tqdm(episodes, desc="Saving episodes")):
        ep_idx = start_idx + i
        if max_episodes is not None and ep_idx > max_episodes:
            break
        if max_transitions is not None and num_collected_transitions + len(episode) > max_transitions:
            break
        actions = episode.actions
        if a_to_full_space:
            # convert from limited action space to full, 18 dimensional action space
            assert env_name is not None
            actions = LIMITED_ACTION_TO_FULL_ACTION[env_name][actions]
        to_save = {
            "states": episode.observations,
            "actions": actions,
            "rewards": episode.rewards,
            "dones": episode.terminal
        }
        if add_rtgs: 
            rtgs = discount_cumsum_np(episode.rewards, 1)
            to_save["returns_to_go"] = rtgs
        if crop_dim != 84:
            to_save["states"] = resize_img_episode(to_save["states"], crop_dim)
        if to_rgb: 
            # i.e., repeat the grayscale channel 3 times to get "RGB"
            to_save["states"] = np.repeat(to_save["states"], 3, axis=1)
        
        file_name = str(start_idx + i).zfill(9)
        save_path = str(save_dir / file_name)
        ep_lengths[file_name] = len(episode.observations)
        ep_total_rewards[file_name] = sum(episode.rewards)
        if save_format == "hdf5":
            compress_kwargs = {"compression": "gzip", "compression_opts": 1}
            with h5py.File(save_path + ".hdf5", "w") as f:
                f.create_dataset('states', data=to_save["states"], **compress_kwargs)
                f.create_dataset('actions', data=to_save["actions"], **compress_kwargs)
                f.create_dataset('rewards', data=to_save["rewards"], **compress_kwargs)
                f.create_dataset('dones', data=to_save["dones"])
                if add_rtgs: 
                    f.create_dataset('returns_to_go', data=to_save["returns_to_go"], **compress_kwargs)
        elif save_format == "npzc": 
            np.savez_compressed(save_path, **to_save)
        elif save_format == "pkl": 
            with open(save_path + ".pkl", "wb") as f:
                pickle.dump(to_save, f)
        else: 
            np.savez(save_path, **to_save)
    return ep_lengths, ep_total_rewards


def infer_slice_indices(slice_indices, num_slices, quality):
    random.seed(0)
    assert quality in ["mixed", "random", "expert"]
    if quality == "mixed":
        return random.sample(slice_indices, num_slices)
    elif quality == "random":
        return slice_indices[:num_slices]
    elif quality == "expert":
        return reversed(slice_indices[-num_slices:])


def load_and_save_atari_episodes(env_name, save_dir, index=0, num_slices=50, quality="mixed", save_format="hdf5",
                                 a_to_full_space=False, add_rtgs=False, to_rgb=False, download_only=False,
                                 max_episodes=None, max_transitions=None, crop_dim=84):
    """
    Downloads the desired number of slices for the specified Atari game (if does not exist) from
    the DQN Replay Dataset.
        - https://research.google/tools/datasets/dqn-replay/
    In total there are 50 slices per game, each containing 1M transitions (state, action, reward, next state).

    After loading a slice, the episodes are saved to the given save_dir.
    To avoid running into memory issues, slices are loaded & saved sequentially.

    Files are saved as follows:
    ```
    environment family (e.g. atari)
    - environment name (e.g. breakout)
    -- one npz file (numbered, zero padded to 9 digits) per episode with fields: states, actions, rewards, dones

    ```

    Args:
        env_name: Str. Name of Atari game.
        index: Int.
        num_slices: Int. Number of slices to download. If <50, random slices in range [0,50] are selected.
        quality: Str. One of ["mixed", "random", "expert"]. If "mixed", random slices are selected.
        max_episodes: Int or None.
        max_transitions: Int or None.
        a_to_full_space: Bool. If True, actions are remapped to full action space.
        to_rgb: Bool. If True, states are converted to RGB.
        crop_dim: Int.

    Returns: list of episodes.

    """
    # do d4rl_atari import here, to ensure envs are visible to worker processes
    import d4rl_atari
    
    num_collected_transitions, num_collected_episodes = 0, 0
    all_epname_to_len, all_epname_to_total_rewards = {}, {}
    ep_lens, ep_returns = [], []
    slice_indices = range(0, 50)
    if num_slices < 50:
        slice_indices = infer_slice_indices(slice_indices, num_slices, quality)
    slice_indices = list(slice_indices)
    print(f"Collecting slices with indices: {slice_indices}")

    # collect and save episodes
    for i in tqdm(slice_indices, desc="Loading slices"):
        # load slice
        try: 
            env = gym.make(f"{env_name}-epoch-{i + 1}-v{index}", sticky_action=True)
            dataset = d3rlpy.dataset.MDPDataset(discrete_action=True, **env.get_dataset())
        except Exception as e:
            print(e)
            print("Slice does not exist, continuing to next slice.") 
            continue
        if download_only: 
            continue
        episodes = list(dataset.episodes)
        # save invidiual episodes 
        epname_to_len, epname_to_total_rewards = save_episodes(
            episodes, save_dir, start_idx=num_collected_episodes, max_episodes=max_episodes,
            num_collected_transitions=num_collected_transitions, max_transitions=max_transitions,
            a_to_full_space=a_to_full_space, env_name=env_name, save_format=save_format, add_rtgs=add_rtgs, 
            to_rgb=to_rgb, crop_dim=crop_dim
        )
        # store episode lengths 
        all_epname_to_len.update(epname_to_len)
        all_epname_to_total_rewards.update(epname_to_total_rewards)
        num_collected_episodes += len(epname_to_len.keys())
        num_collected_transitions += sum([len(ep) for ep in episodes])
        ep_lens += [v for v in epname_to_len.values()]
        ep_returns += [ep.rewards.sum() for ep in episodes]
        if max_episodes is not None and num_collected_episodes > max_episodes:
            print("Max number of episodes reached.")
            break
        if max_transitions is not None and num_collected_transitions > max_transitions:
            print("Max number of transitions reached.")
            break 
    
    if download_only: 
        return

    # compute and dumpy episode stats
    stats = {
        "slices": num_slices, "episodes": num_collected_episodes, "transitions": num_collected_transitions,
        "mean_episode_len": np.mean(ep_lens).round(decimals=4).item(),
        "min_episode_len": np.min(ep_lens).round(decimals=4).item(),
        "max_episode_len": np.max(ep_lens).round(decimals=4).item(),
        "mean_episode_return": np.mean(ep_returns).round(decimals=4).item(),
        "min_episode_return": np.min(ep_returns).round(decimals=4).item(),
        "max_episode_return": np.max(ep_returns).round(decimals=4).item()
    }
    print(" | ".join([f"{k}: {v}" for k, v in stats.items()]))
    with open(save_dir / "stats.json", "w") as f:
        json.dump(stats, f)
    with open(save_dir / "episode_lengths.json", "w") as f:
        json.dump(all_epname_to_len, f)
    with open(save_dir / "episode_returns.json", "w") as f:
        json.dump(all_epname_to_total_rewards, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', nargs='+', default=['breakout'])
    parser.add_argument('--save_dir', type=str, default='DATADIR')

    parser.add_argument('--num_slices', type=int, default=50,
                        help="Num of slices to download/save. In total there are 50 slices (each 1M steps) per game.")
    parser.add_argument('--quality', type=str, default="mixed",
                        help="Trajectory quality, only used if num_slices < 50. "
                             "Either 'mixed' (randomly selects slices), 'random' (selectes slices 1 to num_slices) or "
                             "'expert' (selects slices 50 - num_slices to 50). ")
    parser.add_argument('--max_episodes', type=int, help="Max episodes to use per game.")
    parser.add_argument('--max_transitions', type=int, help="Max transitions to use per game.")
    parser.add_argument('--a_to_full_space', action="store_true", help="Whether actions should be remapped to full action space.")
    parser.add_argument('--save_format', type=str, default="hdf5", help="File format to save episodes in.")
    parser.add_argument('--add_rtgs', action="store_true", help="Wheter to precompute and add return-to-gos to files.")
    parser.add_argument('--to_rgb', action="store_true", 
                        help="Wheter to convert the gray-scale images to RGB format by repeating the grayscale channel 4 times")
    parser.add_argument('--crop_dim', type=int, default=84,
                        help="Crop dimension for state-images. Default is 84x84.")
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--download_only', action="store_true", 
                        help="Whether to only download checkpoints. This is useful for n_jobs, as downloading causes issuesin multiprocessing.")
    args = parser.parse_args()

    envs = args.envs if "all" not in args.envs else ATARI_GAMES
    constant_kwargs = {
        "num_slices": args.num_slices,
        "quality": args.quality,
        "max_episodes": args.max_episodes,
        "max_transitions": args.max_transitions,
        "a_to_full_space": args.a_to_full_space,
        "save_format": args.save_format,
        "add_rtgs": args.add_rtgs,
        "to_rgb": args.to_rgb,
        "crop_dim": args.crop_dim,
        "download_only": args.download_only
    }
    
    fn = functools.partial(load_and_save_atari_episodes, **constant_kwargs)
    def call_fn(env): 
        save_dir = pathlib.Path(args.save_dir) / env
        save_dir.mkdir(exist_ok=True, parents=True)
        fn(env_name=env, save_dir=save_dir, index=1 if env == "asterix" else 0)    
    
    if args.n_jobs == 1: 
        for env in envs:
            call_fn(env)
    else: 
        ProgressParallel(n_jobs=args.n_jobs, total=len(envs), timeout=5000)(delayed(call_fn)(env) for env in envs)
    