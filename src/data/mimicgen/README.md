# Mimicgen 
- Paper: https://arxiv.org/pdf/2310.17596
- Code: https://github.com/NVlabs/mimicgen
- Documentation: https://mimicgen.github.io/docs/introduction/overview.html

## Installation
Mimicgen requires mujoco, robosuite, robomimic and robosuite_task_zoo. 

Install `mimicgen` as follows:
```
# mujoco
pip install mujoco==2.3.2

# robosuite 
pip install robosuite==1.4.1
# requires 
pip install gymnasium==0.28.1

# git cone, install, robomimic: https://github.com/ARISE-Initiative/robomimic
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .

# git clone, install: https://mimicgen.github.io/docs/introduction/installation.html
git clone https://github.com/NVlabs/mimicgen.git
cd mimicgen
pip install -e .

# git clone install: https://github.com/ARISE-Initiative/robosuite-task-zoo
git clone https://github.com/ARISE-Initiative/robosuite-task-zoo.git
cd robosuite-task-zoo
pip install -e .
pip install mujoco==2.3.2
pip install mujoco_py==2.0.2.5
```

## Troubleshooting
`egl-probe` may fail. Solve by: 
```
pip install cmake
```

## Data download
Download the 26 original `core` datasets provided by the Mimicgen publication: 
```
# using gdown
pip install gdown
gdown --folder https://drive.google.com/drive/folders/14uywHbSdletLBJUmR8c5UrBUZkALFcUz
# or any of the methods described here: https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html
```

## Data structure
Every .hdf5 contains a field `data` and individual fields for each episode.
Each episodes contains `states`, `actions`, `rewards`, `dones`, and `obs`.
- `states` contains the simulation state, not the actual continous observation state. 
- `obs` contains the individual observation attributes. 

`mimigen` uses `robomimic` underneath, which leverages the `EnvRobosuite` env. This class removes a number of fields
from the actual `obs` than would be returned by the original `robosuite` env. Consequently, the data does not contain
these fields and subsequently, the interaction environment also has to be an `EnvRobosuite` env.

In particular, `EnvRobosuite` removes: 
- `object-state` containing the state of the object --> added back via `object` field.
- `robot0_proprio-state` containing the state of the robot --> has to be reconstructed from the individual robot states. 

The stored dataset additionally contain image fields `agentview_image` and `robot0_eye_in_hand_image`. 
Consequently, we sort fields alphabetically and remove the image fields to get the actual observation fields. 

## Data preparation
Prepare the 26 original `core` datasets used for our experiments: 
```
# low dim keys, binary reward 
python prepare_data.py --add_rtgs --low_dim_keys --compress --sparse_reward --save_dir=./data/mimicgen/core_processed_sparse
```

Then download our generated datasets (additional robot arms) and extract them (e.g., using `untar_files.sh`) to the same directory: 
```
cd ./data/mimicgen
huggingface-cli download ml-jku/mimicgen59 --local-dir=./mimicgen59 --repo-type dataset
bash untar_files.sh mimicgen59 core_processed_sparse
```

Other splits can be produced similarly: 
```
# low dim keys only
python prepare_data.py --add_rtgs --low_dim_keys --compress --save_dir=./data/mimicgen/core_processed

# img observations
python prepare_data.py --add_rtgs --compress --img_key=agentview_image --crop_dim=64 --save_dir=./data/mimicgen/core_processed_agentview
```

## Data generation
To generate your own datasets using the `mimicgen` framework we refer to:  
- https://github.com/NVlabs/mimicgen/blob/ea0988523f468ccf7570475f1906023f854962e9/docs/tutorials/getting_started.md