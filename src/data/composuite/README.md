# Composuite 
- Paper: https://arxiv.org/pdf/2207.04136
- Code: https://github.com/Lifelong-ML/CompoSuite
- Documentation & Data Download: https://datadryad.org/stash/dataset/doi:10.5061/dryad.9cnp5hqps

## Installation
Composuite uses mujoco and robosuite underneath. We use mujoco 2.3.0 and robosuite 1.4.1 to remain compatible with mimicgen. 
Composuite officially requires robosuite==1.4.0, but it is possible to use robosuite==1.4.1. 
Requirement is to have gymnasium==0.28.1 installed. Consequently, we make use of compatibitliy wrappers during env creation.

Install `compusuite` as follows: 
```
# mujoco
pip install mujoco==2.3.2

# robosuite 
pip install robosuite==1.4.1
# requires
pip install gymnasium==0.28.1

# git cone, install composuite: https://github.com/Lifelong-ML/CompoSuite.git
git clone https://github.com/Lifelong-ML/CompoSuite.git
cd CompoSuite
pip install -e .
```

## Troubleshooting
`evdev` installation may cause error when installing robosuite: 
```
mamba install -c conda-forge evdev=1.7.1
```
May result in issue with libffi:
```
pip uninstall cffi
pip install cffi==1.15.0
```

## Data preparation
First, download and extract the `expert` datasets from https://datadryad.org/stash/dataset/doi:10.5061/dryad.9cnp5hqps.

Then prepare the datasets accordingly:
```
cd src/data/composuite
python prepare_data.py --add_rtgs --compress --data_dir=DATA_DIR --save_dir=SAVE_DIR
```
