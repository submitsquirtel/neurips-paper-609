# Scene Placement in Simulation


## Introduction

This repository provides a framework for scene placement in simulation environments. 

## Repository Structure

```plaintext
.
├── README.md
├── bash_scripts [bash scripts for running the code]
├── configs [configuration files for different datasets]
├── data [dataset folder]
├── env [environment files]
├── Hunyuan3D-2
├── InvSR
├── lama
├── MINIMA
├── third_party
├── output
├── src [source code]
└── utils
```


## Environment Setup

> Note: The checkpoints for each model are not included in the repository. You need to download them separately.

### Step 0:  **Clone the Repository**
```bash
git clone --recurse-submodules <repository_url>
cd <repository_name>
```


### Step 1: Set up environment `match`

> Please make sure you have all checkpoints downloaded and placed in the correct folders.

* set up the environment [in the ropo folder]

```bash
conda env create -f env/environment_match.yaml
conda activate match
pip install -e .
```

* Download respective checkpoints/submodules for `MINIMA`
* You can check for more detailed instructions here https://github.com/LSXI7/MINIMA


```bash
git submodule update --init --recursive
git submodule update --recursive --remote
sed -i '1s/^/from typing import Tuple as tuple\n/' third_party/RoMa/romatch/models/model_zoo/__init__.py
cd MINIMA
bash weights/download.sh 
cd ..
```

* Download respective checkpoints/submodules for `sam2`

```bash
cd .. # avoid path conflict 
git clone https://github.com/facebookresearch/sam2.git && cd sam2
cd checkpoints && \
./download_ckpts.sh && \
cd ..
pip install -e .
```


### Step 2: Set up environment `gemini`

```bash
conda env create -f env/environment_gemini.yaml
```

### Step 3: Set up environment `hunyuan`

> Please make sure you have all checkpoints downloaded and placed in the correct folders.

* You can check for more detailed instructions here https://github.com/Tencent/HunyuanVideo for **environment setup** and **checkpoint download**.

Follow the instructions in the `Hunyuan3D-2` to set up the environment. 

```bash
cd Hunyuan3D-2
# create a conda environment
conda create -n hunyuan python=3.10
conda activate hunyuan

pip install -r requirements.txt
# for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
```
### Step 4: Set up environment `lama`

> Please make sure you have all checkpoints downloaded and placed in the correct folders.

* You can check for more detailed instructions here https://github.com/advimman/lama for **environment setup** and **checkpoint download**.


```bash
cd lama
conda env create -f conda_env.yml
conda activate lama
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
pip install pytorch-lightning==1.2.9
```
### Step 5: Set up environment `invsr`

> Please make sure you have all checkpoints downloaded and placed in the correct folders.

* You can check for more detailed instructions here https://github.com/zsyOAOA/InvSR for **environment setup** and **checkpoint download**.


```bash
conda env create -f env/environment_invsr.yaml
conda activate invsr
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
cd InvSR/
pip install -e ".[torch]"
cd ..
```
### Step 6: Set up environment `genesis_fixed`

Follow the instructions here https://github.com/Genesis-Embodied-AI/Genesis

## Running Structure and Procedure

### Code Structure

```bash
├── scene_generation
│   ├── ...
│   ├── submodules
│   ├── lama
│   ├── src
│   │   ├── pipeline
│   │   │   ├── 
│   │   ├── utils
│   │   ├── sim
```


### Config structure

```
├── configs
│   ├── rh20t_config.yaml  : running scene_generation script for rh20t
│   ├── config_rh20t.yaml  : base config for rh20t
│   ├── bridge_config.yaml : running scene_generation script for bridge
│   ├── config_bridge.yaml : base config for bridge
│   ├── DROID_config.yaml  : running scene_generation script for droid
│   ├── config_DROID.yaml  : base config for droid
```
### Segmentation

We provide 3 different segmentation methods [how to segment out robot and target object in the video] for scene generation. 

1. **gemini**: This method uses the latest gemini model for segmentation. 

* config setting:

* in `configs/*_config.yaml` file, set the following parameters:

```yaml
use_bypass: false
use_molmo: false
```

* in `configs/config_*.yaml` file, set the following parameters:

```yaml
use_bypass: false
gemini_mask: true
```

2. **gemini + molmo + sam2**: Firsr use gemini to get text prediction of objects the robot interacts with, then use molmo to `point` to the object, and finally use sam2 to get the mask of the object.

* config setting:

* in `configs/*_config.yaml` file, set the following parameters:

```yaml
use_bypass: false
use_molmo: true
```
* in `configs/config_*.yaml` file, set the following parameters:

```yaml
use_bypass: false
molmo:
  enabled: true  # Set to True if using MoLMO for video generation
  cache_dir: "the path to the cache directory where the model is stored"
  text: 'Please point me to all foreground objects the robot interacts with in the scene and do not point to the robot itself.'
gemini_mask: false
```

3. **manual_mask**: This method allows the user to manually create masks for the scene. 

* config setting:

* in `configs/*_config.yaml` file, set the following parameters:

```yaml
use_bypass: true
```

* in `configs/config_*.yaml` file, set the following parameters:

```yaml
use_bypass: true
```

* Get the robot mask video and object mask video from SAM2 demo https://ai.meta.com/sam2/ and save them in each scene's folder as `mask.mp4` and `robot_mask.mp4` respectively.


### Running the code

### 1. Running preprocessing script

#### **For Bridge dataset**

Download the bridge dataset and put it in <bridge_root> where you want to store the dataset. 

The dataset should be in the following structure:

```bash
├── bridge_root
│   ├── bridge_data_v1
│   ├── bridge_data_v2
│   ├── flap
│   ├── icra
│   ├── rss
```

```bash
bash bash_scripts/preprocess.bash configs/bridge_config.yaml 
```

You can change the args in `bash_scripts/preprocess.bash` to set the dataset path and other parameters.

```bash
python3 src/pipeline/preprocess_camera.py \
    --data_dir "./data/bridge" \
    --raw_path "<your_bridge_dataset_root>/<choose_from_bridge_data_v2/icra...>" \
    --depth 4 \
    --num_scenes <number_of_scenes_to_process>
```

* After running the above command, you will get the following structure in `data/bridge`:

```bash
├── data
│   ├── bridge
│   │   ├── scene{number}
│   │   │   ├── images0 
│   │   │   ├── obs_dict.pkl 
│   │   │   ├── lang.txt 
│   │   │   ├── video.mp4
│   │   │   ......
```


#### **For DROID dataset**


* After running the above command, you will get the following structure in `data/DROID`:

```bash
├── data
│   ├── bridge
│   │   ├── scene{number}
│   │   │   ├── images0 
│   │   │   ├── grippers.npy
│   │   │   ├── joints.npy
│   │   │   ......
```

#### **For RH20T dataset**

Download the RH20T dataset and put it in <rh20t_root> where you want to store the dataset. [For this preprocess, it only takes care of scenes in one cfg, so you may need to run it multiple times for different cfgs]

The dataset should be in the following structure:

```bash
├── rh20t_root
│   ├── RH20T_cfg{number} [raw_folder]
│   ├── RH20T_cfg{number}_depth [raw_folder]
```

* You shuld assign a certain camera serial number to the camera you want to use in `configs/rh20t_config.yaml` file, for example:

```yaml
camera_serial: 1234567890
```

```bash
elif [ "$data_type" == "rh20t" ]; then
    python3 src/pipeline/preprocess_camera.py \
        --camera_to_use "$camera_serial" \
        --data_type "rh20t" \
        --data_dir "./data/scene/"\
        --raw_path "<your_rh20t_dataset_root>" \
        --num_scenes <number_of_scenes_to_process>\
        --cfg <cfg_number> 
```

* After running the above command, you will get the following structure in `data/scene`:

```bash
├── data
│   ├── scene
│   │   ├── task_0008_user_0010_scene_0003_cfg_0005
│   │   │   ├── images0 [extracted_frames]
│   │   │   ├── depth [extracted_real_depth]
│   │   │   ├── color.mp4 
│   │   │   ├── configs.json
│   │   │   ├── extrinsics.npy
│   │   │   ├── intrinsics.npy
│   │   │   ├── tcp.npy
│   │   │   ├── gripper.npy
│   │   │   ├── joint.npy
│   │   │   ...
```

### 2. Recover intrinsics, extrinsics and depth

```bash
├── data
│   ├── bridge
│   │   │   ├── depth   [recovered]
│   │   │   ├── extrinsics.npy [recovered]
│   │   │   ├── intrinsics.npy [recovered]
│   │   │   ├── ... [store_scene_generation_results]
│   ├── DROID
│   │   │   ├── same as above
```

### 3. Running pipeline script

Before running the pipeline script, make sure you have : 

1. Downloaded the checkpoints for each model and placed them in the correct folders. 
2. Set up the environment for each model as instructed in the respective sections above.
3. Set the parameters in the config files as instructed in the respective sections above.

* For Bridge dataset 

```bash
bash bash_scripts/pipeline_bridge.bash configs/bridge_config.yaml 
```

* For Droid dataset

```bash
bash bash_scripts/pipeline_droid.bash configs/droid_config.yaml 
```

* For RH20T dataset 

```bash
bash bash_scripts/pipeline_rh20t.bash configs/rh20t_config.yaml 
```

* For custome scene

```
bash bash_scripts/custom_onescene.bash configs/custom_config.yaml
```

For this script you need to first put the scene in the respective folder with correct structure.

* put the data in one of the following folders: `data/bridge` for bridge dataset, `data/DROID` for DROID and `data/scene` for rh20t dataset.

```bash
├── data
│   ├── bridge 
│   │   │   ├── depth   [recovered]
│   │   │   ├── extrinsics.npy [recovered]
│   │   │   ├── intrinsics.npy [recovered]
│   │   │   ├── ... [store_scene_generation_results]
│   ├── DROID
│   │   │   ├── same as above
│   ├── scene
│   │   │   ├── same as above
```

* Then modify configs/custom_config.yaml file to set the following parameters:

```yaml
scene_name: "scene_name" # the name of the scene you want to run
camera_to_use: [optional] # the camera serial number you want to use only for rh20t dataset
data_type: "bridge" # the type of dataset you want to use, can be bridge, droid or rh20t
robot: "WidowX" # the robot you want to use, can be WidowX, franka or UR5
use_bypass: false # set to true if you want to use manual mask
use_molmo: false # set to true if you want to use molmo
```

* And you should also modify the respective config in `configs/config_<dataset>.yaml` file to set the parameters as instructed in the section above.