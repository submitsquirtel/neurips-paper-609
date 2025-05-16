# Robotic Policy Evaluation Framework

## Environment Setup

### 1. Environment Setup for CogAct

* You should follow the instructions on how to download/use the CogAct model as instructed here https://github.com/microsoft/CogACT

```bash
conda env create -f env/cogact.yml
conda activate cogact
cd CogACT
pip install -e .
pip install uvicorn fastapi "tomli>=1.1.0" "rpds-py>=0.7.1" "traitlets>=5.3"
cd ../SimplerEnv
pip install -e .
pip install -r requirements_full_install.txt 
cd ManiSkill2_real2sim
pip install -e .
pip install --upgrade typing_extensions
cd ../..
```

### 2. Environment Setup for RoboVLM

* You should follow the instructions on how to use/download the RoboVLM model as instructed here https://github.com/Nicolinho/RoboVLM

```bash
conda env create -f env/robovlm.yml
conda activate robovlms
cd RoboVLMs
pip install -e .
cd ../SimplerEnv
pip install -e .
cd ManiSkill2_real2sim
pip install -e .
cd ..
```

### 3. Environment Setup for SpatialVLA and Octo

* You should follow the instructions on how to use/download the Octo model and spatialVla model as instructed here https://github.com/octo-models/octo and https://github.com/SpatialVLA/SpatialVLA

```bash
conda create -n simpler_env python=3.10
conda activate simpler_env
cd octo
pip install -e .
pip install -r requirements.txt
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
cd ../SimplerEnv
pip install -e .
cd ManiSkill2_real2sim
pip install -e .
cd ../..
pip install uvicorn fastapi json_numpy draccus
pip install "scipy<1.13"
pip3 install torch torchvision torchaudio
pip install "transformers == 4.47.0"
```

### 4. Genesis env `genesis_fixed`

* Follow the instructions here https://github.com/Genesis-Embodied-AI/Genesis


## Running Evaluation

### 0. Running Policy Servers

Before running any evaluation scripts, the respective policy server must be active. These servers handle the action generation based on observations and instructions provided by the evaluation environment.

> Make sure in the test script and the server script, you use the same port number.

you can specify the port number in each script by changing the `port` variable in the script.
```python
def run(self, host: str = "0.0.0.0", port: int = 9030) -> None:
```

#### Octo
Activate the Conda environment and run the server:
```bash
conda activate simplier_env
export PYTHONPATH=$(pwd)
python src/server/server_octo.py
```



#### CogAct
Activate the Conda environment and run the server:
```bash
conda activate cogact
export HF_HOME="/data/hf_cache/"
python src/pipeline/server_cogact.py
```

#### spatialVla


* Before running the script, make sure to set the model path to your local model path in the `src/server/server_spatial.py` file. 
* You can download the model from instrustions here https://github.com/SpatialVLA/SpatialVLA

```python
model = SpatialVLAInference(
    saved_model_path=<path_to_your_model>,
    policy_setup="widowx_bridge",
    action_scale=1,
)
```
Activate the Conda environment and run the server:
```bash
conda activate simpler_env
export PYTHONPATH=$(pwd)
python src/pipeline/server_spatialvla.py
```

### RoboVlm

Before running the script, make sure to set the model path to your local model path in the `eval/simpler/server_robovlm.py` file.

```python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ckpt_path = <path_to_your_model>
    config_path = <path_to_your_config_file>
    
    from robovlms.utils.config_utils import load_config
    eval_log_dir = os.path.dirname(ckpt_path)
    policy_setup = "widowx_bridge"
    configs = load_config(config_path)
    model = BaseModelInference(
        ckpt_path=ckpt_path,
        configs=configs,
        device=torch.device("cuda"),
        save_dir=eval_log_dir,
        policy_setup=policy_setup,
    )
    server = RoboVLMServer(model)
    server.run()
    
```

Activate the Conda environment and run the server:
```bash
cd Robovlms
conda activate robovlms
python eval/simpler/server_robovlm.py
```

## Example Data

We provide a sample data for the scene generation and the test scripts.
The data is located in the `examples/data` folder with the following structure:

```text
examples/data
├── assets (contains the assets used in the scene)
│   ├── scene1 [generated scene]
│   ├── scene2 [generated scene]
│   ├── default1 [default scene simulated in simpler_env]
│   ├── default2 [default scene simulated in simpler_env]
├── bridge (contains the camera calibration data and asset positions, physical properties, etc.)
│   ├── scene1
│       ├── extrinsics.npy (Recovered camera extrinsics)
│       ├── intrinsics.npy (Recovered camera intrinsics)
│       ├── lang.txt (Language description of the task)
│       ├── masks
│           ├── result.json (object information and physical properties [only generated scenes])
│           ├── transformations.json (object positions and rotations information)
│   ├── scene2
│   ├── default1
│   ├── default2
├── scene_background (contains the inpainted background image used in the scene)
│   ├── scene1
│       ├── scene_background.png 
│   ├── scene2
│   ├── default1
│   ├── default2
```

## Evaluation 

We provide several evaluation scripts to test the performance of the policy in different scenarios.


> We have provided bash scripts to run all the tests in `bash_scripts` folder. You can modify the arguments in the bash scripts as instructed below to run the tests.

```bash
bash bash_scripts/default_test.bash
bash bash_scripts/background_test.bash
bash bash_scripts/adv_background_test.bash
bash bash_scripts/camera_test.bash
bash bash_scripts/permute_test.bash
bash bash_scripts/pose_test.bash
bash bash_scripts/simpler_test.bash
```

### 0. Common Configurations

Here in `configs/default.yaml`, we provide a default configuration file for the test scripts.

```yaml
base_folder: "./examples/data" # Root path to your generated scene data
robot: "WidowX" # Specifies the robot model
scene_name: "default1" # Identifier for the base scene for this test
```


### 1.Default Test [For both Default and Generated Scenes]

This test evaluates the performance of the policy in a simulated scene with all the default settings. (camera angle, background, object positions, etc.)

The test is performed on both the default scene (e.g., `scene1`) and the generated scenes (e.g., `scene2`).

> Before running the test, make sure you have already run the server for the policy you want to test on the desired port.


* Command:

```bash
conda activate genesis_fixed
export PYTHONPATH=$(pwd)
python src/pipeline/default_test.py \
    --output_dir <output_dir> \  # Path to the output directory where the results will be saved
    --run_all True \ # Set to True to run the test on all the scenes in the dataset
    --config <config_file> \ # Path to the config file, default to `configs/default.yaml`
    --port <port_number> # Port number for the policy server
```

`--output_dir`: Define the type of test and target scene

* To test a default scene, use: `--output_dir ./default_test/<policy_name>`
* To test a generated scene, use: `--output_dir ./generate_test/<policy_name>`


`--run_all`: Decide how many scenes to run

* `--run_all True` Runs all scenes under the base folder (default or generate scenes, depending on your given output_dir path).
* `--run_all False (or omitted)` Runs only the scene specified in the `--config` file.


**Example Commands**

Run all default scenes with a given policy:
```bash
python src/pipeline/default_test.py \
  --output_dir ./default_test/<policy> \
  --run_all True \
  --config configs/default.yaml
```

Run a single generate scene:

```bash
python src/pipeline/default_test.py \
  --output_dir ./generate_test/<policy> 
```

This will save results in the below structure:
```text
default_test
├── <policy_name>
│   ├── default_test
│       ├── <scene_name>
```
```text
generate_test
├── <policy_name>
│   ├── default_test
│       ├── <scene_name>
```

### 2. Background Variation Test [For both Default and Generated Scenes]

This test evaluates the performance of the policy in a simulated scene with a different background image. It will test the scene on all the background images in the specified folder and 5 example background images for testing are provided in the `examples/background` folder.

* Command:

```bash
conda activate genesis_fixed
export PYTHONPATH=$(pwd)
python src/pipeline/background_test.py \
    --output_dir <output_dir> \  # Path to the output directory where the results will be saved
    --run_all True \ # Set to True to run the test on all the scenes in the dataset
    --config <config_file> \ # Path to the config file, default to `configs/default.yaml`
    --port <port_number> # Port number for the policy server
    --background_folder <path_to_background_folder> # Path to the folder containing the background images default to `examples/background`
```

Same as the `default_test.py` script, you can specify the output directory to save the results and the config file to use.

Results will be saved in the same structure as the `default_test.py` script.

```text
default_test
├── <policy_name>
│   ├── background_test
│       ├── <scene_name>
```
```text
generate_test
├── <policy_name>
│   ├── background_test
│       ├── <scene_name>
```

### 3. Background Color Variation Test [For both Default and Generated Scenes]

This script evaluates how changing the color composition of the background in a simulated scene affects the robustness of a robotic policy. The background image is gradually blended with its RGB-transformed variant at various strengths, and a predefined test pipeline is executed on each variant.

**Command:**

```bash
conda activate genesis_fixed
export PYTHONPATH=$(pwd)
python src/pipeline/adv_background_test.py \
    --output_dir <output_dir> \  # Path to the output directory where the results will be saved
    --run_all True \ # Set to True to run the test on all the scenes in the dataset
    --config <config_file> \ # Path to the config file, default to `configs/default.yaml`
    --port <port_number> # Port number for the policy server
```

As with the previous scripts, you can specify the output directory to save the results and the config file to use and results will be saved in the same structure as the `default_test.py` script.

### 4. Camera Variation Test [For both Default and Generated Scenes]

This test evaluates the performance of the policy in a simulated scene with a different camera angle. It will move the camera `up`, `down`, `left`,`right`, `forward`, and `backward` by a certain distance and test the scene on all the camera angles to see how the policy performs.

* Command:

```bash
conda activate genesis_fixed
export PYTHONPATH=$(pwd)
python src/pipeline/camera_test.py \
    --output_dir <output_dir> \  # Path to the output directory where the results will be saved
    --run_all True \ # Set to True to run the test on all the scenes in the dataset
    --config <config_file> \ # Path to the config file, default to `configs/default.yaml`
    --port <port_number> # Port number for the policy server
```

### 5. Permutation Test [For Generated Scenes Only]

This test only evaluates the generated scenes. It will exchange the positions of the objects in the scene -- use different permutations of the objects in the scene to see how the policy performs.

* Command:

```bash
conda activate genesis_fixed
export PYTHONPATH=$(pwd)
python src/pipeline/permute_test.py \
    --output_dir <output_dir> \  # Path to the output directory where the results will be saved
    --run_all True \ # Set to True to run the test on all the scenes in the dataset
    --config <config_file> \ # Path to the config file, default to `configs/default.yaml`
    --port <port_number> # Port number for the policy server
```

### 6. Pose Variation Test [For Default Scenes Only]

This test will only evaluate the default scenes. It will randomly generate different poses and rotations of the objects in the scene and test the scene on all the poses to see how the policy performs.

* Command:

```bash
conda activate genesis_fixed
export PYTHONPATH=$(pwd)
python src/pipeline/pose_test.py \
    --output_dir <output_dir> \  # Path to the output directory where the results will be saved
    --run_all True \ # Set to True to run the test on all the scenes in the dataset
    --config <config_file> \ # Path to the config file, default to `configs/default.yaml`
    --port <port_number> # Port number for the policy server
```


### 7. Object Variation Test [For Default Scenes Only]

This test will only evaluate the default scenes. It will replace the original target object for the task will be changed (e.g., from a default spoon to another object generated frin another real scene specified by `obj_cnt` in the config), and the task is repeated.

* Only this script requires the `obj_cnt` parameter in the config file to specify which object to use for the target object variation, and its config is defaulted to `configs/simpler.yaml`.

**Configuration Example (`configs/simpler.yaml`):**
```yaml
base_folder: < Root path to your generated scene data>
robot: "WidowX" # Specifies the robot model
scene_name: "default1" # Identifier for the base scene for this test
replace_name: "scene1" # Scene to be used for background variation and target object variation
obj_cnt: 4 # Index of objects to be used for target object variation [in this case it will be the banana in scene1]
```

* You can choose which object to use for the target object variation by changing the `obj_cnt` parameter in the config file. You can check the object index and its name in the `masks/result.json` file in each scene's folder.

* Command:

```bash
conda activate genesis_fixed
export PYTHONPATH=$(pwd)
python src/pipeline/simpler_test.py \
    --output_dir <output_dir> \  # Path to the output directory where the results will be saved
    --run_all True \ # Set to True to run the test on all the scenes in the dataset
    --config <config_file> \ # Path to the config file, default to `configs/simpler.yaml`
    --port <port_number> # Port number for the policy server
```


## Output File Structure

If you have run all the tests, you will have the following structure in your output folder:

```text
default_test
├── <policy_A>
│   ├── adv_background_test
│       ├── <scene_name>
│   ├── asset_test
│   ├── background_test
│   ├── camera_test
│   ├── default_test
│   ├──pose_test
├── <policy_A>
...
```

```
generate_test
├── <policy_A>
│   ├── adv_background_test
│       ├── <scene_name>
│   ├── background_test
│   ├── camera_test
│   ├── default_test
│   ├── permute_test
├── <policy_B>
...
```


# GVL Automated Scoring Script

This script provides automated scoring for **GVL (Grounded Video Language)** using **Gemini 2.5 Pro Preview**. It supports multithreaded processing of video trials/tests and saves evaluation scores per video. An API key is required for accessing Gemini.

## Features

- Automated video evaluation via Gemini
- Fast multithreaded inference for scoring multiple trials
- Supports single-shot and zero-shot evaluations
- Saves structured results for analysis

## Requirements

- Python 3.7+
- Gemini 2.5 Pro API key

Install dependencies:

```bash
pip install -r requirements.txt
```

Please use the bash script GVL.bash for this.

```bash
policy="spatial"
variant="background_test"

python src/pipeline/GVL_multithreaded.py \
    --inference "/data/evaluation/generate_test/$policy/$variant/" \
    --base_dir "/data/scene/scene_generation/data/bridge"\
    --key "" \
    --zero true \
    --frequency 3 \
    --dir "/data/scene/openvla/eval_paper_latest_generate_new_test" \
    --test $variant \
    --policy $policy \
    --debug False \
    --model "gemini-2.5-pro-preview-05-06" \
