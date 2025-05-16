#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh  # Change this to your actual conda path
conda activate genesis_fixed

# Set project root as PYTHONPATH
export PYTHONPATH=$(pwd)

echo "Starting Evaluation on Asset Test"

# Run pose test for default RoboVLM scenes
python src/pipeline/simple_test.py \
  --output_dir ./default_test/robovlm \
  --run_all true \
  --port 9000
  # --config <config_file>  # Optional: defaults to config/simpler.yaml

# Notes:
# - Only runs default scenes, output is saved to ./default_test/robovlm/pose_test
# - If running a different policy, change "robovlm" to the respective policy name
# - Results will be saved to ./default_test/robovlm/asset_test
# - Set --run_all to false to test only the scene specified in the config
# You should specify the certain objects you want to use for replacement in the config file