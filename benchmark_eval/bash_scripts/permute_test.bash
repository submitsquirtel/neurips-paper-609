#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh  # Change this to your actual conda path
conda activate genesis_fixed

# Set project root as PYTHONPATH
export PYTHONPATH=$(pwd)

echo "Starting Evaluation on Permute Test"

# Run pose test for default RoboVLM scenes
python src/pipeline/permute_test.py \
  --output_dir ./generate_test/robovlm \
  --run_all true \
  --port 9000
  # --config <config_file>  # Optional: defaults to config/default.yaml

# Notes:
# - Only runs generated scenes, output is saved to ./default_test/robovlm/permute_test
# - Results will be saved to ./generate_test/robovlm/permute_test
# - If running a different policy, change "robovlm" to the respective policy name
# - Set --run_all to false to test only the scene specified in the config