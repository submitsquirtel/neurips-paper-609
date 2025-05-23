#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh  # Change this to your actual conda path
conda activate genesis_fixed

# Set project root as PYTHONPATH
export PYTHONPATH=$(pwd)

echo "Starting Evaluation on Default Test"

# Run pose test for default RoboVLM scenes
python src/pipeline/default_test.py \
  --output_dir ./default_test/robovlm \
  --run_all true \
  --port 9000
  # --config <config_file>  # Optional: defaults to config/default.yaml

# Notes:
# - If you want to test only generated scenes, change the output directory to ./generate_test/robovlm
# - Results will be saved to ./default_test/robovlm/default_test or ./generate_test/robovlm/default_test
# - If running a different policy, change "robovlm" to the respective policy name
# - Set --run_all to false to test only the scene specified in the config