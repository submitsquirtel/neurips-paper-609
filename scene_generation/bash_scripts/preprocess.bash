#!/bin/bash
config_file=$1

# Ensure the config file exists
if [ ! -f "$config_file" ]; then
    echo "Error: Config file $config_file does not exist."
    exit 1
fi
data_type=$(yq -r '.data_type' "$config_file")
if [ "$data_type" == "rh20t" ]; then
    camera_serial=$(yq -r '.camera_serial' "$config_file")
fi
# **Run the pipeline process**
source ~/miniconda3/etc/profile.d/conda.sh  
echo "Running preprocessing..."
conda activate match
export PYTHONPATH=$(pwd)
if [ "$data_type" == "bridge" ]; then
    python3 src/pipeline/preprocess_camera.py \
        --data_type "bridge" \
        --data_dir "./data/bridge" \
        --raw_path <PATH_TO_RAW_DATA> \
        --depth 4\
        --num_scenes 5


elif [ "$data_type" == "rh20t" ]; then
    python3 src/pipeline/preprocess_camera.py \
        --camera_to_use "$camera_serial" \
        --data_type "rh20t" \
        --data_dir "./data/scene/"\
        --raw_path <PATH_TO_RAW_DATA> \
        --num_scenes 5 \
        --cfg 5
else
    echo "Invalid data type. Please use 'bridge' or 'scene'."
    exit 1
fi

conda deactivate
