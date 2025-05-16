#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e
# Load Conda environment (user should ensure this path is correct and conda is initialized)
source ~/miniconda3/etc/profile.d/conda.sh  
echo "Starting scene generation..."
# Read the provided config file path
config_file=$1
# Ensure the config file path is provided
if [ -z "$config_file" ]; then
    echo "Error: No config file provided."
    echo "Usage: $0 <path_to_config_file>"
    exit 1
fi
# Ensure the config file exists
if [ ! -f "$config_file" ]; then
    echo "Error: Config file '$config_file' does not exist."
    exit 1
fi

scene_name=$(yq -r '.scene_name' "$config_file")
if [ "$scene_name" == "null" ] || [ -z "$scene_name" ]; then # Check for null string or empty
    echo "Error: 'scene_name' key not found or is null in config file: $config_file"
    exit 1
fi
data_type=$(yq -r '.data_type' "$config_file")
use_bypass=$(yq -r '.use_bypass' "$config_file") 
use_molmo=$(yq -r '.use_molmo' "$config_file")  

robot=$(yq -r '.robot' "$config_file")
if [[ "$data_type" == "bridge" ]]; then
    proprioception_path="data/bridge/${scene_name}/obs_dict.pkl"
    data_folder="bridge"
    config_filepath="configs/config_bridge.yaml"
fi

if [[ "$data_type" == "rh20t" ]]; then
    camera_to_use=$(yq -r '.camera_to_use' "$config_file")
    camera_serial=$(yq -r '.camera_serial' "$config_file")
    proprioception_path=$(yq -r '.proprioception_path' "$config_file")
    data_folder="scene"
    config_filepath="configs/config_rh20t.yaml"
fi

if [[ "$data_type" == "droid" ]]; then
    proprioception_path="data/DROID/${scene_name}/joints.npy"
    data_folder="DROID"
    config_filepath="configs/config_DROID.yaml"
fi

# Generate paths related to the current scene (same logic as original script's loop)
object_position="data/${data_folder}/${scene_name}/masks/transformations_new.json"
mask_folder="./data/${data_folder}/${scene_name}/bounding_boxes_and_masks"
rgb_folder="./data/${data_folder}/${scene_name}/images0"
pcd_folder="./data/${data_folder}/${scene_name}/pcds"
scene_video="./data/${data_folder}/${scene_name}/video.mp4"
object_mask_video="./data/${data_folder}/${scene_name}/mask.mp4"
robot_mask_video="./data/${data_folder}/${scene_name}/robot_mask.mp4"

bypass_output_and_lama_input_dir="./data/input_mask/${scene_name}/"
mkdir -p "$bypass_output_and_lama_input_dir" # Crucial for bypass output & LaMa input staging
# Dynamic config path
# write to original config file
cat <<EOT > "$config_file"
scene_name: "$scene_name"
proprioception_path: "$proprioception_path"
camera_to_use: "$camera_to_use"
object_position: "$object_position"
data_type: "$data_type"
robot: "$robot"
use_bypass: $use_bypass
use_molmo: $use_molmo
sam2gemini:
    mask_folder: "$mask_folder"
    rgb_folder : "$rgb_folder"
    pcd_folder : "$pcd_folder"
bypass:
    scene_video: "$scene_video"
    object_mask_video: "$object_mask_video"
    robot_mask_video: "$robot_mask_video"
    output_folder: "$bypass_output_and_lama_input_dir" # Where bypass stage outputs masks
EOT

# If bypass is enabled, specific steps are taken
if [ "$use_bypass" == "true" ]; then
    echo "Bypass enabled. Using existing masks and bounding boxes, running alternative point cloud generation."
    export PYTHONPATH="${PYTHONPATH}:$(pwd)" # Ensure current dir is in PYTHONPATH if scripts need it
    conda activate match
    python3 src/pipeline/interative_sam.py --dataset "$data_type" --single_scene True # Assumes this uses dynamic_config_filepath
    python3 src/pipeline/segment_object.py --scene_name "$scene_name" --use_bypass "$use_bypass" --dataset "$data_type" 
    conda deactivate
fi

if [[ "$use_bypass" == "false" && "$use_molmo" == "false" ]]; then
    read w h <<< $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \-of csv=p=0:s=x ./data/$data_folder/$scene_name/images0/0.jpg | tr 'x' ' ')
    mkdir -p ./data/$data_folder/$scene_name/images0/super_res/
    cp ./data/$data_folder/$scene_name/images0/1.jpg ./data/$data_folder/$scene_name/images0/super_res/
    conda activate invsr
    cd InvSR
    python3 inference_invsr.py \
        -i ../data/$data_folder/${scene_name}/images0/super_res/ \
        -o ../data/$data_folder/${scene_name}/images0/super_res/ \
        --num_steps 2
    cd ..
    ffmpeg -i ./data/$data_folder/${scene_name}/images0/super_res/1.png \
            -vf "scale=${w}:${h}" \
            -y ./data/$data_folder/${scene_name}/images0/super.png
    rm -rf ./data/$data_folder/${scene_name}/images0/super_res/
fi



# Run SAM-related processing if bypass is false
if [ "$use_bypass" == "false" ]; then
    echo "Running segmentation processing (bypass is false)..."
    if [ "$use_molmo" == "true" ]; then
        echo "Using molmo for segmentation..."
        conda activate match
    else # use_molmo is false
        conda activate gemini
    fi
    export PYTHONPATH=$(pwd)
    python3 src/pipeline/segment_object.py --scene_name "$scene_name" --use_bypass "$use_bypass" --dataset "$data_type"
    if [ "$use_molmo" == "true" ]; then
        python3 src/pipeline/interative_sam.py --dataset "$data_type" --single_scene True
        echo "Saved masks in ./data/$data_folder/$scene_name/masks/"
        conda activate match
        export PYTHONPATH=$(pwd)
        python3 src/pipeline/prompt_physics.py --scene_name "$scene_name" --dataset "$data_type"
        conda deactivate
    fi
    if [ "$use_molmo" == "false" ]; then
        rm -rf ./data/input_mask/$scene_name
        cp -r ./data/$data_folder/$scene_name/input_mask ./data/input_mask/$scene_name
        rm -rf ./data/$data_folder/$scene_name/input_mask
        echo "Saved masks in data/$data_folder/$scene_name/masks/"
        conda deactivate
    fi
fi

echo "Running LaMa for background inpainting..."
conda activate lama
cd lama # Script expects to be run from lama directory
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/../data/input_mask/$scene_name outdir=$(pwd)/../data/output_mask/$scene_name scene_name=$scene_name
conda deactivate
cd ../

# Image enhancement using InvSR (conditional blocks from original)
asset_images_base_folder="./data/asset_images/${scene_name}"
asset_images_cropped_folder="${asset_images_base_folder}/cropped"
asset_images_enhanced_folder="${asset_images_base_folder}/enhanced"
mkdir -p "$asset_images_cropped_folder" "$asset_images_enhanced_folder"

# Condition 1: (bypass OR molmo) -> enhance data/asset_images/$scene_name/cropped/
if [[ "$use_bypass" == "true" || "$use_molmo" == "true" ]]; then
    echo "Enhancing images with InvSR (from $asset_images_cropped_folder to $asset_images_enhanced_folder)..."
    conda activate invsr
    cd InvSR
    python3 inference_invsr.py \
        -i "../$asset_images_cropped_folder" \
        -o "../$asset_images_enhanced_folder" \
        --num_steps 2
    conda deactivate
    cd ../
    conda deactivate
fi

if [[ "$use_bypass" == "false" && "$use_molmo" == "false" ]]; then
    read w h <<< $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \-of csv=p=0:s=x ./data/$data_folder/$scene_name/images0/0.jpg | tr 'x' ' ')
    echo "Enhancing images with InvSR..."
    conda activate invsr
    cd InvSR
    python3 inference_invsr.py -i ../data/$data_folder/$scene_name/masks/ -o ../data/$data_folder/$scene_name/masks/ --num_steps 2
    src="../data/$data_folder/$scene_name/masks"
    out_dir="../data/$data_folder/$scene_name/masks"
    echo "Resizing images in $src to ${w}x${h}..."
    for img in "../data/$data_folder/$scene_name/masks/"*.png; do
        [ -e "$img" ] || continue
        filename=$(basename "$img")
        tmpfile=$(mktemp --suffix=".${filename##*.}")
        ffmpeg -i "$img" -vf "scale=${w}:${h}" -y "$tmpfile"
        mv "$tmpfile" "../data/$data_folder/$scene_name/masks/$filename"
        echo "Resized $img to ${w}x${h} and saved to ../data/$data_folder/$scene_name/masks/$filename"
    done
    conda activate invsr
    python3 inference_invsr.py -i ../data/$data_folder/$scene_name/masks/cropped/ -o ../data/asset_images/$scene_name/enhanced/ --num_steps 2
    cd ../
    conda deactivate
fi


echo "Retrieving assets from Hunyuan3D..."
export OMP_NUM_THREADS=8
conda activate hunyuan
mkdir -p ./data/assets/$scene_name
python3 src/pipeline/hunyuan.py --folder ./data/asset_images/$scene_name/
conda deactivate


# Find correct object positions and orientations
echo "Finding object positions and orientations..."
conda activate match
export PYTHONPATH=$(pwd)
CONFIG_FILE=${config_filepath}
CHECKPOINT_DIR=$(yq -r '.paths.MINIMA_checkpoint' "$CONFIG_FILE")
folder="./data/assets/$scene_name"

for mesh_glb in "$folder"/*.glb; do
    [ -f "$mesh_glb" ] || continue 
    filename=$(basename -- "$mesh_glb")
    mesh_cnt=$(echo "$filename" | cut -d'_' -f2)
    folder_name=$(basename "$folder")
    mesh_name=$(basename "$mesh_glb")
    echo "Processing: folder=$folder_name, mesh_glb=$mesh_name, mesh_cnt=$mesh_cnt"
    python3 src/pipeline/MatchFindRot.py \
        --folder="$folder_name" \
        --mesh_glb="$mesh_name" \
        --mesh_cnt="$mesh_cnt" \
        --config_path="$CONFIG_FILE" \
        --checkpoint_dir="$CHECKPOINT_DIR"
done
conda deactivate
        
conda activate match
export PYTHONPATH=$(pwd)
python3 src/pipeline/postprocess.py --scene_name "$scene_name" --dataset "$data_type"

# Run the final scene rendering process

echo "Running scene rendering"
conda activate genesis_fixed # As per original
export PYTHONPATH=$(pwd)
if [[ "$data_type" == "rh20t" ]]; then
    python3 src/demo/rh20t_demo.py --config "$config_file" # Uses the dynamic config
elif [[ "$data_type" == "bridge" ]]; then
    python3 src/demo/bridge_demo.py --config "$config_file" # Uses the dynamic config
elif [[ "$data_type" == "droid" ]]; then
    python3 src/demo/droid_demo.py --config "$config_file" # Uses the dynamic config
fi

conda deactivate

echo "Finished processing scene: $scene_name"