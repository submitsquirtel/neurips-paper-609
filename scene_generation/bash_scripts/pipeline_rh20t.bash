#!/bin/bash

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh  
echo "Starting scene generation..."

# Read the provided config file path
config_file=$1

# Ensure the config file exists
if [ ! -f "$config_file" ]; then
    echo "Error: Config file $config_file does not exist."
    exit 1
fi

camera_serial=$(yq -r '.camera_serial' "$config_file")
for scene_dir in ./data/scene/*/; do
    if [ -d "$scene_dir" ]; then
        # Extract scene_name from the directory name
        scene_name=$(basename "$scene_dir")
        # If scene_name doesn't start with "task", skip the scene
        echo "Processing scene: $scene_name"
        if [[ ! "$scene_name" =~ ^task ]]; then
            echo "Skipping scene : $scene_name"
            continue
        fi
        # Read fixed parameters from the config file
        camera_serial=$(yq -r '.camera_serial' "$config_file")
        camera_to_use=$(yq -r '.camera_to_use' "$config_file")
        data_type=$(yq -r '.data_type' "$config_file")
        use_bypass=$(yq -r '.use_bypass' "$config_file")
        use_molmo=$(yq -r '.use_molmo' "$config_file")
        proprioception_path=$(yq -r '.proprioception_path' "$config_file")
        # Generate paths related to the current scene
        object_position="data/scene/${scene_name}/masks/transformations.json"
        mask_folder="./data/scene/${scene_name}/bounding_boxes_and_masks"
        rgb_folder="./data/scene/${scene_name}/images0"
        pcd_folder="./data/scene/${scene_name}/pcds"
        scene_video="./data/scene/${scene_name}/color.mp4"
        object_mask_video="./data/scene/${scene_name}/mask.mp4"
        robot_mask_video="./data/scene/${scene_name}/robot_mask.mp4"
        output_folder="./data/input_mask/${scene_name}/"

        # **Update configs/rh20t_config.yaml dynamically for the current scene**
        cat <<EOT > configs/rh20t_config.yaml
scene_name: "$scene_name"
proprioception_path: "$proprioception_path"
object_position: "$object_position"
camera_to_use: "$camera_to_use"
camera_serial: "$camera_serial"
data_type: "$data_type"
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
    output_folder: "$output_folder"
EOT

        echo "Updated configs/rh20t_config.yaml for scene: $scene_name"
        # If bypass is enabled, skip certain steps
        if [ "$use_bypass" == "true" ]; then
            if [ -z "${scene_name:-}" ]; then
                echo "Error: scene_name is not set."
                exit 1
            fi
            echo "Skipping SAM processing and using existing masks and bounding boxes."
            export PYTHONPATH=$(pwd)
            python3 src/pipeline/interative_sam.py
            echo "Saved masks in data/scene/$scene_name/masks/"
            conda activate match
            echo "Getting point clouds..."
            python3 src/pipeline/segment_object.py --scene_name "$scene_name" --use_bypass "$use_bypass" --dataset "$data_type"
            conda deactivate
        fi

        if [[ "$use_bypass" == "false" && "$use_molmo" == "false" ]]; then
            read w h <<< $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \-of csv=p=0:s=x ./data/rh20t/$scene_name/images0/0.jpg | tr 'x' ' ')
            mkdir -p ./data/rh20t/$scene_name/images0/super_res/
            cp ./data/rh20t/$scene_name/images0/1.jpg ./data/rh20t/$scene_name/images0/super_res/
            conda activate invsr
            cd InvSR
            python3 inference_invsr.py \
                -i ../data/rh20t/${scene_name}/images0/super_res/ \
                -o ../data/rh20t/${scene_name}/images0/super_res/ \
                --num_steps 2
            cd ..
            ffmpeg -i ./data/rh20t/${scene_name}/images0/super_res/1.png \
                -vf "scale=${w}:${h}" \
                -y ./data/rh20t/${scene_name}/images0/super.png
            rm -rf ./data/rh20t/${scene_name}/images0/super_res/
        fi

        # Run SAM processing if bypass is false
        if [ "$use_bypass" == "false" ]; then
            echo "Running SAM processing..."
            echo "running segmentation processing..."
            if [ "$use_molmo" == "true" ]; then
                conda activate match
            fi
            if [ "$use_molmo" == "false" ]; then
                conda activate gemini
            fi
            export PYTHONPATH=$(pwd)
            python3 src/pipeline/segment_object.py --scene_name "$scene_name" --use_bypass "$use_bypass" --dataset "$data_type"
            
            if [ "$use_molmo" == "true" ]; then
                python3 src/pipeline/interative_sam.py --dataset "$data_type"
                echo "Saved masks in data/scene/$scene_name/masks/"
                conda activate match
                export PYTHONPATH=$(pwd)
                python3 src/pipeline/prompt_physics.py --scene_name "$scene_name" --dataset "$data_type"
                conda deactivate
            fi
            if [ "$use_molmo" == "false" ]; then
                rm -rf ./data/input_mask/$scene_name
                cp -r ./data/scene/$scene_name/input_mask ./data/input_mask/$scene_name
                rm -rf ./data/scene/$scene_name/input_mask
                echo "Saved masks in data/scene/$scene_name/masks/"
                conda deactivate
            fi
        fi

        # Background inpainting using LaMa
        echo "Running LaMa for background inpainting..."
        conda activate lama
        cd lama
        export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
        python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/../data/input_mask/$scene_name outdir=$(pwd)/../data/output_mask/$scene_name scene_name=$scene_name
        conda deactivate
        cd ../

        # Image enhancement using InvSR
        if [[ "$use_bypass" == "true" || "$use_molmo" == "true" ]]; then
            # Image enhancement using InvSR
            echo "Enhancing images with InvSR..."
            conda activate invsr
            cd InvSR
            python3 inference_invsr.py -i ../data/asset_images/$scene_name/cropped/ -o ../data/asset_images/$scene_name/enhanced/ --num_steps 2
            cd ../
            conda deactivate
        fi
        if [[ "$use_bypass" == "false" && "$use_molmo" == "false" ]]; then
            read w h <<< $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \-of csv=p=0:s=x ./data/scene/$scene_name/images0/0.jpg | tr 'x' ' ')
            echo "Enhancing images with InvSR..."
            conda activate invsr1
            cd InvSR
            python3 inference_invsr.py -i ../data/scene/$scene_name/masks/ -o ../data/scene/$scene_name/masks/ --num_steps 2
            src="../data/scene/$scene_name/masks"
            out_dir="../data/scene/$scene_name/masks"
            echo "Resizing images in $src to ${w}x${h}..."
            for img in "../data/scene/$scene_name/masks/"*.png; do
                [ -e "$img" ] || continue
                filename=$(basename "$img")
                tmpfile=$(mktemp --suffix=".${filename##*.}")
                ffmpeg -i "$img" -vf "scale=${w}:${h}" -y "$tmpfile"
                mv "$tmpfile" "../data/scene/$scene_name/masks/$filename"
                echo "Resized $img to ${w}x${h} and saved to ../data/scene/$scene_name/masks/$filename"
            done
            conda activate invsr1
            python3 inference_invsr.py -i ../data/scene/$scene_name/masks/cropped/ -o ../data/asset_images/$scene_name/enhanced/ --num_steps 2
            cd ../
            conda deactivate
        fi

        # Retrieve assets from Hunyuan3D
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
        CONFIG_FILE="configs/config_rh20t.yaml"
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
                --checkpoint_dir="$CHECKPOINT_DIR"\
                --config_path="$CONFIG_FILE" 
        done
        conda deactivate

        conda activate match
        export PYTHONPATH=$(pwd)
        python3 src/pipeline/postprocess.py --scene_name "$scene_name" --dataset "$data_type"
        # Run the final scene rendering process
        echo "Running scene rendering..."
        conda activate genesis_fixed
        export PYTHONPATH=$(pwd)
        python3 src/demo/rh20t_demo.py
        conda deactivate
        echo "Finished processing scene: $scene_name"
    fi
done

echo "All scenes have been processed!"