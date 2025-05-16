from src.utils.dataloader_utils import SamGemini
from src.utils.physics_utils import query_physical_properties_from_object_list
import os

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_name', type=str, help='Path to the configuration file')
    parser.add_argument("--use_bypass", type=str2bool, default=False)
    parser.add_argument("--dataset", type=str, default="bridge")
    args = parser.parse_args()
    folder = os.path.join(args.scene_name)
    print(f'Processing folder: {folder}')
    if args.dataset == "bridge":
        config_path = 'configs/config_bridge.yaml'
    elif args.dataset == "rh20t":
        config_path = 'configs/config_rh20t.yaml'
    else:
        config_path = 'configs/config_DROID.yaml'
    sam = SamGemini(folder, use_bypass=args.use_bypass, dataset=args.dataset, config_path=config_path)
    input_json = os.path.join(sam.mask_dir, "result.json")
    output_json = input_json
    query_physical_properties_from_object_list(input_json, output_json, config_path)
