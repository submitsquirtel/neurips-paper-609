from src.utils.dataloader_utils import SamGemini
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
    
    if not args.use_bypass:
        sam.get_masked_objects()

    frame_idx = 1
    sam.mask_to_point_cloud(frame_idx)
    # sam.som_gemini()   # for 123
    # sam.gemini_som_gemini() # For 4th
    # sam.molmo_points()
