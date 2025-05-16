from src.utils.dataloader_utils import SamGemini
import os

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_name', type=str, help='Path to the configuration file')
    parser.add_argument("--use_bypass", type=str2bool, default=False)
    parser.add_argument("--method", type=str, help="Method to run")
    args = parser.parse_args()
    folder = os.path.join(args.scene_name)
    print(f'Processing folder: {folder}')
    sam = SamGemini(folder, use_bypass=args.use_bypass)
    if args.method == "molmo_sam2":
        sam.molmo_points()
    elif args.method == "som_gemini":
        sam.som_gemini()
    elif args.method == "gemini_som_gemini":
        sam.gemini_som_gemini()
    else:
        print("Method not recognized")
