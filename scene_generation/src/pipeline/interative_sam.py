import numpy as np
import yaml
import os
import cv2
import json
import re
import argparse
from src.utils.matching_utils import crop_and_resize
def is_not_only_numbers(obj_name):
    return bool(re.search(r'\D', obj_name)) 

t = 1
def main(data):
    '''
    This function is used to read the Mask video and rgb video and mimic grounded dino results -- we save the masks and the rgb images we separate object amsks from single masksimage''
    '''
    use_bypass = data["use_bypass"]
    output_folder = data["bypass"]["output_folder"]
    os.makedirs(output_folder, exist_ok=True)
    mask_folder = None

    if data['data_type'] == "rh20t":
        mask_folder = os.path.join(f"data/scene/{data['scene_name']}/masks/")
    elif data['data_type'] == "bridge":
        mask_folder = os.path.join(f"data/bridge/{data['scene_name']}/masks/")
    else:
        mask_folder = os.path.join(f"data/DROID/{data['scene_name']}/masks/")
    if use_bypass:
        rgb_video = data["bypass"]["scene_video"]
        object_mask_video = data["bypass"]["object_mask_video"]
        robot_mask_video = data["bypass"]["robot_mask_video"]
        rgb = cv2.VideoCapture(rgb_video)
        mask = cv2.VideoCapture(object_mask_video)
        images = []
        masks = []

        while True:
            ret, rgb_frame = rgb.read()
            ret, mask_frame = mask.read()
            
            if not ret:
                break
            masks.append(mask_frame)
            images.append(rgb_frame)
        
        # given binary mask  separate out the object masks
        os.makedirs(output_folder, exist_ok=True)
        
        gray = cv2.cvtColor(masks[t], cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)

        kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel
        iterations = 1  # Increase if more erosion is needed

        # Apply erosion to separate close regions before contour detection
        thresh = cv2.dilate(thresh, kernel, iterations=iterations)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 1000 # Adjust this value based on your specific needs

        # Filter contours based on area
        contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
        if not contours:
            print("No contours found.")
            return

        os.makedirs(f"data/asset_images/{data['scene_name']}", exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)
        
        for j, contour in enumerate(contours):
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, contours, j, (255, 255, 255), -1)

            mask = cv2.cvtColor(mask.astype(np.uint8) , cv2.COLOR_GRAY2RGB)
            binary_mask = (mask > 0).astype(np.uint8)
            # Save the mask directly as grayscale (no need for extra RGB conversion)
            mask_path = os.path.join(mask_folder, f"image_{j:04d}_mask.png")
            cv2.imwrite(mask_path, binary_mask * 255)
            #  dilate 
            kernel = np.ones((9,9),np.uint8)
            cur_res = cv2.dilate(mask, kernel, iterations = 4)

            cv2.imwrite(f"{output_folder}/image_{j:04d}.png", images[t])
            cv2.imwrite(f"{output_folder}/image_{j:04d}_mask.png", cur_res)
            # Convert the mask to binary
            binary_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            binary_mask = (binary_mask > 0).astype(np.uint8) 
            # Apply the mask to extract the segmented region
            segmented_rgb = cv2.bitwise_and(images[t], images[t], mask=binary_mask)
            # Save the segmented image
            cv2.imwrite(f"data/asset_images/{data['scene_name']}/image_{j:04d}_segmented.png", segmented_rgb)
            output_path = os.path.join(f"data/asset_images/{data['scene_name']}/cropped", f"image_{j:04d}_segmented.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            crop_and_resize(cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR), output_path=output_path)
            
        
        robot_mask_video = cv2.VideoCapture(robot_mask_video)
        robot_masks = []
        while True:
            ret, robot_mask_frame = robot_mask_video.read()
            if not ret:
                break
            robot_masks.append(cv2.cvtColor(robot_mask_frame, cv2.COLOR_BGR2RGB))
        os.path.join(mask_folder, f"image_{j:04d}_mask.png")
        gray = cv2.cvtColor(robot_masks[0], cv2.COLOR_RGB2GRAY)
        kernel = np.ones((9,9),np.uint8)
        cur_res = cv2.dilate(gray, kernel, iterations = 4)
        cv2.imwrite(f"{output_folder}/image_{j+1:04d}.png", images[0])
        cv2.imwrite(f"{output_folder}/image_{j+1:04d}_mask.png", cur_res)
    else:
        os.makedirs(f"data/asset_images/{data['scene_name']}", exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)
        mask_output_folder = mask_folder
        mask_folder = data["sam2gemini"]["mask_folder"]
        rgb_folder = data["sam2gemini"]["rgb_folder"]
        frame_path = os.path.join(rgb_folder, "1.jpg")
        rgb = cv2.imread(frame_path)
        for file in os.listdir(mask_folder):
            if file.startswith("frame_00001"):
                json_path = os.path.join(mask_folder, file)
                # Load the json file
                with open(json_path, "r") as f:
                    json_data = json.load(f)
                masks = {obj["object_name"]: np.array(obj["mask"]) for obj in json_data}
                count = 0
                result_list = []
                
                for obj_name, mask in masks.items():
                    # obj_name should not contain "robot"
                    if is_not_only_numbers(obj_name) and "robot" not in obj_name:
                        # have to make another dict indicting the mask, object_name and its respective count in the dict
                        # remove any number from obj_name
                        obj_name_dict = re.sub(r'\d+', '', obj_name)
                        # if obj_name already in the dict, then add the count
                        obj_name_list = [obj["object_name"] for obj in result_list]
                        if obj_name_dict in obj_name_list:
                            obj_index = obj_name_list.index(obj_name_dict)
                            result_list[obj_index]["count"].append(count)
                        else:
                            json_dict = {
                                "object_name": obj_name_dict,
                                "count": [count]
                            }
                            result_list.append(json_dict)
                        
                        masked_rgb = np.zeros_like(rgb, dtype=np.uint8)
                        gray_masked_rgb = np.zeros_like(rgb, dtype=np.uint8)
                        masked_rgb[mask] = rgb[mask]
                        output_path = os.path.join(f"data/asset_images/{data['scene_name']}/cropped", f"image_{count:04d}_segmented.png")
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        crop_and_resize(cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR), output_path=output_path)
                        cv2.imwrite(f"data/asset_images/{data['scene_name']}/image_{count:04d}_segmented.png", masked_rgb)
                        gray_masked_rgb[mask] = 255
                        gray_masked_rgb[~mask] = 0 
                        mask_path = os.path.join(mask_output_folder, f"image_{count:04d}_mask.png")
                        cv2.imwrite(mask_path, gray_masked_rgb)
                        kernel = np.ones((9,9),np.uint8)
                        cur_res = cv2.dilate(gray_masked_rgb, kernel, iterations = 4)
                        cv2.imwrite(f"{output_folder}/image_{count:04d}.png", rgb)
                        cv2.imwrite(f"{output_folder}/image_{count:04d}_mask.png", cur_res)
                        # have to rename the pcd to {count:04d}_frame1.ply
                        pcd_dir = data["sam2gemini"]["pcd_folder"]
                        pcd_file = os.path.join(pcd_dir, f"{obj_name}_frame1.ply")
                        if not os.path.exists(pcd_file):
                            print(f"File {pcd_file} does not exist.")
                        else:
                            new_pcd_file = os.path.join(pcd_dir, f"{count:04d}_frame1.ply")
                            os.rename(pcd_file, new_pcd_file)
                        count += 1
                # Save the json file
                json_path = os.path.join(mask_output_folder, "result.json")
                with open(json_path, "w") as f:
                    json.dump(result_list, f)
                
                for obj_name, mask in masks.items():
                    if not is_not_only_numbers(obj_name) or "robot" in obj_name:
                        gray_masked_rgb = np.zeros_like(rgb, dtype=np.uint8)
                        gray_masked_rgb[mask] = 255
                        gray_masked_rgb[~mask] = 0 
                        kernel = np.ones((9,9),np.uint8)
                        cur_res = cv2.dilate(gray_masked_rgb, kernel, iterations = 4)
                        cv2.imwrite(f"{output_folder}/image_{count:04d}.png", rgb)
                        cv2.imwrite(f"{output_folder}/image_{count:04d}_mask.png", cur_res)
                        count += 1
                    

def dataloader(dataset):
    '''
    This function is used to load the data from the yaml file. The data is stored in the yaml file in the form of a dictionary
    '''
    if dataset == "rh20t":
        with open("configs/rh20t_config.yaml", "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    elif dataset == "bridge":
        with open("configs/bridge_config.yaml", "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    elif dataset == "droid":
        with open("configs/droid_config.yaml", "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    return data

if __name__ == "__main__":
    # add argparse to get the dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="rh20t", help='Path to the configuration file')
    parser.add_argument('--single_scene', type=bool, default=False)
    args = parser.parse_args()
    data = dataloader(args.dataset)
    if args.single_scene:
        with open("configs/custom_config.yaml", "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    main(data)
    