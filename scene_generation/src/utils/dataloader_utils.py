import os
import re
import json
import cv2
import pdb
import time
import torch
import yaml
import pickle
import open3d as o3d
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from src.utils.track_utils import sample_points_from_masks
from src.utils.video_utils import create_video_from_images
from src.utils.matching_utils import get_uncropped_keypoints, project_pixel_to_point
import supervision as sv


class SamGemini:
    def __init__(self, task_folder, config_path="configs/config_rh20t.yaml", use_bypass=False, dataset = "rh20t"):
        self.config = self.load_config(config_path)
        dataset = self.config['dataset']
        self.task_folder = task_folder
        self.data_dir = self.config['paths']['data_dir']
        self.data_dir_task = os.path.join(self.data_dir, task_folder)
        self.dataset = dataset
        result_dir = self.config['paths']['result_dir']
        os.makedirs(result_dir, exist_ok=True)
        self.video_dir = os.path.join(self.data_dir_task, "images0")
        self.depth_dir = os.path.join(self.data_dir_task, "depth")
        self.mask_dir = os.path.join(self.data_dir_task, "masks")
        self.pcd_dir = os.path.join(self.data_dir_task, "pcds")
        self.save_dict = os.path.join(self.data_dir_task, "bounding_boxes_and_masks")
        
        #TODO remove this when not necessary
        self.save_dir = os.path.join(result_dir, "tracking_results", task_folder)
        self.gemini_output = os.path.join(result_dir, "gemini_output", task_folder)
        self.molmo_dir = os.path.join(result_dir , "molmo", task_folder)
        self.SoM_dir = os.path.join(result_dir, "SoM", task_folder)
        self.gemini_non = os.path.join(result_dir, "gemini_non", task_folder)
        os.makedirs(self.gemini_non, exist_ok=True)
        os.makedirs(self.SoM_dir, exist_ok=True)
        os.makedirs(self.gemini_output, exist_ok=True)
        os.makedirs(self.molmo_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        assert(os.path.exists(self.data_dir_task))
        assert(os.path.exists(self.video_dir))
        assert(os.path.exists(self.depth_dir))
        os.makedirs(self.save_dict, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)
        self.use_bypass = use_bypass
        self.intrinsic = np.load(os.path.join(self.data_dir_task, "intrinsics.npy"), allow_pickle=True)
        self.extrinsic = np.load(os.path.join(self.data_dir_task, "extrinsics.npy"), allow_pickle=True)
        if self.intrinsic.shape != (3, 3):
            self.intrinsic = self.intrinsic[0]
            # write back to path
            np.save(os.path.join(self.data_dir_task, "intrinsics.npy"), self.intrinsic)
              

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def initialize_cuda(self):
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def initialize_sam2(self):
        from sam2.build_sam import build_sam2_video_predictor, build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        sam2_checkpoint = self.config['paths']['sam2_checkpoint']
        model_cfg = self.config['paths']['model_cfg']
        
        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
    
    def initialize_grounding_dino(self):    
        model_id = self.config['models']['grounding_dino']['model_id']
        device = self.config['models']['grounding_dino']['device']
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    def generate_prompt(self,task):
        if task == "":
            return self.config["prompts"]["foreground_objects"]

    def gemini_CoT(self, all_video_files, output_json, text=None):
        import google.generativeai as genai
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part
        """
        Use the Gemini model to generate text for the given video/image files.
        @param all_video_files: List of images or single image file path
        @param output_json: Path to the output JSON file
        @return: The generated text
        """
        
        single_file = False
        if isinstance(all_video_files, str):
            all_video_files = [all_video_files]
            single_file = True

        GOOGLE_API_KEY = self.config['models']['gemini']['api_key']
        genai.configure(api_key=GOOGLE_API_KEY)
        project_id = self.config['models']['gemini']['project_id']
        vertexai.init(project=project_id, location="us-central1")

        model = genai.GenerativeModel(model_name=self.config['models']['gemini']['model_name'])

        if not os.path.exists(output_json):
            with open(output_json, 'w') as f:
                json.dump({}, f)
        with open(output_json, 'r') as f:
            output = json.load(f)

        task = self.config['prompts']['task_specific']['task']
        if single_file:
            prompt = self.config['SoM']['gemini_prompt']
        else:
            prompt = self.generate_prompt(task) 
        if text is not None:
            prompt = "Select the set of mark annotations which give the bounding box for the " + text + " in the image except the robot. Do not output anything else than the mark number. Make sure you take only rigid objects and do not consider large surfaces. Do not confuse parts of large surfaces as well. Output Format: 1,2,3"
        
        for i, file_name in enumerate(all_video_files):
            chat_model = model.start_chat(history=[])
            print(f"[INFO] Processing video file {i+1}/{len(all_video_files)}: {file_name}")
            if single_file:
                video_file = genai.upload_file(path=file_name, mime_type='image/jpeg')  # Assuming image is in JPEG format
            else:
                video_file = genai.upload_file(path=file_name, mime_type='video/mp4')
            while video_file.state.name == "PROCESSING":
                print('.', end='')
                time.sleep(10)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                raise ValueError(video_file.state.name)
            
            print(f"Retrieved file '{video_file.display_name}' as: {video_file.uri}")
                    
            response = chat_model.send_message([video_file, prompt], request_options={"timeout": 600})
            
            if response.candidates:
                text_output_ = response.candidates[0].content.parts[0].text
                output[file_name] = text_output_
                with open(output_json, 'w') as f:
                    json.dump(output, f, indent=4)
                time.sleep(1)
                genai.delete_file(video_file.name)
            
            else:
                print(f"No response for video {video_file}, skipping.")
        return response.candidates[0].content.parts[0].text
    
    def test_gemini(self,input_json, output_json):
        with open(input_json) as f:
            video_files_data = json.load(f)
        video_files = []
        video_files.extend(k for k, v in video_files_data.items())
        response = self.gemini_CoT(video_files, output_json)
        return response
        
    def put_text(self, image, text, pos, max_width, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7, color=(180, 230, 230), thickness=2):
        import textwrap 
        y = pos[1]
        for line in textwrap.wrap(text, width=max_width // 10):
            while cv2.getTextSize(line, font, scale, thickness)[0][0] > max_width:
                scale -= 0.05  # Reduce font size if text is too wide
            cv2.putText(image, line, (pos[0], y), font, scale, color, thickness, cv2.LINE_AA)
            y += cv2.getTextSize(line, font, scale, thickness)[0][1] + 5
    
    def gemini_mask(self, image):
        from google import genai
        from google.genai import types
        import google
        from io import BytesIO
        from src.utils.gemini_utils import parse_segmentation_masks, plot_segmentation_masks, SegmentationMask
        client = genai.Client(api_key=self.config['models']['gemini']['api_key'])
        prompt = self.config['prompts']['gemini_mask']
        task = None
        if self.dataset == "bridge":
            lang_instruction = os.path.join(self.data_dir_task, "lang.txt")
            if os.path.exists(lang_instruction):
                with open(lang_instruction, 'r') as f:
                    task_lines = f.readlines()
                task_lines = [line.strip() for line in task_lines if "confidence" not in line]
                task_lines = task_lines[0]
                task = " ".join(task_lines)
        elif self.dataset == "droid":
            for file in os.listdir(self.data_dir_task):
                if file.startswith("language_annotations") and file.endswith(".json"):
                    with open(os.path.join(self.data_dir_task, file), 'r') as f:
                        task_lines = json.load(f)
                    task = task_lines["language_instruction1"]
        
        all_segmentation_masks = []
        model_name = self.config['models']['gemini']['model_name']
        safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ]
        im = Image.open(BytesIO(open(image, "rb").read()))
        im.thumbnail([1024,1024], Image.Resampling.LANCZOS)
        
        # ----------- [1] Detect the robot only -----------
        robot_prompt = self.config['prompts']['gemini_robot']
        response_robot = client.models.generate_content(
            model=model_name,
            contents=[robot_prompt, im],
            config=types.GenerateContentConfig(
                temperature=0.5,
                safety_settings=safety_settings,
            )
        )
        robot_masks = parse_segmentation_masks(response_robot.text, img_height=im.size[1], img_width=im.size[0])
        all_segmentation_masks.extend(robot_masks)
        
        # ----------- [2] Detect the objects -----------
        prompt = self.config['prompts']['gemini_mask']
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, im],
            config = types.GenerateContentConfig(
                temperature=0.5,
                safety_settings=safety_settings,
            )
        )
        segmentation_masks = parse_segmentation_masks(response.text, img_height=im.size[1], img_width=im.size[0])
        all_segmentation_masks.extend(segmentation_masks)
        img2 = image.replace("1.jpg", "super.png")
        img2 = Image.open(BytesIO(open(img2, "rb").read()))
        img, object_dicts = plot_segmentation_masks(img2, all_segmentation_masks, self.mask_dir)
    
        object_dict_path = os.path.join(self.save_dict, f"frame_00001_bounding_boxes_and_masks.json")
        with open(object_dict_path, 'w') as json_file:
            json.dump(object_dicts, json_file, indent=4)
        # ------------ [3] Find target / destination object -----------
        result_json = os.path.join(self.data_dir_task, "masks", "result.json")
        # collect all object name in the json file
        with open(result_json, 'r') as f:
            data = json.load(f)
        # collect all object names and their index from the "count" field
        object_name_to_index = {
            obj['object_name']: obj['count'][0] for obj in data
        }
        object_names = list(object_name_to_index.keys())
        if task is not None:
            prompt = f"""
                consider this specific task: {task}, given the object names: {object_names},
                please select the object names that are relevant to the task and identify their role in the task from "target" and "destination".
                Output in this format exactly as: target: <target_object>, destination: <destination_object>.
                Please try to identify the target object (If you can't not find an exact match, use the closest one from given object names) and if there is no destination object, only output the target object.
            """
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    safety_settings=safety_settings,
                )
            )
            # Parse the response to extract target and destination objects and change the name in the json file
            target = None
            destination = None
            if response.text is not None and "target:" in response.text:
                target = response.text.split("target: ")[1].split(",")[0].strip()
            if response.text is not None and "destination:" in response.text:
                destination = response.text.split("destination: ")[1].split(",")[0].strip()
            # write the new json file and replace the object names with _target and _destination
            new_json = []
            for obj in data:
                if obj['object_name'] == target:
                    obj['object_name'] = target + "_target"
                elif obj['object_name'] == destination:
                    obj['object_name'] = destination + "_destination"
                new_json.append(obj)
            # Overwrite the original json file with the new one
            with open(result_json, 'w') as f:
                json.dump(new_json, f, indent=4)
        return img

    def get_masked_objects(self): 
        self.initialize_cuda()
        if self.config['gemini_mask']:
            self.gemini_mask(os.path.join(self.video_dir, "1.jpg"))
            return
        input_json = os.path.join(self.gemini_output, self.config["paths"]["genimi_input_json"])

        create_video_from_images(self.video_dir, input_json)
        response = self.test_gemini(
            input_json = input_json,
            output_json= os.path.join(self.gemini_output, "images0_gemini.json")
        )
        image =  cv2.imread(os.path.join(self.video_dir, "1.jpg"))
        max_w = image.shape[1] - 20
        self.put_text(image, f"Response: {response}", (10, image.shape[0] - 50), max_w)
        cv2.imwrite(os.path.join(self.gemini_output, "gemini_text.jpg"), image)
        
        if self.config['molmo']['enabled']:
            self.molmo_points()
        else:
            self.process_video()
    
    def som(self, gemini=False):
        from src.utils.SoM import ImageSegmentation
        somProcessor = ImageSegmentation()
        image = Image.open(os.path.join(self.video_dir, "1.jpg"))
        slider = 2  # For example, choose the value between 1 and 3
        mode = "Automatic"  # or "Interactive"
        alpha = 0.5  # Mask alpha
        label_mode = "Number"  # or "Alphabet"
        # anno_mode = ["Mask", "Mark"]  # Annotations to apply
        anno_mode = ["Mark"]  # Annotations to apply
        
        # Call the inference function
        result, mask = somProcessor.inference(image, slider, mode, alpha, label_mode, anno_mode)
        if isinstance(result, np.ndarray):
            result = Image.fromarray(result)
        if gemini:
            image_file_path = os.path.join(self.gemini_non, "SoM_output.png")
        else:
            image_file_path = os.path.join(self.SoM_dir, "SoM_output.png")
        result.save(image_file_path)
        return mask
    def gemini_som_gemini(self):
        from src.utils.SoM import ImageSegmentation
        self.initialize_cuda()
        mask = self.som(gemini=True)
        self.initialize_cuda()
        input_json = os.path.join(self.gemini_non, self.config["paths"]["genimi_input_json"])
        create_video_from_images(self.video_dir, input_json)
        response = self.test_gemini(
            input_json = input_json,
            output_json= os.path.join(self.gemini_non, "images0_gemini.json")
        )
        image =  cv2.imread(os.path.join(self.video_dir, "1.jpg"))
        max_w = image.shape[1] - 20
        self.put_text(image, f"Response: {response}", (10, image.shape[0] - 50), max_w)
        cv2.imwrite(os.path.join(self.gemini_non, "gemini_text.jpg"), image)
        output_json= os.path.join(self.gemini_non, "images0_gemini.json")
        
        with open(output_json, 'r') as f:
            output_data = json.load(f)
        video_file = list(output_data.keys())[0]
        if "\n" in output_data[video_file]:
            text = output_data[video_file].split("\n")[-1].strip()
        else:
            text = output_data[video_file].strip()
        text = '. '.join([item for item in text.split('. ') if 'robot' not in item.lower()])
        
        splits = text.split(".")
        image_file_path = os.path.join(self.gemini_non, "SoM_output.png")
        output_all = []
        for txt in splits:
            if txt == "" or txt == " ":
                continue
            output_json = os.path.join(self.gemini_non, "SoM.json")
            output = self.gemini_CoT(image_file_path, output_json, txt)
            self.visualize_som(output, mask, self.gemini_non, txt)
            output_all.append(output)
            #self.visualize_som(output, mask, self.gemini_non)
        self.visualize_som(",".join(output_all), mask, self.gemini_non)
        # output = self.gemini_CoT(image_file_path, output_json, text)
        # self.visualize_som(output, mask, self.gemini_non)
        
        
    def visualize_som(self, output, mask, output_dir, txt = None):
        numbers = list(map(int, output.split(",")))
        image = cv2.imread(os.path.join(self.video_dir, "1.jpg"))
        overlaid = np.array(image.copy())
        mask_all = np.zeros((overlaid.shape[0], overlaid.shape[1]), dtype=np.uint8)
        for i in numbers:
            curr = mask[i-1]["segmentation"]
            # Ensure the mask is the same size as the overlaid image
            if curr.dtype != np.uint8:
                curr = (curr * 255).astype(np.uint8) 
            curr = cv2.resize(curr, (overlaid.shape[1], overlaid.shape[0]))
            mask_all = cv2.bitwise_or(mask_all, curr)
            mask_rgb = cv2.cvtColor(curr, cv2.COLOR_GRAY2RGB)
            img = cv2.addWeighted(mask_rgb, 0.6, image, 0.4, 0)
            img_path = os.path.join(output_dir, f"{i}.jpg")
            cv2.imwrite(img_path, img)
        # Save the final result
        mask_all = cv2.cvtColor(mask_all, cv2.COLOR_GRAY2RGB)
        overlaid = cv2.addWeighted(mask_all, 0.6, overlaid, 0.4, 0)
        if txt!=None:
            cv2.imwrite(os.path.join(output_dir, f"{txt}_overlay.jpg"), overlaid)
        else:
            cv2.imwrite(os.path.join(output_dir, "SoM_overlay.png"), overlaid)
        
    def som_gemini(self):
        self.initialize_cuda()
        mask = self.som()
        
        image_file_path = os.path.join(self.SoM_dir, "SoM_output.png")
        output_json = os.path.join(self.SoM_dir, "SoM.json")
        output = self.gemini_CoT(image_file_path, output_json)
        self.visualize_som(output, mask, self.SoM_dir)
            
    def molmo_points(self):
        from src.utils.molmo import ForegroundPointExtractor
        output_json= os.path.join(self.gemini_output, "images0_gemini.json")
        with open(output_json, 'r') as f:
            output_data = json.load(f)
        video_file = list(output_data.keys())[0]
        if "\n" in output_data[video_file]:
            text = output_data[video_file].split("\n")[-1].strip()
        else:
            text = output_data[video_file].strip()
        text = '. '.join([item for item in text.split('. ') if 'robot' not in item.lower()])
        if 'robot' not in text.lower():
            text += " robot arm."
        if 'gripper' not in text.lower():
            text += " robot gripper."
        
        dataset = self.config['dataset']
        if dataset == "droid":
            dataset = "DROID"
        pointsExtractor = ForegroundPointExtractor(config_path = f"configs/config_{dataset}.yaml")
        frame_index = 1
        image_path = os.path.join(self.video_dir, f"{frame_index}.jpg")
        image = Image.open(image_path)
        
        parts = text.split(".")
        all_points = {}
        
        for part in parts:
            if part == "" or part == " ":
                continue
            output_dir = os.path.join(self.molmo_dir, part)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "0.jpg")
            points = pointsExtractor.extract_points(image_path, output_path, part)
            all_points[part] = points
        self.initialize_sam2()
        
        object_dicts = []
        for part, points in all_points.items():
            if points is None:
                continue
            for i, point in enumerate(points):
                points = np.array(point).reshape(1, 2)
                #  check if multiple points
                image_copy = np.array(image.copy().convert("RGB"))
                for point in points:
                    print(point)
                    cv2.circle(image_copy, (int(point[0]), int(point[1])), 5, (255, 255, 0), -1)
                output_dir = os.path.join(self.molmo_dir, part)
                self.image_predictor.set_image(np.array(image.convert("RGB")))
                points_labels = np.ones((points.shape[0]), dtype=np.int32)
                
                masks, scores, logits = self.image_predictor.predict(
                    point_coords=points,
                    point_labels=points_labels,
                    box=None,
                    multimask_output=False,
                )
                object_name = f"{part}{i+1}"
                mask_store = masks[0].copy().astype(bool)
                object_dict = {
                    'object_name': object_name,
                    'mask': mask_store.tolist(),  # Mask as a list
                }
                object_dicts.append(object_dict)
                # save object dict in data folder
                object_dict_path = os.path.join(self.save_dict, f"frame_00001_bounding_boxes_and_masks.json")
                with open(object_dict_path, 'w') as json_file:
                    json.dump(object_dicts, json_file, indent=4)
                
    
    def save_overlay(self, image, mask, i, output_dir):
        # image = np.array(image.convert("RGB"))
        # Assuming masks[0] is already in the correct format
        mask = (mask.astype(np.uint8) * 255)  # Convert mask to uint8 and scale it

        # Convert the mask to a 3-channel image (to match the image's channels)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # Blend the image and mask using weighted addition
        overlay = cv2.addWeighted(mask_rgb, 0.5, image, 0.5, 0)

        # Save the overlay image
        overlay_path = os.path.join(output_dir, f"{i}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay)
    
    def process_video(self):
        self.initialize_sam2()
        self.initialize_grounding_dino()
        
        output_json= os.path.join(self.gemini_output, "images0_gemini.json")
        with open(output_json, 'r') as f:
            output_data = json.load(f)
        video_file = list(output_data.keys())[0]
        text = output_data[video_file].replace('\n', ' ').strip()
        if 'robot' not in text.lower():
            text += " robot arm."
        if 'gripper' not in text.lower():
            text += " robot gripper."
            
        print(f"Extracted text for {video_file}: {text}")
        
        frame_names = sorted([p for p in os.listdir(self.video_dir) if p.endswith(".jpg")], key=lambda p: int(os.path.splitext(p)[0]))
        inference_state = self.video_predictor.init_state(video_path=self.video_dir)
        
        ann_frame_idx = 1
        img_path = os.path.join(self.video_dir, frame_names[ann_frame_idx])
        image = Image.open(img_path)
        
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.config['models']['grounding_dino']['device'])
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold= self.config['grounding_dino']['box_threshold'],
            text_threshold= self.config['grounding_dino']['threshold'],
            target_sizes=[image.size[::-1]]
        )
        
        self.image_predictor.set_image(np.array(image.convert("RGB")))
        input_boxes = results[0]["boxes"].cpu().numpy()
        OBJECTS = results[0]["labels"]
        
        masks, scores, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 3:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)
            
        # PROMPT_TYPE_FOR_VIDEO = "box" # or "point"

        PROMPT_TYPE_FOR_VIDEO = self.config['tracking']['prompt_type']
        
        assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

        # If you are using point prompts, we uniformly sample positive points based on the mask
        if PROMPT_TYPE_FOR_VIDEO == "point":
            # sample the positive points from mask for each objects
            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

            for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )
        # Using box prompt
        elif PROMPT_TYPE_FOR_VIDEO == "box":
            for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )
        # Using mask prompt is a more straightforward way
        elif PROMPT_TYPE_FOR_VIDEO == "mask":
            for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
                labels = np.ones((1), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    mask=mask
                )
        else:
            raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")


        video_segments = {
            out_frame_idx: {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                inference_state
            )
        }
        ID_TO_OBJECTS = dict(enumerate(OBJECTS, start=1))
        
        for frame_idx, segments in video_segments.items():
            # TODO see if there is still need to get bbox/mask for whole video
            if frame_idx > 1:
                break
            img = cv2.imread(os.path.join(self.video_dir, frame_names[frame_idx]))
            
            object_ids = list(segments.keys())
            
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)
            
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                mask=masks, # (n, h, w)
                class_id=np.array(object_ids, dtype=np.int32),
            )
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(os.path.join(self.save_dir, f"{frame_idx:05d}.jpg"), annotated_frame)
            objects_info = []

            # Save the bounding box, mask, and object_id for each object
            for obj_id, mask, box in zip(object_ids, detections.mask, detections.xyxy):
                object_name = ID_TO_OBJECTS[obj_id]  # Get the object name from the ID
                combined_name = f"{object_name}{obj_id}"
                object_info = {
                    'object_name': combined_name, 
                    # 'object_id': obj_id,
                    'mask': mask.tolist(),  # Mask as a list
                    'bounding_box': box.tolist(),  # Bounding box as a list
                }
                objects_info.append(object_info)

            frame_json_path = os.path.join(self.save_dict, f"frame_{frame_idx:05d}_bounding_boxes_and_masks.json")
            with open(frame_json_path, 'w') as json_file:
                json.dump(objects_info, json_file, indent=4)
        
        self.get_tracking_results(1)

    def get_mask_and_bounding_box(self,frame_idx):
        file_path = os.path.join(self.save_dict, f"frame_{frame_idx:05d}_bounding_boxes_and_masks.json")
        with open(file_path, 'r') as f:
                frame_data = json.load(f)
        return {obj["object_name"]: (np.array(obj["mask"]), np.array(obj["bounding_box"])) for obj in frame_data}
        
    def get_tracking_results(self,frame_idx):
        rgb = self.get_rgb(frame_idx)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        output_json= os.path.join(self.gemini_output, "images0_gemini.json")
        with open(output_json, 'r') as f:
            output_data = json.load(f)
        gemini_text =list(output_data.values())[0]
        
        mask_bbox = self.get_mask_and_bounding_box(frame_idx)
        for obj_name, (mask, box) in mask_bbox.items():
            detections = sv.Detections(
                xyxy=np.expand_dims(box, axis=0),  # (1, 4)
                mask=np.expand_dims(mask, axis=0),  # (1, h, w)
                class_id=np.array([0], dtype=np.int32),
            )
            
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=rgb.copy(), detections=detections)
            
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, 
                detections=detections, 
                labels=[obj_name]  # Use object name as label
            )
            
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            
            cv2.putText(
                annotated_frame, gemini_text, (10, annotated_frame.shape[0] - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 230, 230), 2, cv2.LINE_AA
            )
            
            obj_save_dir = os.path.join(self.save_dir)
            os.makedirs(obj_save_dir, exist_ok=True)
            save_path = os.path.join(obj_save_dir, f"{obj_name}.jpg")
            cv2.imwrite(save_path, annotated_frame)            
    
    def get_bounding_box(self, frame_idx):
        """Retrieve bounding box information for a specific frame index."""
        file_path = os.path.join(self.save_dict, f"frame_{frame_idx:05d}_bounding_boxes_and_masks.json")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"[ERROR] Bounding box file for frame {frame_idx} not found.")
            return None

        # Load the bounding box data from JSON
        with open(file_path, 'r') as f:
            frame_data = json.load(f)

        # Return a dictionary mapping object_id to its bounding box
        return {obj["object_name"]: obj["bounding_box"] for obj in frame_data}

    def get_mask(self, frame_idx, use_bypass=False):
        if not self.use_bypass and not use_bypass:
            """Retrieve mask information for a specific frame index."""
            file_path = os.path.join(self.save_dict, f"frame_{frame_idx:05d}_bounding_boxes_and_masks.json")
            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"[ERROR] Mask file for frame {frame_idx} not found.")
                return None
            # Load the mask data from JSON
            with open(file_path, 'r') as f:
                frame_data = json.load(f)
            # Return a dictionary mapping object_id to its mask (converted to NumPy array)
            # if there's key 'count' in frame_data, remove it
            masks_dict = {}
            for obj in frame_data:
                if 'count' in obj:
                    count = str(obj['count']).zfill(4)
                    masks_dict[count] = np.array(obj["mask"])
                else:
                    masks_dict[obj["object_name"]] = np.array(obj["mask"])
            return masks_dict
            # return {obj["object_name"]: np.array(obj["mask"]) for obj in frame_data}
        
        else:
            # Use the bypass method to get the mask
            assert(frame_idx == 1)
            masks = {}
            for img in os.listdir(self.mask_dir):
                if img.endswith("_mask.png"):
                    img_path = os.path.join(self.mask_dir, img)
                    img_cnt = img.split("_")[1]
                    # turn 0001 to int
                    mask_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    # trun the mask to true or false
                    masks[img_cnt] = mask_image > 0
            return masks
                    

    
    def get_rgb(self, frame_idx):
        frame_path = os.path.join(self.video_dir, f"{frame_idx}.jpg")
        if os.path.exists(frame_path):
            return cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        print(f"RGB frame {frame_idx} not found.")
        return None
    
    def get_depth(self, frame_idx):
        depth_path = os.path.join(self.depth_dir, f"{frame_idx}.npy")
        if os.path.exists(depth_path):
            return np.load(depth_path)
        return None
    
    def filter_pointclouds(self, pcd, num_neighbors:int, std_ratio:float, radius):
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=num_neighbors, std_ratio=std_ratio)
        pcd = pcd.select_by_index(ind)
        if radius > 0:
            _, ind = pcd.remove_radius_outlier(nb_points=num_neighbors, radius=radius)
            pcd = pcd.select_by_index(ind)

        return pcd
    
    def get_original(self, frame_idx, mesh_cnt, X,Z, keypoints = None):
        masks = self.get_mask(frame_idx, use_bypass=True)
        rgb = self.get_rgb(frame_idx)
        depth = self.get_depth(frame_idx)
        demo_image = None
        
        # convert depth to float32
        depth = depth.astype(np.float32)
        if rgb is None or depth is None:
            return None
        
        width, height = rgb.shape[1], rgb.shape[0]
        rgb = cv2.resize(rgb, (int(width ), int(height ))).astype(np.int8)
        depth = cv2.resize(depth, (int(width ), int(height)))
        
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.set_intrinsics(int(width ), int(height ),
                                     self.intrinsic[0, 0] , self.intrinsic[1, 1] ,
                                     self.intrinsic[0, 2] , self.intrinsic[1, 2] )
        extrinsics = self.extrinsic
        
        for obj_name, mask in masks.items():
            if obj_name == mesh_cnt:
                masked_rgb = np.zeros_like(rgb, dtype=np.uint8)
                masked_depth = np.zeros_like(depth)
                # Apply mask to rgb and depth arrays
                masked_rgb[mask] = rgb[mask]
                masked_depth[mask] = depth[mask]
                # Set the regions outside the mask to white (255)
                masked_rgb[~mask] = 0
                demo_image = masked_rgb
                
                image1 = np.zeros(masked_rgb.shape, dtype=np.uint8)
                rgbd_image = o3d.geometry.RGBDImage()
                rgbd_image = rgbd_image.create_from_color_and_depth(
                    o3d.geometry.Image(masked_rgb), o3d.geometry.Image(masked_depth), 
                    depth_scale=1.0, convert_rgb_to_intensity=False)
                
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_o3d, extrinsic=self.extrinsic)
                pcd = self.filter_pointclouds(pcd, 10, 2.0, 0.01)
                points_3d = []
                
                if keypoints is not None:
                    keypoints = get_uncropped_keypoints(keypoints, X,Z)
                    points_2d = keypoints
                    depth_key = np.zeros(masked_depth.shape)
                    for point in points_2d:
                        depth_point = masked_depth[point[1], point[0]]
                        u, v = point
                        if depth_point <= 0:
                            # find the nearest point
                            for distance in range(1, 10):
                                for m, n in [(v-distance, u), (v+distance, u), (v, u-distance), (v, u+distance)]:
                                    if masked_depth[int(m), int(n)] > 0 and int(m) < height and int(n) < width:
                                        depth_point = masked_depth[int(m), int(n)]
                                        u, v = n, m
                                        break
                        
                        depth_key[v,u] = depth_point
                        cv2.circle(masked_rgb, (point[0], point[1]), radius=1, color=(0, 255, 0))
                        
                        if depth_point > 0:
                            point3d = project_pixel_to_point(u,v,depth_point, self.intrinsic, self.extrinsic)
                        else:
                            point3d = [-1, -1, -1]
                        points_3d.append(point3d)
                        image1[v, u] = (0, 255, 0)
                    
                    points_3d = np.array(points_3d)
                    depth = np.array(depth_key)
        
        if self.config['gemini_mask'] and not self.config['use_bypass']:
            for img in os.listdir(self.mask_dir):
                if img.endswith("_mask.png"):
                    img_path = os.path.join(self.mask_dir, img)
                    img_cnt = img.split("_")[1]
                    if img_cnt == mesh_cnt:
                        demo_image = cv2.imread(img_path)
                        demo_image = cv2.cvtColor(demo_image, cv2.COLOR_BGR2RGB)
                        # also get the mask from .json file
                        mask = self.get_mask(frame_idx, use_bypass=False)[mesh_cnt]
                        demo_image[mask==False] = (0, 0, 0)
                        
        if keypoints is not None:
            return points_3d,pcd
        else:
            return demo_image, depth, self.intrinsic, extrinsics
        
    def mask_to_point_cloud(self, frame_idx, min_depth_m=0.1, max_depth_m=3.0,is_l515=False):
        
        masks = self.get_mask(frame_idx) 
        rgb = self.get_rgb(frame_idx)
        depth = self.get_depth(frame_idx)
        
        if rgb is None or depth is None:
            return None

        width, height = rgb.shape[1], rgb.shape[0]
        rgb = cv2.resize(rgb, (int(width), int(height))).astype(np.int8)
        depth = cv2.resize(depth, (int(width), int(height)))
        
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.set_intrinsics(int(width), int(height),
                                    self.intrinsic[0, 0], self.intrinsic[1, 1],
                                    self.intrinsic[0, 2], self.intrinsic[1, 2])
        object_positions = {}
        
        for obj_name, mask in masks.items():
            masked_rgb = np.zeros_like(rgb)
            masked_rgb[mask] = rgb[mask]
            masked_depth = np.zeros_like(depth)
            masked_depth[mask] = depth[mask]
            
            
            rgbd_image = o3d.geometry.RGBDImage()
            rgbd_image = rgbd_image.create_from_color_and_depth(
                o3d.geometry.Image(masked_rgb), o3d.geometry.Image(masked_depth), 
                depth_scale=1.0, convert_rgb_to_intensity=False)
            
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_o3d, extrinsic=self.extrinsic)
            pcd = self.filter_pointclouds(pcd, 10, 2.0, 0.01)
            
            # Compute the mean (centroid) of the point cloud (object's position)
            points = np.asarray(pcd.points)  
            # object_position = np.mean(points, axis=0)  
            # object_positions[obj_name] = object_position.tolist()
            pcd_path = self.pcd_dir
            o3d.io.write_point_cloud(os.path.join(pcd_path,f"{obj_name}_frame{frame_idx}.ply"), pcd)
            print(f"Saved point cloud for {obj_name} at frame {frame_idx}")
            
