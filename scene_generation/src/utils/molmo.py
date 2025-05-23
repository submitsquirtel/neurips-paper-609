from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import numpy as np
import yaml
import cv2
import gc

class ForegroundPointExtractor:
    def __init__(self, config_path='configs/config_rh20t.yaml', model_name='allenai/Molmo-7B-D-0924'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.cache_dir = self.config.get('molmo', {}).get('cache_dir', None)
        self.text_prompt = self.config.get('molmo', {}).get('text', 'Please point me to all foreground objects in the scene')
        if self.cache_dir is None:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto',
            )
        else:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto',
                cache_dir=self.cache_dir
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto',
                cache_dir=self.cache_dir
            )
    
    def extract_points(self, image_path, output_path, text_prompt=None):
        image = Image.open(image_path)
        width, height = image.size
        if text_prompt is None:
            text_prompt = self.text_prompt.strip()
        else:
            # text_prompt = "Please give points to only the distinct foreground objects in the scene based on the following list:" + text_prompt + "Avoid marking large background areas like fabric, tabletops, or entire surfaces."
            # text_prompt = "Please point me to the following foreground objects in the scene:" + text_prompt + "ignore the background like fabric, tabletops, or entire surfaces."
            text_prompt = "Point to all the " + text_prompt + " in the scene"
            # text_prompt = "Please point me to the contour of " + text_prompt + " in the scene"
        inputs = self.processor.process(
            images=[image], 
            text=text_prompt,
        )
        
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer
            )
        
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        points = self.parse_points(generated_text, width, height)
        # Draw points on the original image use cv2
        image = cv2.imread(image_path)
        if points is None:
            return None
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 5, (255, 255, 0), -1)
        cv2.putText(
                image, text_prompt, (10, image.shape[0] - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 230, 230), 2, cv2.LINE_AA
            )
        cv2.imwrite(output_path, image)
        del output, inputs
        torch.cuda.empty_cache()
        gc.collect() 
        return points
        
    
    def parse_points(self, text, width, height):
        import re
        pattern = r'(x\d*?)="([\d\.]+)" (y\d*?)="([\d\.]+)"'
        matches = re.findall(pattern, text)
        if matches == []:
            return None
        points = np.array([(float(x), float(y)) for _, x, _, y in matches])
        points[:, 0] = points[:, 0] / 100 * width
        points[:, 1] = points[:, 1] / 100 * height
        return points 

