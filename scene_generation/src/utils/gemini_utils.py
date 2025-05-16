# @title Segmentation Utils

import dataclasses
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import json
import io
import os
import cv2
import numpy as np

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def dilate_single_mask(mask_img, min_area=1000, dilation_kernel_size=9, dilation_iterations=4):
    """
    Apply dilation to a single mask image after filtering by contour area.

    Parameters:
        mask_img (np.ndarray): Input mask image (can be RGB or grayscale).
        min_area (int): Minimum contour area to keep during filtering.
        dilation_kernel_size (int): Size of the square kernel used for dilation.
        dilation_iterations (int): Number of dilation iterations to perform.

    Returns:
        np.ndarray: Dilated binary mask image (uint8, values 0 or 255).
    """
    # Convert RGB to grayscale if needed
    if len(mask_img.shape) == 3:
        gray = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = mask_img

    # Threshold the grayscale mask to binary
    _, thresh = cv2.threshold(gray, 127, 255, 0)

    # Optionally apply a small dilation to connect close regions before contour extraction
    thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours with area smaller than the minimum
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    if not contours:
        print("No valid contours found.")
        return np.zeros_like(gray)

    # Create a binary mask from the filtered contours
    filtered_mask = np.zeros_like(gray)
    cv2.drawContours(filtered_mask, contours, -1, 255, -1)

    # Apply dilation to the filtered binary mask
    dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(filtered_mask, dilation_kernel, iterations=dilation_iterations)

    return dilated_mask


@dataclasses.dataclass(frozen=True)
class SegmentationMask:
    # bounding box pixel coordinates (not normalized)
    y0: int # in [0..height - 1]
    x0: int # in [0..width - 1]
    y1: int # in [0..height - 1]
    x1: int # in [0..width - 1]
    mask: np.array # [img_height, img_width] with values 0..255
    label: str
    mass: float = 0.0
    friction: float = 0.0
    surface: str = "unknown"

def parse_segmentation_masks(
    predicted_str: str, *, img_height: int, img_width: int
) -> list[SegmentationMask]:
    items = json.loads(parse_json(predicted_str))
    masks = []
    for item in items:
        raw_box = item["box_2d"]
        abs_y0 = int(item["box_2d"][0] / 1000 * img_height)
        abs_x0 = int(item["box_2d"][1] / 1000 * img_width)
        abs_y1 = int(item["box_2d"][2] / 1000 * img_height)
        abs_x1 = int(item["box_2d"][3] / 1000 * img_width)
        if abs_y0 >= abs_y1 or abs_x0 >= abs_x1:
            print("Invalid bounding box", item["box_2d"])
            continue
        label = item["label"]
        png_str = item["mask"]
        if not png_str.startswith("data:image/png;base64,"):
            print("Invalid mask")
            continue
        png_str = png_str.removeprefix("data:image/png;base64,")
        png_str = base64.b64decode(png_str)
        mask = Image.open(io.BytesIO(png_str))
        bbox_height = abs_y1 - abs_y0
        bbox_width = abs_x1 - abs_x0
        if bbox_height < 1 or bbox_width < 1:
            print("Invalid bounding box")
            continue
        mask = mask.resize((bbox_width, bbox_height), resample=Image.Resampling.BILINEAR)
        np_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        np_mask[abs_y0:abs_y1, abs_x0:abs_x1] = mask
        masks.append(SegmentationMask(abs_y0, abs_x0, abs_y1, abs_x1, np_mask, label,
                                    mass=item.get("mass", 0.0),
                                    friction=item.get("friction", 0.0),
                                    surface=item.get("surface_type", "unknown")))
    return masks

def overlay_mask_on_img(
    img: Image,
    mask: np.ndarray,
    color: str,
    alpha: float = 0.7
) -> Image.Image:
    """
    Overlays a single mask onto a PIL Image using a named color.

    The mask image defines the area to be colored. Non-zero pixels in the
    mask image are considered part of the area to overlay.

    Args:
        img: The base PIL Image object.
        mask: A PIL Image object representing the mask.
              Should have the same height and width as the img.
              Modes '1' (binary) or 'L' (grayscale) are typical, where
              non-zero pixels indicate the masked area.
        color: A standard color name string (e.g., 'red', 'blue', 'yellow').
        alpha: The alpha transparency level for the overlay (0.0 fully
               transparent, 1.0 fully opaque). Default is 0.7 (70%).

    Returns:
        A new PIL Image object (in RGBA mode) with the mask overlaid.

    Raises:
        ValueError: If color name is invalid, mask dimensions mismatch img
                    dimensions, or alpha is outside the 0.0-1.0 range.
    """
    
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Alpha must be between 0.0 and 1.0")

    # Convert the color name string to an RGB tuple
    try:
        color_rgb: Tuple[int, int, int] = ImageColor.getrgb(color)
    except ValueError as e:
        # Re-raise with a more informative message if color name is invalid
        raise ValueError(f"Invalid color name '{color}'. Supported names are typically HTML/CSS color names. Error: {e}")

    # Prepare the base image for alpha compositing
    img_rgba = img.convert("RGBA")
    width, height = img_rgba.size

    # Create the colored overlay layer
    # Calculate the RGBA tuple for the overlay color
    alpha_int = int(alpha * 255)
    overlay_color_rgba = color_rgb + (alpha_int,)

    # Create an RGBA layer (all zeros = transparent black)
    colored_mask_layer_np = np.zeros((height, width, 4), dtype=np.uint8)

    # Mask has values between 0 and 255, threshold at 127 to get binary mask.
    mask_np_logical = mask > 127

    # Apply the overlay color RGBA tuple where the mask is True
    colored_mask_layer_np[mask_np_logical] = overlay_color_rgba

    # Convert the NumPy layer back to a PIL Image
    colored_mask_layer_pil = Image.fromarray(colored_mask_layer_np, 'RGBA')

    # Composite the colored mask layer onto the base image
    result_img = Image.alpha_composite(img_rgba, colored_mask_layer_pil)

    return result_img

def plot_segmentation_masks(img: Image, segmentation_masks: list[SegmentationMask], output_folder: str = None):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img: The PIL.Image.
        segmentation_masks: A string encoding as JSON a list of segmentation masks containing the name of the object,
         their positions in normalized [y1 x1 y2 x2] format, and the png encoded segmentation mask.
    """
    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors
    # font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    font = ImageFont.load_default()
    ori_image = img.copy()
    # Do this in 3 passes to make sure the boxes and text are always visible.
    # Overlay the mask
    for i, mask in enumerate(segmentation_masks):
        color = colors[i % len(colors)]
        img = overlay_mask_on_img(img, mask.mask, color)

    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Draw the bounding boxes
    for i, mask in enumerate(segmentation_masks):
        color = colors[i % len(colors)]
        draw.rectangle(
            ((mask.x0, mask.y0), (mask.x1, mask.y1)), outline=color, width=4
        )

    # Draw the text labels
    for i, mask in enumerate(segmentation_masks):
        color = colors[i % len(colors)]
        if mask.label != "":
            draw.text((mask.x0 + 8, mask.y0 - 20), mask.label, fill=color, font=font)
    # Save cropped masks excluding robot/gripper
    object_dicts = []
    robot_mask = []
    physic_dicts = []
    # extract base directory
    base_dir = os.path.dirname(output_folder)
    path2 = os.path.join(base_dir, "input_mask")
    os.makedirs(path2, exist_ok=True)
    if output_folder:
        count = 0
        for segmentation_mask in segmentation_masks:
            if 'robot' in segmentation_mask.label.lower() or 'gripper' in segmentation_mask.label.lower():
                robot_mask.append(segmentation_mask.mask)
                continue
            # 1.save the original image with mask and black background
            saved_mask = segmentation_mask.mask.copy().astype(bool)
            object_dicts.append(
                {
                    'object_name': segmentation_mask.label,
                    'mask': saved_mask.tolist(),
                    'count': count,
                }
            )
            physic_dicts.append(
                {
                    'object_name': segmentation_mask.label,
                    'count': [count],
                    'mass': segmentation_mask.mass,
                    'friction': segmentation_mask.friction,
                    'surface': segmentation_mask.surface,
                }
            )
            original_img = ori_image.copy()
            original_img = np.array(original_img) 
            original_img[segmentation_mask.mask == 0] = 0
            bnw = np.zeros_like(original_img, dtype=np.uint8)
            bnw[segmentation_mask.mask != 0] = 255
            # if bbox small adjust min_area
            bnw = dilate_single_mask(bnw, min_area=300)
            cv2.imwrite(os.path.join(path2, f"image_{count:04d}_mask.png"), bnw)
            ori_image.save(os.path.join(path2, f"image_{count:04d}.png"), "PNG")
            
            original_img = Image.fromarray(original_img)
            original_img.save(os.path.join(output_folder, f"image_{count:04d}_mask.png"), "PNG")

            # Get bounding box
            padding = 40
            img_width, img_height = original_img.size
            x0 = max(int(segmentation_mask.x0) - padding, 0)
            y0 = max(int(segmentation_mask.y0) - padding, 0)
            x1 = min(int(segmentation_mask.x1) + padding, img_width)
            y1 = min(int(segmentation_mask.y1) + padding, img_height)
            # x0, y0, x1, y1 = int(segmentation_mask.x0), int(segmentation_mask.y0), int(segmentation_mask.x1), int(segmentation_mask.y1)
            cropped_img = original_img.crop((x0, y0, x1, y1))
            path_dir = os.path.join(output_folder, "cropped")
            os.makedirs(path_dir, exist_ok=True)
            cropped_img.save(os.path.join(path_dir, f"image_{count:04d}_segmented.png"), "PNG")
            count += 1
    
        for segmentation_mask in robot_mask:
            original_img = ori_image.copy()
            original_img = np.array(original_img) 
            original_img[segmentation_mask == 0] = 0
            original_img[segmentation_mask != 0] = 255
            original_img = dilate_single_mask(original_img)
            cv2.imwrite(os.path.join(path2, f"image_{count:04d}_mask.png"), original_img)
            ori_image.save(os.path.join(path2, f"image_{count:04d}.png"), "PNG")
        # save results to output folder
        with open(os.path.join(output_folder, "result.json"), "w") as f:
            json.dump(physic_dicts, f, indent=4)
    return img, object_dicts
