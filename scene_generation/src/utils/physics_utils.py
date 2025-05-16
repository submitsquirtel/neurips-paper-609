import json
import time
import yaml
import google.generativeai as genai

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def configure_gemini_from_config(config):
    """Set up Gemini API from config."""
    gemini_config = config['models']['gemini']
    genai.configure(api_key=gemini_config['api_key'])
    return gemini_config['model_name']


def load_object_entries(json_path):
    """Load list of object entries from JSON, each with 'object_name' and 'count'."""
    with open(json_path, 'r') as f:
        return json.load(f)


def ask_gemini_about_physical_properties(object_name, model, retries=3):
    """Query Gemini for estimated mass and friction of an object."""

    prompt = (
        f"You are an asset augmentation tool for simulation environments. Please analyze the physical properties of the object: {object_name}'.\n"
        f"Return three values: estimated with, height and length of the object, estimated mass in kilograms, coefficient of friction and surface type.\n"
        f"Use consistent and reasonable values so that similar assets receive similar properties.\n"
        f"Output surface type should be chosen from the following list: [Glass, Water, Emission, Plastic, Rough, Smooth, Reflective, Metal, Iron, Aluminium, Copper, Gold].\n"
        f"Use this format exactly: width: <value in meter>, height: <value in meter>, length: <value in meter>, mass: <value in kg>, friction: <value>, surface: <surface type>.\n"
        f"If unknown, make a reasonable guess based on common use and surface.\n"
        f"Do not ask followup questions. The output must contain no additional commentary or extraneous text."
    )

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[WARNING] Attempt {attempt+1} failed for {object_name}: {e}")
            time.sleep(2)
    return "Error: Failed to retrieve response"


def query_physical_properties_from_object_list(input_json, output_json, config_path, delay=1.0):
    """Main entry point to query Gemini and store result keyed by object count/index."""
    config = load_config(config_path)
    model_name = configure_gemini_from_config(config)
    model = genai.GenerativeModel(model_name=model_name)

    object_entries = load_object_entries(input_json)
    results = []

    for entry in object_entries:
        object_name = entry['object_name'].strip()
        count = entry['count']
        print(f"[INFO] Querying Gemini for: {object_name} (id: {count})")

        answer = ask_gemini_about_physical_properties(object_name, model)
        # extract mass, friction, and surface from the answer
        try:
            # Split the response by commas and strip whitespace
            width, height, length, mass, friction, surface = [x.strip() for x in answer.split(",")]
            # ensure width, height, length are in the format "width: <value>"
            if width.split(": ")[0] != "width":
                raise ValueError(f"Unexpected response format for width: {width}")
            width = float(width.split(": ")[1])
            if height.split(": ")[0] != "height":
                raise ValueError(f"Unexpected response format for height: {height}")
            height = float(height.split(": ")[1])
            if length.split(": ")[0] != "length":
                raise ValueError(f"Unexpected response format for length: {length}")
            length = float(length.split(": ")[1])
            # ensure mass.split(": ")[0] is "mass"
            if mass.split(": ")[0] != "mass":
                raise ValueError(f"Unexpected response format for mass: {mass}")
            mass = float(mass.split(": ")[1])
            if friction.split(": ")[0] != "friction":
                raise ValueError(f"Unexpected response format for friction: {friction}")
            friction = float(friction.split(": ")[1])
            if surface.split(": ")[0] != "surface":
                raise ValueError(f"Unexpected response format for surface: {surface}")
            surface = surface.split(": ")[1]
        
        except Exception as e:
            print(f"[ERROR] Failed to parse response for {object_name}: {e}")
            mass, friction, surface = None, None, None
        result = {
            "object_name": object_name,
            "count": count,
            "width": width,
            "height": height,
            "length": length,
            "mass": mass,
            "friction": friction,
            "surface": surface
        }
        results.append(result)
        time.sleep(delay)
    
    # Save results to JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] Results saved to {output_json}")
