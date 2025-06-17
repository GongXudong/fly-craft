import json

def load_config(config_name: str):
    
    with open(str(config_name), "r") as f:
        return json.load(f)