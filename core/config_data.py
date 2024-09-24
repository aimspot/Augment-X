from pathlib import Path
import yaml

task_path = Path('configs') / 'task.yaml'
image_path = Path('configs') / 'images.yaml'

def open_config(config_path):
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data

def get_data_type():
    config_data = open_config(task_path)
    data_type = config_data.get("data_type")
    return data_type
#------------------------------------------IMAGE-----------------------------------
def get_data_path():
    config_data = open_config(image_path)
    data_path = config_data.get("data_path")
    return data_path


def get_classes_path():
    config_data = open_config(image_path)
    classes_txt_path = config_data.get("classes_txt_path")
    return classes_txt_path

def get_image_config():
    config_data = open_config(image_path)
    return config_data

