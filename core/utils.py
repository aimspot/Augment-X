import os
from datetime import datetime

def get_current_time():
    """
    Возвращает текущее время в формате hh_mm_ss.
    """
    current_time = datetime.now().strftime("%H_%M_%S")
    return current_time


def create_new_directory(path):
    """
    Создаёт новую директорию по указанному пути.
    """
    os.makedirs(path, exist_ok=False)


def get_parent_directory(path):
    """
    Возвращает родительскую директорию для path.
    """
    parent_dir = os.path.dirname(path)
    return parent_dir


def count_similar_folders(path, folder_name):
    folder_name_lower = folder_name.lower()
    count = 0
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            if folder_name_lower in dir_name.lower():
                count += 1
    return count


def get_basename(path):
    """
    Возвращает базовое имя директории path.
    """
    base_name = os.path.basename(path)
    return base_name


def replicate_directory_structure(old_dir_path, new_dir_path):
    """
    Воспроизводит структуру поддиректорий из data_path внутри new_dir_path.
    """
    for root, dirs, files in os.walk(old_dir_path):
        rel_path = os.path.relpath(root, old_dir_path)
        target_dir = os.path.join(new_dir_path, rel_path)
        for d in dirs:
            dir_path = os.path.join(target_dir, d)
            os.makedirs(dir_path, exist_ok=True)
           