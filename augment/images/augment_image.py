import os
import cv2
from tqdm import tqdm
from pathlib import Path
from core.utils import get_current_time, create_new_directory, get_parent_directory, get_parent_directory
from core.utils import get_basename, replicate_directory_structure, count_similar_folders


class AugmentImageDataset():
    def __init__(self, data_path, classes_path):
        self.data_path = data_path
        self.classes_path = classes_path
        self.new_data_path = self.create_new_dataset_folder()
        self.all_data = self.preprocess()
        self.process_dataset()
        

    # Preprocessing
    def preprocess(self):
        tmp_all_data = {}
        for subdir in os.listdir(self.data_path):
            subdir_path = os.path.join(self.data_path, subdir)
            if not os.path.isdir(subdir_path):
                continue
            data_dict = self.process_subdirectory(subdir_path)
            if data_dict:
                tmp_all_data[subdir] = data_dict
        return tmp_all_data


    def process_subdirectory(self, subdir_path):
        images_path = os.path.join(subdir_path, 'images')
        labels_path = os.path.join(subdir_path, 'labels')
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            return
        data_dict = self.create_data_dict(images_path, labels_path)
        self.remove_unpaired_files(data_dict, images_path, labels_path)
        return data_dict


    def create_data_dict(self, images_path, labels_path):
        image_files = os.listdir(images_path)
        label_files = os.listdir(labels_path)
        data_dict = {}
        for f in image_files:
            basename, ext = os.path.splitext(f)
            if basename not in data_dict:
                data_dict[basename] = {'image': None, 'label': None}
            data_dict[basename]['image'] = f
        for f in label_files:
            basename, ext = os.path.splitext(f)
            if basename not in data_dict:
                data_dict[basename] = {'image': None, 'label': None}
            data_dict[basename]['label'] = f
        return data_dict
    

    
    def remove_unpaired_files(self, data_dict, images_path, labels_path):
        for basename, files in data_dict.items():
            if files['image'] is None and files['label'] is not None:
                label_file_path = os.path.join(labels_path, files['label'])
                os.remove(label_file_path)
            elif files['label'] is None and files['image'] is not None:
                image_file_path = os.path.join(images_path, files['image'])
                os.remove(image_file_path)



    # processing
    def process_dataset(self):
        for key, sub_dict in tqdm(self.all_data.items()):
            for sub_key, value in tqdm(sub_dict.items(), desc=f"Processing folder {key}"):
                for key, value in value.items():
                    if key == 'image':
                        image = Image()
                    # elif key == 'label':
                    #     annotation = Annotation()
        


    # new dataset folder
    def create_new_dataset_folder(self):
        path = Path(self.data_path)
        parent_dir = get_parent_directory(path)
        base_name = get_basename(path)
        timestamp = get_current_time()
        version = count_similar_folders(parent_dir, base_name)
        new_dir_name = f"{base_name}-V{version}-{timestamp}"
        new_dir_path = os.path.join(parent_dir, new_dir_name)
        
        create_new_directory(new_dir_path)
        replicate_directory_structure(path, new_dir_path)
        return new_dir_path



class Image():
    def __init__(self) -> None:
        self.image_path = None
        self.image_shape = None
        self.image_name = None

    def get_image_name(self):
        return self.image_name
    
    def save_image(self, name, path, img):
        cv2.imwrite('output.jpg', img)


class Annotation():
    def __init__(self) -> None:
        self.annotaion_name = None
        self.annotaion_path = None

    def get_annotation_name(self):
        return self.annotaion_name




