import os
import cv2
from tqdm import tqdm
from pathlib import Path
from core.config_data import get_image_config
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
            data_dict[basename]['image'] = Path(images_path) / f
        for f in label_files:
            basename, ext = os.path.splitext(f)
            if basename not in data_dict:
                data_dict[basename] = {'image': None, 'label': None}
            data_dict[basename]['label'] = Path(labels_path) / f
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
        for key, sub_dict in self.all_data.items():
            for sub_key, value in tqdm(sub_dict.items(), desc=f"Processing folder {key}"):
                ap = AugmentProcessor(image_path=value['image'], 
                                      annotation_path=value['label'],
                                      save_folder_path=self.new_data_path)
        


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


#--------------------------------------------------------Processor------------------------------------------------------
class AugmentProcessor():
    def __init__(self, image_path, annotation_path, save_folder_path):
        self.image_path = image_path
        self.annotation_path = annotation_path
        
        self.save_folder_path = save_folder_path
        self.image_folder_type = self._get_relative_difference(current_path = self.image_path, save_path=self.save_folder_path)
        self.annotation_folder_type = self._get_relative_difference(current_path = self.annotation_path, save_path=self.save_folder_path)
        
        self.save_image_path = Path(self.save_folder_path) / self.image_folder_type
        self.save_annotation_path = Path(self.save_folder_path) / self.annotation_folder_type

        self.image_basename = get_basename(self.image_path)
        self.annotation_basename = get_basename(self.annotation_path)

        self.image = self._open_image()
        self.annotation = self._open_annotation()
        # print(self.annotation_path)
        # print(self.annotation)
        self.processing_augmentation()


    def _get_relative_difference(self, current_path, save_path):
        current_path = Path(current_path)
        save_path = Path(save_path)
        common_parts_length = len([part1 for part1, part2 in zip(current_path.parts, save_path.parts) if part1 == part2])
        relative_path = str(Path(*current_path.parts[common_parts_length:]))
        parts = relative_path.split(os.sep)
        return get_parent_directory(str(os.path.join(*parts[1:])))
    
    #processing
    def processing_augmentation(self):
        config_data = get_image_config()
        if config_data['change_size']:
            self._change_size_image(width=config_data['w_img'],
                                    height=config_data['h_img'],
                                    propoptions=config_data['save_propoptions'])


    # open files
    def _open_image(self):
        return cv2.imread(self.image_path)
    
    def _open_annotation(self):
        annotations = []
        with open(self.annotation_path, 'r') as f:
            for line in f:
                annotation = [float(x) for x in line.strip().split()]
                annotations.append(annotation)
        return annotations
    
    
    
    #resize 
    def _change_size_image(self, width, height, propoptions=True):
        img = self.image
        original_height, original_width = img.shape[:2]

        if propoptions:
            aspect_ratio = original_width / original_height
            if width / height > aspect_ratio:
                new_height = height
                new_width = int(height * aspect_ratio)
            else:
                new_width = width
                new_height = int(width / aspect_ratio)
        else:
            new_width = width
            new_height = height

        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        self._change_size_annotation(original_width=original_width,
                                    original_height=original_height,
                                    new_height=new_height,
                                    new_width=new_width)

        self._save_image(name=self.image_basename, img=resized_img)

    
    def _change_size_annotation(self, original_width, original_height, new_width, new_height):
        scale_x = new_width / original_width
        scale_y = new_height / original_height
        new_annotations = []
        for annotation in self.annotation:
            try:
                class_id = int(annotation[0])
                x_center = annotation[1]
                y_center = annotation[2]
                bbox_width = annotation[3]
                bbox_height = annotation[4]

                x_center_resized = x_center * scale_x
                y_center_resized = y_center * scale_y
                bbox_width_resized = bbox_width * scale_x
                bbox_height_resized = bbox_height * scale_y
                
                new_annotations.append([class_id, x_center_resized, y_center_resized, bbox_width_resized, bbox_height_resized])
            except:
                pass


        self._save_yolo_annotations(annotations=new_annotations,
                                    name=self.annotation_basename)

   # save
    def _save_image(self, name, img):
        cv2.imwrite(self.save_image_path / name, img)


    def _save_yolo_annotations(self, annotations, name):
        new_annotation_path = Path(self.save_annotation_path) / name
        with open(new_annotation_path, 'w') as f:
            for annotation in annotations:
                f.write(" ".join(map(str, annotation)) + '\n')




