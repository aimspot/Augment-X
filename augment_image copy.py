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

        self.image = self._open_image(self.image_path)
        self.annotation = self._open_annotation(self.annotation_path)

        self.function_mapping = {
            'resize_image': self._change_size_image,
            'crop_image': self._crop_image
        }

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
        preprocessing = config_data["preprocessing"]
        augmentations = config_data["augmentations"]
        for prepare in preprocessing:
            self.function_mapping[prepare](preprocess=True, **config_data)
        for augmentation in augmentations:
            self.function_mapping[augmentation](preprocess=False, **config_data)



    # open files
    def _open_image(self, image_path):
        return cv2.imread(image_path)
    
    
    def _open_annotation(self, annotation_path):
        annotations = []
        with open(annotation_path, 'r') as f:
            for line in f:
                annotation = [float(x) for x in line.strip().split()]
                annotations.append(annotation)
        return annotations
    
    
    
    #resize 
    def _change_size_image(self, w_img, h_img, save_proportions, preprocess, **kwargs):
        img = self.image
        original_height, original_width = img.shape[:2]

        if save_proportions:
            aspect_ratio = original_width / original_height
            if w_img / h_img > aspect_ratio:
                new_height = h_img
                new_width = int(h_img * aspect_ratio)
            else:
                new_width = w_img
                new_height = int(w_img / aspect_ratio)
        else:
            new_width = w_img
            new_height = h_img

        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        self._change_size_annotation(original_width=original_width,
                                    original_height=original_height,
                                    new_height=new_height,
                                    new_width=new_width,
                                    preprocess=preprocess)
        
        self._save_image(name=self.image_basename, img=resized_img, preprocessing=preprocess)

    
    def _change_size_annotation(self, original_width, original_height, new_width, new_height, preprocess, **kwargs):
        new_annotations = []
        scale_x = new_width / original_width
        scale_y = new_height / original_height
        for ann in self.annotation:
            if ann:
                class_id, x_center, y_center, width, height = ann
                x_center_abs, y_center_abs, width_abs, height_abs = self._get_absolute_coordinates(x_center = x_center,
                                                                                                   y_center=y_center,
                                                                                                   width=width,
                                                                                                   height=height,
                                                                                                   original_height=original_height,
                                                                                                   original_width=original_width)
                x_center_new = x_center_abs * scale_x / new_width
                y_center_new = y_center_abs * scale_y / new_height
                width_new = width_abs * scale_x / new_width
                height_new = height_abs * scale_y / new_height

                x_center_new = self._get_limitations(x_center_new)
                y_center_new = self._get_limitations(y_center_new)
                width_new = self._get_limitations(width_new)
                height_new = self._get_limitations(height_new)

                new_ann = [class_id, x_center_new, y_center_new, width_new, height_new]
                new_annotations.append(new_ann)

        self._save_yolo_annotations(annotations=new_annotations,
                                    name=self.annotation_basename,
                                    preprocessing=preprocess)
        

    #crop
    def _crop_image(self, crop_left, crop_right, crop_top, crop_bottom, preprocess, **kwargs):
        img = self.image
        original_height, original_width = img.shape[:2]
        new_left = crop_left
        new_right = original_width - crop_right
        new_top = crop_top
        new_bottom = original_height - crop_bottom
        cropped_img = img[new_top:new_bottom, new_left:new_right]
        self._save_image(name=self.image_basename, img=cropped_img, preprocessing=preprocess)
        self._crop_annotations(crop_left=crop_left, crop_right=crop_right, 
                               crop_top=crop_top, crop_bottom=crop_bottom, 
                               original_width=original_width, original_height=original_height,
                               preprocess=preprocess)


    def _crop_annotations(self, crop_left, crop_right, crop_top, crop_bottom, original_width, original_height, preprocess):
        new_annotations = []
        new_width = original_width - crop_left - crop_right
        new_height = original_height - crop_top - crop_bottom
        for ann in self.annotation:
            if ann:
                class_id, x_center, y_center, width, height = ann
                x_center_abs, y_center_abs, width_abs, height_abs = self._get_absolute_coordinates(x_center=x_center,
                                                                                                   y_center=y_center,
                                                                                                   width=width,
                                                                                                   height=height,
                                                                                                   original_height=original_height,
                                                                                                   original_width=original_width)
                x_center_abs -= crop_left
                y_center_abs -= crop_top
                if x_center_abs < 0 or y_center_abs < 0 or x_center_abs > new_width or y_center_abs > new_height:
                    continue

                x_center_new = x_center_abs / new_width
                y_center_new = y_center_abs / new_height
                width_new = width_abs / new_width
                height_new = height_abs / new_height
                x_center_new = self._get_limitations(x_center_new)
                y_center_new = self._get_limitations(y_center_new)
                width_new = self._get_limitations(width_new)
                height_new = self._get_limitations(height_new)

                new_ann = [class_id, x_center_new, y_center_new, width_new, height_new]
                new_annotations.append(new_ann)

        self._save_yolo_annotations(annotations=new_annotations,
                                    name=self.annotation_basename,
                                    preprocessing=preprocess)


    # annotation extra function
    def _get_absolute_coordinates(self, x_center, y_center, width, height, original_width, original_height):
        x_center_abs = x_center * original_width
        y_center_abs = y_center * original_height
        width_abs = width * original_width
        height_abs = height * original_height
        return x_center_abs, y_center_abs, width_abs, height_abs
    
    def _get_limitations(self, value):
        return min(max(value, 0), 1)

   # save
    def _save_image(self, name, img, preprocessing):
        if not preprocessing:
            cv2.imwrite(self.save_image_path / name, img)
        else:
            self.image = img


    def _save_yolo_annotations(self, annotations, name, preprocessing):
        if not preprocessing:
            new_annotation_path = Path(self.save_annotation_path) / name
            with open(new_annotation_path, 'w') as f:
                for annotation in annotations:
                    f.write(" ".join(map(str, annotation)) + '\n')
        else:
            self.annotation = annotations




