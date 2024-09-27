from pathlib import Path
from core.utils import get_basename
from core.utils import get_basename, get_stem, get_suffix, set_new_filename

class AnnotationProcessor:
    def __init__(self, annotation_path, save_annotation_path):
        self.annotation_path = annotation_path
        self.annotation_basename = get_basename(self.annotation_path)
        self.annotation_stem_name = get_stem(self.annotation_basename)
        self.annotation_suffix_name = get_suffix(self.annotation_basename)
        self.annotation = self.open_annotation(self.annotation_path)
        self.save_annotation_path = save_annotation_path


    def open_annotation(self, annotation_path):
        annotations = []
        with open(annotation_path, 'r') as f:
            for line in f:
                annotation = [float(x) for x in line.strip().split()]
                annotations.append(annotation)
        return annotations
    
    
    def save_yolo_annotations(self, annotations, name, preprocessing):
        if not preprocessing:
            new_annotation_path = Path(self.save_annotation_path) / name
            with open(new_annotation_path, 'w') as f:
                for annotation in annotations:
                    f.write(" ".join(map(str, annotation)) + '\n')
        else:
            self.annotation = annotations
    
    #REFACTOR THIS 
    #change size
    def change_size_annotation(self, original_width, original_height, new_width, new_height, preprocess, **kwargs):
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

        self.save_yolo_annotations(annotations=new_annotations,
                                    name=self.annotation_basename,
                                    preprocessing=preprocess)
        
        
    #crop    
    def crop_annotations(self, crop_left, crop_right, crop_top, crop_bottom, original_width, original_height, preprocess):
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

        self.save_yolo_annotations(annotations=new_annotations,
                                    name=self.annotation_basename,
                                    preprocessing=preprocess)
        
    #save basic annotation after preprocessing    
    def preprocessing_save_annotation(self, preprocess, **kwargs):
        self.save_yolo_annotations(annotations=self.annotation,
                                    name=self.annotation_basename,
                                    preprocessing=preprocess)

        
    #flip horizontal 
    def flip_horizontal_annotation(self, preprocess):
        new_annotations = []
        for ann in self.annotation:
            if ann:
                class_id, x_center, y_center, width, height = ann
                x_center = 1.0 - x_center
                x_center = self._get_limitations(x_center)
                y_center = self._get_limitations(y_center)

                new_ann = [class_id, x_center, y_center, width, height]
                new_annotations.append(new_ann)

        new_name = set_new_filename(stem=self.annotation_stem_name, 
                                    augmentation='flip_horizontal', suffix=self.annotation_suffix_name)
        
        self.save_yolo_annotations(annotations=new_annotations,
                                    name=new_name,
                                    preprocessing=preprocess)
        
    #flip vertical
    def flip_vertical_annotation(self, preprocess):
        new_annotations = []
        for ann in self.annotation:
            if ann:
                class_id, x_center, y_center, width, height = ann
                y_center = 1.0 - y_center
                x_center = self._get_limitations(x_center)
                y_center = self._get_limitations(y_center)

                new_ann = [class_id, x_center, y_center, width, height]
                new_annotations.append(new_ann)

        new_name = set_new_filename(stem=self.annotation_stem_name, 
                                    augmentation='flip_vertical', suffix=self.annotation_suffix_name)
        
        self.save_yolo_annotations(annotations=new_annotations,
                                    name=new_name,
                                    preprocessing=preprocess)
    
     #flip vertical
    def flip_both_annotation(self, preprocess):
        new_annotations = []
        for ann in self.annotation:
            if ann:
                class_id, x_center, y_center, width, height = ann
                x_center = 1.0 - x_center
                y_center = 1.0 - y_center
                x_center = self._get_limitations(x_center)
                y_center = self._get_limitations(y_center)

                new_ann = [class_id, x_center, y_center, width, height]
                new_annotations.append(new_ann)

        new_name = set_new_filename(stem=self.annotation_stem_name, 
                                    augmentation='flip_both', suffix=self.annotation_suffix_name)
        
        self.save_yolo_annotations(annotations=new_annotations,
                                    name=new_name,
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
    
