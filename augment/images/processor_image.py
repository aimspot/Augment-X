import cv2
from core.utils import get_basename, get_stem, get_suffix, set_new_filename
from augment.images.processor_annotation import AnnotationProcessor

class ImageProcessor:
    def __init__(self, image_path, annotation_path, save_image_path, save_annotation_path):
        self.image_path = image_path
        self.image_basename = get_basename(self.image_path)
        self.image_stem_name = get_stem(self.image_basename)
        self.image_suffix_name = get_suffix(self.image_basename)
        self.image = self.open_image(self.image_path)
        self.save_image_path = save_image_path

        self.ap = AnnotationProcessor(annotation_path=annotation_path,
                                 save_annotation_path=save_annotation_path)
    
    #open
    def open_image(self, image_path):
        return cv2.imread(image_path)
    
    #resize 
    def change_size_image(self, w_img, h_img, save_proportions, preprocess, **kwargs):
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

        self.ap.change_size_annotation(original_width=original_width,
                                    original_height=original_height,
                                    new_height=new_height,
                                    new_width=new_width,
                                    preprocess=preprocess)
        
        self._save_image(name=self.image_basename, img=resized_img, preprocessing=preprocess)

    
    #crop
    def crop_image(self, crop_left, crop_right, crop_top, crop_bottom, preprocess, **kwargs):
        img = self.image
        original_height, original_width = img.shape[:2]
        new_left = crop_left
        new_right = original_width - crop_right
        new_top = crop_top
        new_bottom = original_height - crop_bottom
        cropped_img = img[new_top:new_bottom, new_left:new_right]
        self._save_image(name=self.image_basename, img=cropped_img, preprocessing=preprocess)
        self.ap.crop_annotations(crop_left=crop_left, crop_right=crop_right, 
                               crop_top=crop_top, crop_bottom=crop_bottom, 
                               original_width=original_width, original_height=original_height,
                               preprocess=preprocess)
        
    #save basic image after preprocessing
    def preprocessing_save_image(self, preprocess, **kwargs):
        self.ap.preprocessing_save_annotation(preprocess=preprocess)
        self._save_image(name=self.image_basename, img=self.image, preprocessing=preprocess)

        
    #flip
    def flip_image(self, image, flip_code):
        return cv2.flip(self.image, flip_code)
    
    
    #flip horizontal
    def flip_horizontal_image(self, preprocess, **kwargs):
        image = self.flip_image(image=self.image, flip_code=1)
        new_name = set_new_filename(stem=self.image_stem_name, 
                                    augmentation='flip_horizontal', suffix=self.image_suffix_name)
        self._save_image(name=new_name, img=image, preprocessing=preprocess)
        self.ap.flip_horizontal_annotation(preprocess=preprocess)


    #flip vertical
    def flip_vertical_image(self, preprocess, **kwargs):
        image = self.flip_image(image=self.image, flip_code=0)
        new_name = set_new_filename(stem=self.image_stem_name, 
                                    augmentation='flip_vertical', suffix=self.image_suffix_name)
        self._save_image(name=new_name, img=image, preprocessing=preprocess)
        self.ap.flip_vertical_annotation(preprocess=preprocess)


    #flip-both
    def flip_both_image(self, preprocess, **kwargs):
        image = self.flip_image(image=self.image, flip_code=-1)
        # image = self.flip_image(image=image, flip_code=1)
        new_name = set_new_filename(stem=self.image_stem_name, 
                                    augmentation='flip_both', suffix=self.image_suffix_name)
        self._save_image(name=new_name, img=image, preprocessing=preprocess)
        self.ap.flip_both_annotation(preprocess=preprocess)



    #save
    def _save_image(self, name, img, preprocessing):
        if not preprocessing:
            cv2.imwrite(self.save_image_path / name, img)
        else:
            self.image = img