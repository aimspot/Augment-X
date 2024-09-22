from augment.images.augment_image import AugmentImageDataset
from core.config_data import get_data_type, get_data_path, get_classes_path


def main():
    data_type = get_data_type()
    if data_type == "images":
        aid = AugmentImageDataset(data_path=get_data_path(), classes_path=get_classes_path())



if __name__ == "__main__":
    main()