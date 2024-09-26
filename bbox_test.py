import cv2

def draw_bounding_boxes(image_path, annotation_path, output_image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    annotations = load_yolo_annotations(annotation_path)

    height, width = image.shape[:2]
    print(f"Размеры изображения: ширина = {width}, высота = {height}")

    for annotation in annotations:
        try:
            class_id = int(annotation[0])
            x_center = annotation[1]
            y_center = annotation[2]
            bbox_width = annotation[3]
            bbox_height = annotation[4]

            x1 = int((x_center - bbox_width / 2) * width)
            y1 = int((y_center - bbox_height / 2) * height)
            x2 = int((x_center + bbox_width / 2) * width)
            y2 = int((y_center + bbox_height / 2) * height)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception as e:
            print(f"Ошибка при обработке аннотации: {annotation}, ошибка: {e}")
    
    cv2.imwrite(output_image_path, image)
    print(f"Изображение с боксами сохранено: {output_image_path}")

def load_yolo_annotations(yolo_annotation_path):
    """
    Читает YOLO аннотационный файл и возвращает список аннотаций.
    
    :param yolo_annotation_path: Путь к аннотационному файлу YOLO.
    :return: Список аннотаций в формате [[class, x_center, y_center, width, height], ...].
    """
    annotations = []
    try:
        with open(yolo_annotation_path, 'r') as f:
            for line in f:
                annotation = [float(x) for x in line.strip().split()]
                annotations.append(annotation)
        print(f"Аннотации успешно загружены из {yolo_annotation_path}")
    except FileNotFoundError:
        print(f"Файл аннотаций не найден: {yolo_annotation_path}")
    except Exception as e:
        print(f"Произошла ошибка при чтении файла аннотаций: {str(e)}")
    
    return annotations

# Пример использования:
image_path = '/Users/misha/Desktop/GitHub/Augment-X/Warp-D-V2-20_18_12/test/images/Monitoring_photo_2_test_25-Mar_11-09-46.jpg'
annotation_path = '/Users/misha/Desktop/GitHub/Augment-X/Warp-D-V2-20_18_12/test/labels/Monitoring_photo_2_test_25-Mar_11-09-46.txt'
output_image_path = '/Users/misha/Desktop/GitHub/Augment-X/output_image_with_boxes.jpg'

draw_bounding_boxes(image_path, annotation_path, output_image_path)
