import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import cv2
import pickle

def extract_features(img_path, model):
    """Извлекает признаки из изображения с использованием предобученной модели VGG16."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

def preprocess_images_cnn(input_folder, output_folder, descriptors_file):
    """Преобразует изображения, извлекает признаки с помощью CNN и сохраняет их."""
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    image_data = {}

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Ошибка загрузки {img_path}")
            continue
        
        # Сохраняем изображение в выходную папку
        cv2.imwrite(output_path, img)
        
        # Извлекаем признаки
        features = extract_features(img_path, model)
        image_data[filename] = features
        
        print(f"Обработано: {filename}")
    
    # Сохраняем признаки в файл
    with open(descriptors_file, "wb") as f:
        pickle.dump(image_data, f)
    print("Признаки сохранены.")

def find_top_similar_cnn(input_image_path, descriptors_file, top_n=5):
    """Ищет наиболее похожие изображения с использованием CNN."""
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    
    # Извлечение признаков для входного изображения
    input_features = extract_features(input_image_path, model)

    # Загружаем признаки из файла
    with open(descriptors_file, "rb") as f:
        database_features = pickle.load(f)

    # Поиск наиболее похожих изображений
    results = []
    for filename, features in database_features.items():
        similarity = np.dot(input_features, features) / (np.linalg.norm(input_features) * np.linalg.norm(features))
        results.append((similarity, filename))

    results.sort(reverse=True, key=lambda x: x[0])
    return results

def display_results_cnn(input_image_path, results, output_folder, grid_size=(2, 3), image_size=(256, 256)):
    """Отображает результаты в цветном формате в виде сетки."""
    input_img = cv2.imread(input_image_path)
    if input_img is None:
        print("Ошибка загрузки входного изображения!")
        return

    input_img_resized = cv2.resize(input_img, image_size)
    images = [input_img_resized]

    for _, filename in results[:5]:
        img_path = os.path.join(output_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(cv2.resize(img, image_size))

    # Заполняем недостающие ячейки пустыми изображениями
    while len(images) < grid_size[0] * grid_size[1]:
        images.append(np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8))

    # Создаем сетку изображений
    rows = []
    for i in range(0, len(images), grid_size[1]):
        row = np.hstack(images[i:i + grid_size[1]])
        rows.append(row)

    collage = np.vstack(rows)
    cv2.imshow("CNN Results", collage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()