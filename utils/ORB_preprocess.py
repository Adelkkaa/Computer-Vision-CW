import cv2
import numpy as np
import os
import pickle
from pathlib import Path

def preprocess_images(input_folder, output_folder, descriptors_file, target_size=(512, 512)):
    """
    Преобразует изображения, извлекает ORB-дескрипторы и гистограммы, и сохраняет их.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    orb = cv2.ORB_create(nfeatures=500)
    image_data = {}

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        img_color = cv2.imread(input_path)
        if img_color is None:
            print(f"Ошибка загрузки {input_path}")
            continue
        
        img_resized = cv2.resize(img_color, target_size)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Извлекаем ORB-дескрипторы
        kp, des = orb.detectAndCompute(img_gray, None)
        
        # Вычисляем гистограммы
        hist = cv2.calcHist([img_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # Нормализуем гистограмму
        
        if des is not None:
            # Сохраняем только координаты и размеры ключевых точек
            keypoints = [(kp.pt, kp.size) for kp in kp]
            image_data[filename] = (keypoints, des, hist)
        
        cv2.imwrite(output_path, img_resized)
        print(f"Обработано: {filename}")
    
    with open(descriptors_file, "wb") as f:
        pickle.dump(image_data, f)
    print("Дескрипторы и гистограммы сохранены.")

if __name__ == "__main__":
    preprocess_images("e-commerce", "preprocessed_images", "descriptors.pkl", target_size=(256, 256))
