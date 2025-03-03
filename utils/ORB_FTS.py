import cv2
import numpy as np
import os
import pickle
from pathlib import Path

def find_top_similar(input_image_path, descriptors_file, top_n=5):
    """Загружает дескрипторы и гистограммы, ищет наиболее похожие изображения."""
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Загружаем входное изображение
    input_img = cv2.imread(input_image_path)
    if input_img is None:
        raise ValueError("Ошибка загрузки входного изображения!")
    input_img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(input_img_gray, None)

    # Вычисляем гистограмму входного изображения
    input_hist = cv2.calcHist([input_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    input_hist = cv2.normalize(input_hist, input_hist).flatten()

    # Загружаем дескрипторы и гистограммы базы
    with open(descriptors_file, "rb") as f:
        image_data = pickle.load(f)
    
    results = []
    for filename, (keypoints_data, des2, hist2) in image_data.items():
        if des2 is None or des1 is None:
            continue

        # Преобразуем кортежи обратно в объекты cv2.KeyPoint
        kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=size) for pt, size in keypoints_data]

        # Сравниваем ORB-дескрипторы
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.70 * n.distance]

        if len(good_matches) > 4:  # Минимум 4 точки для гомографии
            # Преобразуем ключевые точки в координаты
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Находим гомографию с помощью RANSAC
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            mask = mask.ravel().tolist()

            # Фильтруем совпадения с учетом RANSAC
            filtered_matches = [m for m, is_valid in zip(good_matches, mask) if is_valid]

            if len(filtered_matches) > 4:  # Минимум 4 точки после фильтрации
                # Вычисляем ORB-оценку на основе отфильтрованных совпадений
                orb_score = sum(m.distance for m in filtered_matches) / len(filtered_matches)
                orb_weight = min(len(filtered_matches) / 15, 1.0)  # Вес ORB-оценки (от 0 до 1)
            else:
                orb_score = 1.0  # Максимальное значение, если совпадений нет
                orb_weight = 0.0  # Полностью игнорируем ORB-оценку
        else:
            orb_score = 1.0  # Максимальное значение, если совпадений нет
            orb_weight = 0.0  # Полностью игнорируем ORB-оценку

        # Сравниваем гистограммы
        hist_score = cv2.compareHist(input_hist, hist2, cv2.HISTCMP_CORREL)

        # Комбинируем оценки
        combined_score = (orb_score * orb_weight) + (1 - hist_score) * (1 - orb_weight)
        results.append((combined_score, filename))

    results.sort(key=lambda x: x[0])

    return results[:top_n]

def display_results(input_image_path, results, output_folder, grid_size=(2, 3), image_size=(256, 256)):
    """Отображает результаты в цветном формате в виде сетки."""
    input_img = cv2.imread(input_image_path)
    if input_img is None:
        print("Ошибка загрузки входного изображения!")
        return

    input_img_resized = cv2.resize(input_img, image_size)
    images = [input_img_resized]

    for _, filename in results:
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
    cv2.imshow("Results", collage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    input_image_path = "test-images/apparel.jpg"
    descriptors_file = "descriptors.pkl"
    output_folder = "preprocessed_images"
    
    top_matches = find_top_similar(input_image_path, descriptors_file, top_n=5)
    display_results(input_image_path, top_matches, output_folder)
