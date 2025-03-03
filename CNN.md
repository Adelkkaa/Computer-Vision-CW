### 1. **Функция `extract_features`**

```python
def extract_features(img_path, model):
    """Извлекает признаки из изображения с использованием предобученной модели VGG16."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()
```

#### Как это работает:
1. **`img = image.load_img(img_path, target_size=(224, 224))`**:
   - Загружает изображение по указанному пути `img_path`.
   - Изменяет размер изображения до 224x224 пикселей (это требуется для модели VGG16).

2. **`img_data = image.img_to_array(img)`**:
   - Преобразует изображение в массив NumPy. Теперь это трехмерный массив (высота, ширина, каналы).

3. **`img_data = np.expand_dims(img_data, axis=0)`**:
   - Добавляет дополнительную ось к массиву, чтобы он стал четырехмерным (1, высота, ширина, каналы). Это нужно, потому что модель VGG16 ожидает на вход пакет изображений (batch), даже если изображение одно.

4. **`img_data = preprocess_input(img_data)`**:
   - Предобрабатывает изображение для модели VGG16. Это включает нормализацию пикселей и приведение их к формату, который ожидает модель.

5. **`features = model.predict(img_data)`**:
   - Пропускает изображение через модель VGG16 и извлекает признаки. На выходе получается многомерный массив (тензор).

6. **`return features.flatten()`**:
   - "Разворачивает" многомерный массив в одномерный (вектор). Это нужно для удобства сравнения признаков.

---

### 2. **Функция `preprocess_images_cnn`**

```python
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
        
        cv2.imwrite(output_path, img)
        
        features = extract_features(img_path, model)
        image_data[filename] = features
        
        print(f"Обработано: {filename}")
    
    with open(descriptors_file, "wb") as f:
        pickle.dump(image_data, f)
    print("Признаки сохранены.")
```

#### Как это работает:
1. **`model = VGG16(weights='imagenet', include_top=False, pooling='avg')`**:
   - Загружает предобученную модель VGG16 без верхних слоев (только сверточные слои) и с глобальным усреднением (Global Average Pooling).

2. **`image_data = {}`**:
   - Создает пустой словарь для хранения признаков изображений.

3. **Цикл `for filename in os.listdir(input_folder)`**:
   - Проходит по всем файлам в папке `input_folder`.

4. **`img_path = os.path.join(input_folder, filename)`**:
   - Формирует полный путь к изображению.

5. **`img = cv2.imread(img_path)`**:
   - Загружает изображение с помощью OpenCV.

6. **`if img is None:`**:
   - Проверяет, удалось ли загрузить изображение. Если нет, выводит сообщение об ошибке и переходит к следующему файлу.

7. **`cv2.imwrite(output_path, img)`**:
   - Сохраняет изображение в папку `output_folder`.

8. **`features = extract_features(img_path, model)`**:
   - Извлекает признаки изображения с помощью функции `extract_features`.

9. **`image_data[filename] = features`**:
   - Сохраняет признаки в словарь `image_data` под ключом, равным имени файла.

10. **`with open(descriptors_file, "wb") as f:`**:
    - Открывает файл `descriptors_file` для записи в бинарном режиме.

11. **`pickle.dump(image_data, f)`**:
    - Сериализует словарь `image_data` и сохраняет его в файл.

---

### 3. **Функция `find_top_similar_cnn`**

```python
def find_top_similar_cnn(input_image_path, descriptors_file, top_n=5):
    """Ищет наиболее похожие изображения с использованием CNN."""
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    
    input_features = extract_features(input_image_path, model)

    with open(descriptors_file, "rb") as f:
        database_features = pickle.load(f)

    results = []
    for filename, features in database_features.items():
        similarity = np.dot(input_features, features) / (np.linalg.norm(input_features) * np.linalg.norm(features))
        results.append((similarity, filename))

    results.sort(reverse=True, key=lambda x: x[0])
    return results[:top_n]
```

#### Как это работает:
1. **`model = VGG16(weights='imagenet', include_top=False, pooling='avg')`**:
   - Загружает модель VGG16.

2. **`input_features = extract_features(input_image_path, model)`**:
   - Извлекает признаки для входного изображения.

3. **`with open(descriptors_file, "rb") as f:`**:
   - Открывает файл с сохраненными признаками изображений.

4. **`database_features = pickle.load(f)`**:
   - Загружает словарь с признаками из файла.

5. **Цикл `for filename, features in database_features.items()`**:
   - Проходит по всем изображениям в базе данных.

6. **`similarity = np.dot(input_features, features) / (np.linalg.norm(input_features) * np.linalg.norm(features))`**:
   - Вычисляет косинусное сходство между признаками входного изображения и текущего изображения из базы данных.

7. **`results.append((similarity, filename))`**:
   - Добавляет результат (сходство и имя файла) в список `results`.

8. **`results.sort(reverse=True, key=lambda x: x[0])`**:
   - Сортирует результаты по убыванию сходства.

9. **`return results[:top_n]`**:
   - Возвращает топ-N наиболее похожих изображений.

---

### 4. **Функция `display_results_cnn`**

```python
def display_results_cnn(input_image_path, results, output_folder, grid_size=(2, 3), image_size=(256, 256)):
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

    while len(images) < grid_size[0] * grid_size[1]:
        images.append(np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8))

    rows = []
    for i in range(0, len(images), grid_size[1]):
        row = np.hstack(images[i:i + grid_size[1]])
        rows.append(row)

    collage = np.vstack(rows)
    cv2.imshow("CNN Results", collage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

#### Как это работает:
1. **`input_img = cv2.imread(input_image_path)`**:
   - Загружает входное изображение.

2. **`input_img_resized = cv2.resize(input_img, image_size)`**:
   - Изменяет размер входного изображения.

3. **Цикл `for _, filename in results:`**:
   - Загружает и изменяет размер изображений из результатов.

4. **`while len(images) < grid_size[0] * grid_size[1]:`**:
   - Добавляет пустые изображения, если результатов меньше, чем ячеек в сетке.

5. **Создание сетки**:
   - Объединяет изображения в сетку с помощью `np.hstack` и `np.vstack`.

6. **`cv2.imshow("CNN Results", collage)`**:
   - Отображает сетку изображений.

---

### Итог:
- **Этап 1**: Извлечение признаков из изображений и сохранение их в файл.
- **Этап 2**: Поиск похожих изображений по сохраненным признакам.
- **Этап 3**: Визуализация результатов.
