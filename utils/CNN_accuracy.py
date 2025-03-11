from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd


def calculate_precision_at_k(ground_truth, query_image, top_matches, k=5):
    """
    Вычисляет Precision@K для заданного запроса.
    
    :param ground_truth: Словарь с эталонными данными.
    :param query_image: Имя файла запроса (например, "image1.jpg").
    :param top_matches: Список кортежей (сходство, имя файла) с результатами модели.
    :param k: Количество первых результатов для оценки (по умолчанию 5).
    :return: Precision@K.
    """
    if query_image not in ground_truth:
        raise ValueError(f"Запрос {query_image} отсутствует в эталонных данных.")
    
    # Получаем правильные результаты для запроса
    correct_matches = set(ground_truth[query_image])
    
    # Берем первые K результатов из top_matches
    top_k_matches = [filename for _, filename in top_matches[:k]]
    
    # Считаем количество правильных результатов среди первых K
    correct_count = sum(1 for filename in top_k_matches if filename in correct_matches)
    
    # Precision@K = количество правильных / K
    precision_at_k = correct_count / k
    return precision_at_k

def calculate_metrics(input_filename, top_matches):
    """
    Вычисляет Precision, Recall и F1-Score.
    """
    target_substring = input_filename.split('.')[0]  # Берем часть до расширения
    y_true = [1 if target_substring in filename else 0 for _, filename in top_matches]
    y_pred = [1] * len(top_matches)  # Все предсказания считаем положительными

    correct_matches = sum(1 for _, filename in top_matches if target_substring in filename)

    accuracy = (top_matches[0][0]) if top_matches else 0

    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return accuracy, recall, f1



def plot_metrics(precision, recall, f1):
    """Визуализирует метрики в виде bar plot."""
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [precision, recall, f1]

    plt.bar(metrics, values, color=['blue', 'green', 'red'])
    plt.ylabel('Score')
    plt.title('Metrics')
    plt.show()

def display_results_table(top_matches):
    """Отображает результаты в виде таблицы."""
    df = pd.DataFrame(top_matches, columns=['Similarity', 'Filename'])
    print(df)


