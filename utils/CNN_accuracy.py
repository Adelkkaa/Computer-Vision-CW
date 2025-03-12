from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

def calculate_metrics(input_filename, top_matches):
    """
    Вычисляет Precision, Recall и F1-Score.
    """
    target_substring = input_filename.split('.')[0]  # Берем часть до расширения
    y_true = [1 if target_substring in filename else 0 for _, filename in top_matches]
    y_pred = [1] * len(top_matches)  # Все предсказания считаем положительными

    target_substring = input_filename.split('.')[0]  # Берем часть до расширения
    correct_matches = sum(1 for _, filename in top_matches[:5] if target_substring in filename)
    accuracy = (correct_matches / len(top_matches[:5])) if top_matches else 0

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


