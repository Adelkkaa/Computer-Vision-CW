def calculate_accuracy(input_filename, top_matches):
    """
    Вычисляет точность на основе совпадений подстроки в названии файла.
    """
    target_substring = input_filename.split('.')[0]  # Берем часть до расширения
    correct_matches = sum(1 for _, filename in top_matches if target_substring in filename)
    print(top_matches)
    accuracy = (correct_matches / len(top_matches)) * 100 if top_matches else 0
    return accuracy, correct_matches