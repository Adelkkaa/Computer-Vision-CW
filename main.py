# main.py
from tkinter import Tk, Button, Label, filedialog, messagebox
from PIL import Image, ImageTk
import os
from utils.ORB_preprocess import preprocess_images
from utils.ORB_FTS import find_top_similar, display_results
from utils.ORB_accuracy import calculate_accuracy
from utils.CNN_FTS import find_top_similar_cnn, display_results_cnn, preprocess_images_cnn  # Импортируем новый функционал

# === ИНТЕРФЕЙС ===
class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Поиск похожих изображений")

        self.input_folder = "test-images"
        self.preprocess_folder = "e-commerce"
        self.output_folder = "preprocessed_images"
        self.descriptors_file = "descriptors.pkl"
        self.cnn_descriptors_file = "cnn_descriptors.pkl"  # Новый файл для CNN

        # Кнопка для выбора изображения
        self.select_button = Button(root, text="Выбрать изображение", command=self.select_image)
        self.select_button.pack(pady=10)

        # Кнопка для запуска preprocess
        self.preprocess_button = Button(root, text="Обновить базу данных", command=self.run_preprocess)
        self.preprocess_button.pack(pady=10)

        # Кнопка для поиска с использованием CNN
        self.cnn_button = Button(root, text="Поиск с использованием CNN", command=self.run_cnn_search)
        self.cnn_button.pack(pady=10)

        # Метка для отображения выбранного изображения
        self.image_label = Label(root)
        self.image_label.pack(pady=10)

    def select_image(self):
        """Открывает диалог выбора изображения."""
        file_path = filedialog.askopenfilename(initialdir=self.input_folder, title="Выберите изображение",
                                              filetypes=(("JPEG files", "*.jpg"), ("All files", "*.*")))
        if file_path:
            top_matches = find_top_similar(file_path, self.descriptors_file, top_n=5)
            input_filename = os.path.basename(file_path)  # Получаем имя файла
            # Вычисляем точность
            accuracy, correct_matches = calculate_accuracy(input_filename, top_matches)
            messagebox.showinfo("Точность", f"Точность: {accuracy:.2f}% ({correct_matches} из 5)")
            display_results(file_path, top_matches, self.output_folder)

    def run_preprocess(self):
        """Запускает функцию preprocess."""
        preprocess_images(self.preprocess_folder, self.output_folder, self.descriptors_file)
        preprocess_images_cnn(self.preprocess_folder, self.output_folder, self.cnn_descriptors_file)  # Обновляем CNN базу
        messagebox.showinfo("Готово", "База данных обновлена!")

    def run_cnn_search(self):
        """Запускает поиск с использованием CNN."""
        file_path = filedialog.askopenfilename(initialdir=self.input_folder, title="Выберите изображение",
                                              filetypes=(("JPEG files", "*.jpg"), ("All files", "*.*")))
        if file_path:
            top_matches = find_top_similar_cnn(file_path, self.cnn_descriptors_file, top_n=5)
            input_filename = os.path.basename(file_path)  # Получаем имя файла
            # Вычисляем точность
            accuracy, correct_matches = calculate_accuracy(input_filename, top_matches)
            messagebox.showinfo("Точность (CNN)", f"Точность: {accuracy:.2f}% ({correct_matches} из 5)")
            display_results_cnn(file_path, top_matches, self.output_folder)

# Запуск приложения
if __name__ == "__main__":
    root = Tk()
    app = ImageSearchApp(root)
    root.mainloop()