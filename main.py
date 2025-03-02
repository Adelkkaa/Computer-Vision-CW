from tkinter import Tk, Button, Label, filedialog, messagebox
from PIL import Image, ImageTk
from utils.ORB_preprocess import preprocess_images
from utils.ORB_FTS import find_top_similar, display_results


# === ИНТЕРФЕЙС ===
class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Поиск похожих изображений")

        self.input_folder = "test-images"
        self.preprocess_folder = "e-commerce"
        self.output_folder = "preprocessed_images"
        self.descriptors_file = "descriptors.pkl"

        # Кнопка для выбора изображения
        self.select_button = Button(root, text="Выбрать изображение", command=self.select_image)
        self.select_button.pack(pady=10)

        # Кнопка для запуска preprocess
        self.preprocess_button = Button(root, text="Обновить базу данных", command=self.run_preprocess)
        self.preprocess_button.pack(pady=10)

        # Метка для отображения выбранного изображения
        self.image_label = Label(root)
        self.image_label.pack(pady=10)

    def select_image(self):
        """Открывает диалог выбора изображения."""
        file_path = filedialog.askopenfilename(initialdir=self.input_folder, title="Выберите изображение",
                                              filetypes=(("JPEG files", "*.jpg"), ("All files", "*.*")))
        if file_path:
            top_matches = find_top_similar(file_path, self.descriptors_file, top_n=5)
            display_results(file_path, top_matches, self.output_folder)

    def run_preprocess(self):
        """Запускает функцию preprocess."""
        preprocess_images(self.preprocess_folder, self.output_folder, self.descriptors_file)
        messagebox.showinfo("Готово", "База данных обновлена!")

# Запуск приложения
if __name__ == "__main__":
    root = Tk()
    app = ImageSearchApp(root)
    root.mainloop()