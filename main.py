import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import glob
import ctypes

# Настройка DPI для четкого отображения интерфейса
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass


# Функция для выбора последней модели
def get_latest_model_path(models_dir="./trained_model"):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        return None

    model_dirs = glob.glob(f"{models_dir}/best_model_*")
    if not model_dirs:
        return None

    latest_model = max(model_dirs, key=os.path.getctime)
    return latest_model


# Функция предсказания типа отзыва
def predict(input_file, model_path, output_file="./data/output.csv"):
    """Предсказание типа отзыва."""
    try:
        # Загрузка модели и токенизатора
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)

        # Чтение данных
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Файл {input_file} не найден.")
        data = pd.read_csv(input_file, sep=";")
        if "Отзыв" not in data.columns:
            raise KeyError("В файле отсутствует необходимый столбец 'Отзыв'.")

        texts = data["Отзыв"].tolist()

        # Токенизация
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)

        # Предсказание
        with torch.no_grad():
            outputs = model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=1)

        # Преобразование предсказаний
        mapping = {0: "позитивный", 1: "негативный", 2: "спорный", 3: "спам"}
        data["Тип отзыва"] = predictions.cpu().numpy()
        data["Тип отзыва"] = data["Тип отзыва"].apply(lambda x: mapping[x])

        # Сохранение результатов с разделителем ";"
        os.makedirs("./data", exist_ok=True)
        data.to_csv(output_file, sep=";", index=False)
        return output_file

    except Exception as e:
        print(f"Ошибка обработки: {e}")
        raise


# Интерфейс приложения
class ChibbisCommentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chibbis - Ваша оценка важна!")
        self.root.geometry("900x400")
        self.root.resizable(False, False)
        self.root.configure(bg="#4682B4")
        self.root.tk.call("tk", "scaling", 1.25)

        main_frame = ttk.Frame(self.root, padding="20", style="Main.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Загрузка файла
        upload_frame = ttk.Frame(main_frame, style="Upload.TFrame")
        upload_frame.pack(fill=tk.X, padx=20, pady=(10, 15))
        ttk.Label(upload_frame, text="Выберите файл для обработки:", font=("Arial", 14, "bold"),
                  foreground="#FFFFFF", background="#4682B4").pack(anchor=tk.W)
        self.file_path = tk.StringVar()
        file_entry = ttk.Entry(upload_frame, textvariable=self.file_path, state="readonly", font=("Arial", 12),
                               width=60)
        file_entry.pack(fill=tk.X, padx=10, pady=(5, 0))
        upload_button = ttk.Button(upload_frame, text="Выбрать файл", command=self.browse_file, style="Upload.TButton")
        upload_button.pack(pady=(5, 0))

        # Кнопка обработки
        button_frame = ttk.Frame(main_frame, style="Button.TFrame")
        button_frame.pack(fill=tk.X, padx=20, pady=(15, 30))
        publish_button = ttk.Button(button_frame, text="Обработать файл", command=self.publish_file,
                                    style="Publish.TButton")
        publish_button.pack(fill=tk.X, expand=True)

        # Результат
        self.result_frame = ttk.Frame(self.root, padding="20", style="Result.TFrame")
        self.result_frame.pack(fill=tk.BOTH, expand=True)

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Выберите файл",
            filetypes=[("CSV файлы", "*.csv")],
            initialdir=os.getcwd()
        )
        self.file_path.set(path)

    def publish_file(self):
        file_path = self.file_path.get()
        model_path = get_latest_model_path()

        if not model_path:
            messagebox.showerror("Ошибка",
                                 "Не найдено сохранённых моделей. Пожалуйста, обучите модель перед использованием.")
            return

        if not file_path.strip():
            messagebox.showwarning("Внимание", "Выберите файл для обработки.")
            return

        try:
            # Вызываем predict
            output_file = predict(file_path, model_path, output_file="./data/output.csv")
            self.show_result(f"Обработка завершена. Результат сохранён в {output_file}.")

            # Открываем папку с результатом
            self.open_output_folder("data")
        except Exception as e:
            messagebox.showerror("Ошибка обработки", f"Произошла ошибка: {e}")

    def show_result(self, text):
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        result_label = ttk.Label(self.result_frame, text=text, wraplength=400, font=("Arial", 12),
                                 background="#FFFFFF")
        result_label.pack(fill=tk.BOTH, expand=True)

    def open_output_folder(self, folder_path):
        """Открыть папку с результатом в проводнике."""
        try:
            os.startfile(folder_path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть папку: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Main.TFrame", background="#4682B4")
    style.configure("Upload.TFrame", background="#4682B4", borderwidth=1, relief="flat")
    style.configure("Button.TFrame", background="#4682B4")
    style.configure("Publish.TButton", background="#FF8000", foreground="#FFFFFF", padding="10 5",
                    font=("Arial", 14, "bold"))
    style.map("Publish.TButton", background=[("active", "#FFA500"), ("disabled", "#D0D0D0")])
    style.configure("Result.TFrame", background="#FFFFFF")
    style.configure("Upload.TButton", background="#FFFFFF", foreground="#000000", padding="10 5",
                    font=("Arial", 14, "bold"))
    style.map("Upload.TButton", background=[("active", "#4682B4"), ("disabled", "#D0D0D0")])

    app = ChibbisCommentApp(root)
    root.mainloop()
