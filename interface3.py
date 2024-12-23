import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import os

class ChibbisCommentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chibbis - Ваша оценка важна!")
        self.root.configure(bg="#4682B4")  # Синий цвет фона

        self.root.geometry("400x600")  # Фиксированный размер окна 400x600

        # Создание главного фрейма
        main_frame = ttk.Frame(self.root, padding="20", style="Main.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Создание фрейма для загрузки файла
        upload_frame = ttk.Frame(main_frame, style="Upload.TFrame")
        upload_frame.pack(fill=tk.X, padx=20, pady=(30, 15))

        ttk.Label(upload_frame, text="Загрузить файл:", 
                   font=("Arial", 12, "bold"), background="#4682B4").pack(side=tk.LEFT)

        self.file_path = tk.StringVar()
        file_entry = ttk.Entry(upload_frame, textvariable=self.file_path, width=40, font=("Arial", 12))
        file_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)

        upload_button = ttk.Button(upload_frame, text="Browse", command=self.browse_file,
                                   style="Upload.TButton")
        upload_button.pack(side=tk.RIGHT, padx=5)

      

        # Создание фрейма для кнопки публикации
        button_frame = ttk.Frame(main_frame, style="Button.TFrame")
        button_frame.pack(fill=tk.X, padx=20, pady=(15, 30))

        publish_button = ttk.Button(button_frame, text="Опубликовать", command=self.publish_comment,
                                  style="Publish.TButton")
        publish_button.pack(fill=tk.X, expand=True)

        # Создание фрейма для отображения результата
        self.result_frame = ttk.Frame(self.root, padding="20", style="Result.TFrame")
        self.result_frame.pack(fill=tk.BOTH, expand=True)

    def browse_file(self):
        path = filedialog.askopenfilename(title="Select File",
                                           filetypes=[("All Files", "*.*"), ("Text files", "*.txt *.docx")],
                                           initialdir=os.getcwd())
        self.file_path.set(path)

    def publish_comment(self):
        comment = self.comment_entry.get()
        file_path = self.file_path.get()

        if not comment.strip():
            messagebox.showwarning("Внимание!", "Пожалуйста, введите комментарий.")
            return

        result_text = f"Спасибо за ваш отзыв!\n\n"
        result_text += f"Файл: {file_path}\n"
        result_text += f"Комментарий: {comment}"
        
        self.show_result(result_text)

    def show_result(self, text):
        result_label = ttk.Label(self.result_frame, text=text, wraplength=400,
                                 font=("Arial", 12), background="#FFFFFF")
        result_label.pack(fill=tk.BOTH, expand=True)

        # Обновляем размер окна
        self.root.geometry("400x650")  # Обновляем размер окна с учетом новой высоты

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("clam")
    
    # Создание стилей для фреймов и кнопок
    style.configure("Main.TFrame", background="#4682B4")
    style.configure("Upload.TFrame", background="#FFFFFF", borderwidth=1, relief="ridge")
    style.configure("Comment.TFrame", background="#FFFFFF", borderwidth=1, relief="ridge")
    style.configure("Button.TFrame", background="#4682B4")
    style.configure("Publish.TButton", background="#FFFFFF", foreground="#4682B4", padding="10 5")
    style.map("Publish.TButton", background=[("active", "#4682B4"), ("disabled", "#D0D0D0")])
    style.configure("Result.TFrame", background="#FFFFFF")

    app = ChibbisCommentApp(root)
    root.mainloop()