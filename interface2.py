import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import os

class ChibbisCommentApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Chibbis - Ваша оценка важна!")
        
        # Фиксированный размер окна (800x800)
        self.root.geometry("900x280")
        
        # Блокировка изменения размера окна
        self.root.resizable(False, False)
        
        # Заполнение цветом заднего фона
        self.root.configure(bg="#4682B4")  # Синий цвет фона

        main_frame = ttk.Frame(self.root, padding="20", style="Main.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        upload_frame = ttk.Frame(main_frame, style="Upload.TFrame")
        upload_frame.pack(fill=tk.X, padx=20, pady=(30, 15))

        ttk.Label(upload_frame, text="Выберите файл для дальнейшей работы:",
                  font=("Arial", 18, "bold"), foreground="#FFFFFF", background="#4682B4").pack(fill=tk.X, pady=(0, 10))

        self.file_path = tk.StringVar()
        file_entry = ttk.Entry(upload_frame, textvariable=self.file_path, width=40, font=("Arial", 12),
                                state="readonly")
        file_entry.pack(fill=tk.X, expand=True, padx=10)

        upload_button = ttk.Button(upload_frame, text="Выбрать файл", command=self.browse_file,
                                   style="Upload.TButton")
        upload_button.pack(fill=tk.X, pady=(10, 0))

        button_frame = ttk.Frame(main_frame, style="Button.TFrame")
        button_frame.pack(fill=tk.X, padx=20, pady=(15, 30))

        publish_button = ttk.Button(button_frame, text="Отправить файл", command=self.publish_file,
                                    style="Publish.TButton")
        publish_button.pack(fill=tk.X, expand=True)

        self.result_frame = ttk.Frame(self.root, padding="20", style="Result.TFrame")
        self.result_frame.pack(fill=tk.BOTH, expand=True)

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select File",
            filetypes=[("All Files", "*.*"), ("Text files", "*.txt *.docx")],
            initialdir=os.getcwd()
        )
        self.file_path.set(path)

    def publish_file(self):
        file_path = self.file_path.get()

        if not file_path.strip():
            messagebox.showwarning("Внимание!", "Пожалуйста, выберите файл.")
            return

        result_text = f"Файл успешно выбран:\n{file_path}"
        self.show_result(result_text)

    def show_result(self, text):
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        result_label = ttk.Label(self.result_frame, text=text, wraplength=400,
                                 font=("Arial", 12), background="#FFFFFF")
        result_label.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure("Main.TFrame", background="#4682B4")
    style.configure("Upload.TFrame", background="#4682B4", borderwidth=1, relief="flat")
    style.configure("Button.TFrame", background="#4682B4")
    style.configure("Publish.TButton", background="#FF8000", foreground="#FFFFFF", padding="10 5", font=("Arial", 14, "bold"))
    style.map("Publish.TButton", background=[("active", "#FFA500"), ("disabled", "#D0D0D0")])
    style.configure("Result.TFrame", background="#FFFFFF")

    style.configure("Upload.TButton", background="#FFFFFF", foreground="#000000", padding="10 5", font=("Arial", 14, "bold"))
    style.map("Upload.TButton", background=[("active", "#4682B4"), ("disabled", "#D0D0D0")])

    app = ChibbisCommentApp(root)
    root.mainloop()
