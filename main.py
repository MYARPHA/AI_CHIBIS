import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import nltk
from nltk.corpus import stopwords
import csv

nltk.download('stopwords', quiet=True)  #Download stopwords quietly
stop_words = set(stopwords.words('russian'))

def preprocess_text(text):
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

#This function isn't used in the final output, so it can be removed for simplicity.
def extract_problems(text):
    problems = ['переварена', 'холодные']
    found_problems = [problem for problem in problems if problem in text]
    return found_problems


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model_path = './Trained_Model/best_model_20241206_1521.pt' #Update with your model path

try:
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    messagebox.showerror("Error", f"Model not found at {model_path}. Please ensure the model file exists.")
    exit(1)
except Exception as e:
    messagebox.showerror("Error", f"An error occurred while loading the model: {e}")
    exit(1)


def classify_review(text):
    processed_text = preprocess_text(text)
    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return pred


def process_csv_and_save(file_path, output_file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            writer.writerow(["Review Type", "Review Text"])
            next(reader) #Skip header if exists

            for row in reader:
                review_text = row[0]
                category_pred = classify_review(review_text)
                review_type = 0 if category_pred == 0 else 1 # Adjust mapping if needed
                writer.writerow([review_type, '"' + review_text + '"'])
        return True
    except FileNotFoundError:
        messagebox.showerror("Error", f"File not found: {file_path}")
        return False
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        return False


class ChibbisCommentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chibbis - Ваша оценка важна!")
        self.root.geometry("900x280")
        self.root.resizable(False, False)
        self.root.configure(bg="#4682B4")
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
            filetypes=[("CSV files", "*.csv")],
            initialdir=os.getcwd()
        )
        self.file_path.set(path)

    def publish_file(self):
        file_path = self.file_path.get()
        if not file_path.strip():
            messagebox.showwarning("Внимание!", "Пожалуйста, выберите файл.")
            return

        if process_csv_and_save(file_path, "data/output.csv"):
            self.show_result("Файл обработан и сохранен в output.csv")

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