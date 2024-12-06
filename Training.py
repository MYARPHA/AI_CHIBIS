from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3, cache_dir='./model')

from transformers import BertTokenizer

# Путь для сохранения файлов
model_dir = './model'

# Загружаем токенизатор с Hugging Face Hub
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Сохраняем токенизатор в локальную директорию
tokenizer.save_pretrained(model_dir)

print(f"Токенизатор успешно сохранен в {model_dir}")

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import nltk
from nltk.corpus import stopwords
from torch.utils.data import Dataset
import accelerate

# Скачиваем стоп-слова
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

# === Параметры ===
dataset_path = './dataset.csv'  # Путь к датасету
problems_path = './problems.csv'  # Путь к файлу с проблемами
result_file_prefix = './Results/result'  # Префикс для сохранения результатов
model_dir = './Trained_Model'  # Директория для сохранения моделей
cache_dir = './model_cache'  # Директория для кэша моделей
os.makedirs(model_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)
# === Загрузка "Типов проблем" ===
problems = pd.read_csv(problems_path, header=None).squeeze().tolist()
problems = [problem.lower() for problem in problems]  # Приводим к нижнему регистру

# === Функция обработки текста ===
def preprocess_text(text):
    """Удаление стоп-слов и приведение текста к нижнему регистру."""
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def extract_problems(text):
    """Извлечение проблем из текста на основе ключевых слов."""
    found_problems = [problem for problem in problems if problem in text]
    return found_problems

# === Загрузка и обработка данных ===
data = pd.read_csv(dataset_path, sep=';')

# Добавляем обработанный текст в отдельный столбец
data['processed_text'] = data['Отзыв'].apply(preprocess_text)

# Выделяем типы проблем для каждого отзыва
data['Типы проблем'] = data['processed_text'].apply(extract_problems)

# Разделяем данные на тренировочную и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['processed_text'], data['Тип отзыва'], test_size=0.2, random_state=42
)

# === Датасет и токенизация ===
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(
            text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# === Инициализация токенайзера ===
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir=cache_dir)  # Загружаем токенизатор с кэшем

# Создаем DataLoader
train_dataset = ReviewDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
test_dataset = ReviewDataset(test_texts.tolist(), test_labels.tolist(), tokenizer)

# === Проверяем наличие последней сохраненной модели ===
model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pt')])
if model_files:
    # Загружаем последнюю модель из кеша
    latest_model = os.path.join(model_dir, model_files[-1])
    print(f"Загрузка модели: {latest_model}")

    # Создаем модель с указанием кэша
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3, cache_dir=cache_dir)

    model.load_state_dict(torch.load(latest_model, map_location=torch.device('cuda')))
    model.eval()  # Переводим модель в режим оценки
else:
    # Создаем новую модель
    print("Создание новой модели.")
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3, cache_dir=cache_dir)

# === Параметры обучения ===
date = datetime.now().strftime('%Y%m%d_%H%M')
training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=15,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join('logs'),
    logging_steps=6,
    load_best_model_at_end=True,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# === Обучение модели ===
trainer.train()

# Сохранение лучших весов с датой
best_model_path = os.path.join(model_dir, f'best_model_{date}.pt')
torch.save(model.state_dict(), best_model_path)
print(f"Сохранена лучшая модель: {best_model_path}")

# === Оценка модели ===
predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)