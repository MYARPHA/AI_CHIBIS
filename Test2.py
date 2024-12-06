import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import nltk
from nltk.corpus import stopwords

# Скачиваем стоп-слова
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))


# === Функция обработки текста ===
def preprocess_text(text):
    """Удаление стоп-слов и приведение текста к нижнему регистру."""
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


def extract_problems(text):
    """Извлечение проблем из текста на основе ключевых слов."""
    problems = ['переварена', 'холодные']
    found_problems = [problem for problem in problems if problem in text]
    return found_problems


# === Загрузка токенизатора ===
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# === Загрузка модели и весов ===
model_path = './Trained_Model/best_model_20241206_1458.pt'

# Проверяем, существует ли файл модели
if os.path.exists(model_path):
    # Создаем модель с тем же числом меток, что было в процессе обучения
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)

    # Загружаем веса модели из файла
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cuda')))  # Добавлено map_location для загрузки на cuda
    print(f"Модель успешно загружена из {model_path}")
else:
    raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")


# === Функция для классификации одного отзыва ===
def classify_review(text):
    """Классификация нового отзыва"""
    processed_text = preprocess_text(text)
    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()  # Получаем индекс класса с максимальной вероятностью
    detected_problems = extract_problems(processed_text)
    return pred, detected_problems  # Возвращаем числовую метку и извлеченные проблемы


# === Пример одного отзыва ===
review_text = "Вкусно и быстро 10:)2839;₽:₽3&amp;(9;&amp;:);7;8;₽;!:!::8:&amp;&amp;4!4!4"

# Классификация отзыва
category_pred, problem_pred = classify_review(review_text)

# Вывод результатов
print(f"Тип категории предсказание: {category_pred}")
print(f"Типы проблем предсказание: {problem_pred}")