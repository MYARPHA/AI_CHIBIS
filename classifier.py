import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import os
import glob
from datetime import datetime


# Функция для подготовки данных
def prepare_data(file_path):
    data = pd.read_csv(file_path, sep=";")
    data = data.dropna(subset=["Тип отзыва", "Отзыв"])  # Убираем строки с пустыми значениями

    # Делим данные на обучающую и валидационную выборки
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data["Отзыв"].tolist(),
        data["Тип отзыва"].tolist(),
        test_size=0.2,
        random_state=42
    )

    return train_texts, val_texts, train_labels, val_labels


# Класс Dataset для PyTorch
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# Функция для токенизации текста
def tokenize_data(tokenizer, texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    dataset = ReviewDataset(encodings, labels)
    return dataset


# Функция для выбора последней сохранённой модели
def get_latest_model_path(models_dir="./trained_model"):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        return None

    model_dirs = glob.glob(f"{models_dir}/best_model_*")
    if not model_dirs:
        return None

    latest_model = max(model_dirs, key=os.path.getctime)
    return latest_model


# Основной процесс обучения
def train_model(train_file, models_dir="./trained_model", base_model_path=None):
    """Обучение или дообучение модели."""
    # Чтение данных
    train_texts, val_texts, train_labels, val_labels = prepare_data(train_file)

    # Загрузка токенизатора
    tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

    # Если base_model_path указан, загружаем модель из него для дообучения
    if base_model_path:
        print(f"Загрузка модели для дообучения из {base_model_path}")
        model = BertForSequenceClassification.from_pretrained(base_model_path)
    else:
        print("Создание новой модели")
        model = BertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=4)

    # Токенизация данных
    train_dataset = tokenize_data(tokenizer, train_texts, train_labels)
    val_dataset = tokenize_data(tokenizer, val_texts, val_labels)

    # Аргументы для обучения
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=2,
        load_best_model_at_end=True
    )

    # Определяем тренер
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Обучение модели
    trainer.train()

    # Сохраняем модель с текущей датой
    date_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(models_dir, f"best_model_{date_suffix}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Модель сохранена в {save_path}")


# Предсказание типа отзыва
def predict(input_file, model_path, output_file="output.csv"):
    # Загрузка модели и токенизатора
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    # Чтение данных
    data = pd.read_csv(input_file, sep=";")
    texts = data["Отзыв"].tolist()

    # Токенизация
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

    # Предсказание
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=1)

    # Преобразование предсказаний
    mapping = {0: "позитивный", 1: "негативный", 2: "спорный", 3: "спам"}
    data["Тип отзыва"] = predictions.numpy()
    data["Тип отзыва"] = data["Тип отзыва"].apply(lambda x: mapping[x])

    # Сохранение результатов
    data.to_csv(output_file, index=False)
    print(f"Результаты сохранены в {output_file}")


# Точка входа
if __name__ == "__main__":
    print("Введите 'train' для обучения новой модели, 'continue' для дообучения последней или 'predict' для предсказания:")
    user_choice = input().strip().lower()

    if user_choice == "train":
        train_model("datasets/classifier_dataset.csv")
    elif user_choice == "continue":
        latest_model = get_latest_model_path()
        if latest_model:
            print(f"Дообучение последней модели из {latest_model}")
            train_model("datasets/classifier_dataset.csv", base_model_path=latest_model)
        else:
            print("Не найдено сохранённых моделей. Начинаем обучение новой модели.")
            train_model("datasets/classifier_dataset.csv")
    elif user_choice == "predict":
        latest_model = get_latest_model_path()
        if latest_model:
            predict("./data/Разметка_отзывов.csv", latest_model, "./data/output.csv")
        else:
            print("Не найдено сохранённых моделей. Сначала обучите новую модель.")
    else:
        print("Неверный ввод. Введите 'train', 'continue' или 'predict'.")