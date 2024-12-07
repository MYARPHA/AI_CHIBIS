import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import glob
from datetime import datetime


# Функция для подготовки данных
def prepare_data(file_path):
    data = pd.read_csv(file_path, sep=";")
    data = data.dropna(subset=["Отзыв"])  # Убираем строки с пустыми отзывами

    # Разделяем данные на обучающую и валидационную выборки
    train_texts, val_texts = train_test_split(
        data["Отзыв"].tolist(),
        test_size=0.2,
        random_state=42
    )

    # Преобразование меток в числовые значения
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data["Типы проблем"].tolist())

    return train_texts, val_texts, labels, label_encoder


# Класс Dataset для PyTorch
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Убедитесь, что метки - целые числа
        return item


# Функция для токенизации текста
def tokenize_data(tokenizer, texts, labels=None):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    dataset = ReviewDataset(encodings, labels)
    return dataset


# Функция для выбора последней сохранённой модели
def get_latest_model_path(models_dir="./trained_model"):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        return None

    model_dirs = glob.glob(f"{models_dir}/best_model_detector_*")
    if not model_dirs:
        return None

    latest_model = max(model_dirs, key=os.path.getctime)
    return latest_model


# Основной процесс обучения
def train_model(train_file, models_dir="./trained_model", base_model_path=None):
    """Обучение или дообучение модели."""
    # Чтение данных
    train_texts, val_texts, labels, label_encoder = prepare_data(train_file)

    # Загрузка токенизатора
    tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

    # Если base_model_path указан, загружаем модель из него для дообучения
    if base_model_path:
        print(f"Загрузка модели для дообучения из {base_model_path}")
        model = BertForSequenceClassification.from_pretrained(base_model_path, num_labels=len(label_encoder.classes_))
    else:
        print("Создание новой модели")
        model = BertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=len(label_encoder.classes_))

    # Токенизация данных с метками
    train_dataset = tokenize_data(tokenizer, train_texts, labels)
    val_dataset = tokenize_data(tokenizer, val_texts, labels)

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
    save_path = os.path.join(models_dir, f"best_model_detector_{date_suffix}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Модель сохранена в {save_path}")


# Предсказание типа отзыва и классификация проблем и достоинств
def predict(input_file, model_path, output_file="output_detector.csv"):
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
    data["Типы проблем и достоинств"] = predictions.numpy()

    # Преобразуем типы проблем и достоинств в отдельные столбцы
    problems_and_advantages = data["Типы проблем и достоинств"].apply(lambda x: map_predictions_to_labels(x))

    # Сохранение результатов
    data[["Отзыв", "Типы проблем", "Типы достоинств"]] = problems_and_advantages
    data.to_csv(output_file, index=False)
    print(f"Результаты сохранены в {output_file}")


def map_predictions_to_labels(predictions):
    # Загружаем список проблем и достоинств из CSV файла
    file_path = "datasets/advantages_problems.csv"
    try:
        problems_and_advantages = pd.read_csv(file_path, sep=";")
    except FileNotFoundError:
        print(f"Ошибка: файл {file_path} не найден.")
        return pd.Series({"Типы проблем": [], "Типы достоинств": []})

    # Разделяем на проблемы и достоинства
    problems = problems_and_advantages["Проблема"].dropna().tolist()
    advantages = problems_and_advantages["Достоинство"].dropna().tolist()

    # Маппинг предсказаний на метки проблем и достоинств
    problems_labels = [label for label in problems if label in predictions]
    advantages_labels = [label for label in advantages if label in predictions]

    return pd.Series({"Типы проблем": problems_labels, "Типы достоинств": advantages_labels})


# Точка входа
if __name__ == "__main__":
    print(
        "Введите 'train' для обучения новой модели, 'continue' для дообучения последней или 'predict' для предсказания:")
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
            predict("./data/Разметка_отзывов.csv", latest_model, "./data/output_detector.csv")
        else:
            print("Не найдено сохранённых моделей. Сначала обучите новую модель.")
    else:
        print("Неверный ввод. Введите 'train', 'continue' или 'predict'.")
