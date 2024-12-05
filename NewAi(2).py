import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords

# Скачиваем необходимые стоп-слова
nltk.download('stopwords')

# Загрузка данных из CSV
data = pd.read_csv('dataset.csv')

# Проверка на пустые значения
print(data.isnull().sum())

# Предобработка текста
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)  # Преобразуем число в строку
    text = re.sub(r'[^\w\s]', '', text.lower())  # Удаляем пунктуацию, приводим в нижний регистр
    stop_words = set(stopwords.words("russian"))
    words = [word for word in text.split() if word not in stop_words]  # Убираем стоп-слова
    return " ".join(words)

# Применяем предобработку ко всем отзывам
data['processed_text'] = data['review_text'].apply(preprocess_text)

# Разделим данные на обучающую и тестовую выборки для классификации тональности
X = data['processed_text']
y_sentiment = data['sentiment']  # Тональность отзыва (0 - негативный, 1 - позитивный)

# Преобразуем текст в числовые признаки с помощью TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)  # Обучаем векторизатор

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_sentiment, test_size=0.2, random_state=42)

# Используем класс с балансировкой классов
sentiment_model = LogisticRegression(max_iter=1000, class_weight='balanced')

# Обучаем модель для классификации тональности
sentiment_model.fit(X_train, y_train)

# Оценка на тестовой выборке
y_pred_sentiment = sentiment_model.predict(X_test)
print("Оценка модели для тональности:")
print(classification_report(y_test, y_pred_sentiment))

# Функция для предсказания тональности и проблем в отзыве
def predict_review_info(review_text):
    # Предобработка текста
    processed_text = preprocess_text(review_text)

    # Преобразуем текст в вектор, используя уже обученный векторизатор
    review_vectorized = vectorizer.transform([processed_text])  # Используем .transform(), так как векторизатор уже обучен

    # Прогнозируем тональность отзыва
    sentiment_pred = sentiment_model.predict(review_vectorized)[0]

    # Прогнозируем наличие проблемы (если у вас есть модель для обнаружения проблем)
    # В данный момент обучим модель проблем аналогично модели тональности
    issues_model = LogisticRegression(max_iter=1000, class_weight='balanced')  # Пример модели для проблем, тренируйте её аналогично
    issues_model.fit(X_train, y_train)  # Обучаем модель на тех же данных для примера
    issue_pred = issues_model.predict(review_vectorized)[0]

    # Преобразуем метки в текстовые значения
    sentiment_label = 'Positive' if sentiment_pred == 1 else 'Negative'
    issue_label = 'Problem' if issue_pred == 1 else 'No Problem'

    # Возвращаем результат
    return sentiment_label, review_text, issue_label

# Пример нового отзыва
new_review = "еда вкусная, но везли долго, комплект полный, всё теплое, свежее"

# Получаем результаты
sentiment, review, issue = predict_review_info(new_review)

# Сохраняем результат в CSV
result_df = pd.DataFrame({
    'Review Type': [sentiment],
    'Review Text': [review],
    'Issue Type': [issue]
})

# Сохраняем в файл
result_df.to_csv('predicted_review.csv', index=False)

print("Результат сохранён в файл 'predicted_review.csv'")
