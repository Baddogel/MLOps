import pandas as pd
from sklearn.linear_model import LogisticRegression  # Логистичекая регрессия от scikit-learn
import pickle  # Для сохранения модели в файл


# Загружаем предобработанный тренировочный датафрейм из csv
df_train = pd.read_csv('train/weather_data_train_preprocessed.csv')


# Делим датафрейм на признаки и целевую переменную
x, y = df_train.drop(columns = ['is_rain']), df_train['is_rain']


# Обучаем модель
model = LogisticRegression(random_state=42)
model.fit(x, y)


# Сохраняем модель при помощи pickle
with open('model.pickle', 'wb') as file:
    pickle.dump(model, file)
