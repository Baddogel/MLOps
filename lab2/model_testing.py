import pickle
import pandas as pd
from sklearn.metrics import classification_report  # функция scikit-learn которая считает много метрик классификации


# Загружаем модель из файла
with open('model.pickle', 'rb') as file:
    model = pickle.load(file)


# Загружаем предобработанный тестовый датафрейм из csv
df_test = pd.read_csv('test/weather_data_test_preprocessed.csv')


# Делим датафрейм на признаки и целевую переменную
x, y = df_test.drop(columns=['is_rain']), df_test['is_rain']


# Выводим отчет о классификации тестовых данных
print(classification_report(y, model.predict(x)))

