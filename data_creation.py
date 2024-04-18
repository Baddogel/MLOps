import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split#  функция разбиения на тренировочную и тестовую выборку
# в исполнении scikit-learn

DF_SIZE = 1000  # Количество строк в датафрейме


# Генерируем чистые значения температуры, атмосферного давления, геомагнитной активности и
# скорости ветра на основе нормального распределения
temperature = np.random.normal(loc=20.0, scale=3.0, size=DF_SIZE)
atmospheric_pressure = np.random.normal(loc=755.0, scale=5.0, size=DF_SIZE)
geomagnetic_activity = np.random.normal(loc=3, scale=1.0, size=DF_SIZE)
wind_speed = np.random.normal(loc=7, scale=2.0, size=DF_SIZE)


# Задаем целевую переменную, идет ли дождь (в нашем случае вероятность этого будет 20%)
is_rain = []
for i in range(DF_SIZE):
    probability_value = np.random.uniform(low=0.0, high=10.0, size=None)
    if probability_value < 2.0:
        is_rain.append(1)
    else:
        is_rain.append(0)


def add_noise(array, probability=0.1):
    '''
    Функция добавляет шум в массив.
    :param array: Изменяемый массив np.array
    :param probability: Вероятность добавления шумового значения вместо
    элемента массива, рекомендуемые значения от 0.1 до 1.0
    :return: Зашумленный массив np.array
    '''
    with np.nditer(array, op_flags=['readwrite']) as it:
        for x in it:
            probability_value = np.random.uniform(low=0.0, high=10.0, size=None)
            if probability_value < (probability * 10):
                x[...] = x * probability_value
    return array


# Добавляем шум к нашим параметрам
temperature_noised = add_noise(np.copy(temperature))
atmospheric_pressure_noised = add_noise(np.copy(atmospheric_pressure))
geomagnetic_activity_noised = add_noise(np.copy(geomagnetic_activity))
wind_speed_noised = add_noise(np.copy(wind_speed))


# Собираем зашумленные параметры воедино + целевая переменная и пакуем все это в датафрейм
data = {
    'temperature': temperature_noised,
    'atmospheric_pressure': atmospheric_pressure_noised,
    'geomagnetic_activity': geomagnetic_activity_noised,
    'wind_speed': wind_speed_noised,
    'is_rain': is_rain
}
df = pd.DataFrame(data)


# Создаем директории test и train
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)


# Разделяем наш датафрейм на тренировочную и тестовую выборки
df_train, df_test = train_test_split(df, test_size=0.3)


# Сохраняем датасеты по соответствующим папкам в формате csv
df_train.to_csv('train/weather_data_train_raw.csv', index=False)
df_test.to_csv('test/weather_data_test_raw.csv', index=False)
