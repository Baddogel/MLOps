import pandas as pd  # Библиотека Pandas для работы с табличными данными
from sklearn.preprocessing import StandardScaler  # Импортируем стандартизацию от scikit-learn
from sklearn.compose import ColumnTransformer  # т.н. преобразователь колонок
from sklearn.pipeline import Pipeline  # Pipeline. Не добавить, не убавить


# Загружаем датафреймы из csv
df_train = pd.read_csv('train/weather_data_train_raw.csv')
df_test = pd.read_csv('test/weather_data_test_raw.csv')


# Запоминаем названия колонок, это пригодится нам дальше
df_train_columns = df_train.columns
df_test_columns = df_test.columns


# Т.к. по всем признакам у нас нормальное распределение, достаточно будет стандартизировать переменные
pipe_temperature = Pipeline([
    ('scaler', StandardScaler())
])
col_temperature = ['temperature']

pipe_atmospheric_pressure = Pipeline([
    ('scaler', StandardScaler())
])
col_atmospheric_pressure = ['atmospheric_pressure']

pipe_geomagnetic_activity = Pipeline([
    ('scaler', StandardScaler())
])
col_geomagnetic_activity = ['geomagnetic_activity']

pipe_wind_speed = Pipeline([
    ('scaler', StandardScaler())
])
col_wind_speed = ['wind_speed']


# Создадим пайплайн для трансформации наших датафреймов
preprocessors_num = ColumnTransformer(transformers=[
    ('pipe_temperature', pipe_temperature, col_temperature),
    ('pipe_atmospheric_pressure', pipe_atmospheric_pressure, col_atmospheric_pressure),
    ('pipe_geomagnetic_activity', pipe_geomagnetic_activity, col_geomagnetic_activity),
    ('pipe_wind_speed', pipe_wind_speed, col_wind_speed),
])


# Трансформируем датафреймы
df_train_preprocessed = preprocessors_num.fit_transform(df_train)
df_test_preprocessed = preprocessors_num.fit_transform(df_test)


# Возвращаем полученным массивам обратно вид датафрейма
df_train_preprocessed = pd.DataFrame(df_train_preprocessed, columns=df_train_columns[:-1])
df_train_preprocessed = df_train_preprocessed.join(df_train['is_rain'])
df_test_preprocessed = pd.DataFrame(df_test_preprocessed, columns=df_test_columns[:-1])
df_test_preprocessed = df_test_preprocessed.join(df_test['is_rain'])


# Сохраняем предобработанные датафреймы по соответствующим папкам в формате csv
df_train_preprocessed.to_csv('train/weather_data_train_preprocessed.csv', index=False)
df_test_preprocessed.to_csv('test/weather_data_test_preprocessed.csv', index=False)
