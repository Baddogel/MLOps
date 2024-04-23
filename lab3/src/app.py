import pickle
from fastapi import FastAPI
from pydantic import BaseModel


# Определяем входные параметры модели
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


app = FastAPI()


# Загружаем модель из файла
with open('data/model/iris_model.pkl', 'rb') as file:
    model, scaler = pickle.load(file)


# Сопоставляем индексы с названиями классов
class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}


# Создаем эндпоинт для классификации ирисов
@app.post("/predict/")
async def predict(item: IrisInput):
    input_data = [[item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]]
    input_data_scaled = scaler.transform(input_data)
    prediction_index = model.predict(input_data_scaled)[0]
    prediction_class = class_names[prediction_index]
    return {"prediction": prediction_class}


# Создаем эндпоинт возвращающий информационное сообщение
@app.get("/")
async def get():
    return {
        "message": "For iris classification, send a POST request to the /predict endpoint.",
        "example_body": {
            "sepal_length": 1.6,
            "sepal_width": 4.4,
            "petal_length": 1.4,
            "petal_width": 3.6
        }
    }