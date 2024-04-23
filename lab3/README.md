Это демонстрация простейшей ml-модели классификации для развертывания через Docker.

Модель обучена для классификации ирисов на основе параметров "sepal_length", "sepal_width", "petal_length", "petal_width".
  
<br>
Описание файлов:

* src/train_model.py - файл тренировки модели
* src/app.py - натренированная модель с доступом через FastAPI
* data/datasets/iris_dataset.csv - датасет ирисов
* data/model/iris_model.pkl - сохраненная в pickle модель
* requirements.txt - файл с зависимостями
* Dockersfile - файл Docker
* docker-compose.yaml - файл docker-compose


<br>
Эндпоинты:

1. `GET /` - информационный эндпоинт с примером использования
2. `POST /predict` - эндпоинт для предсказания класса ириса

<br>
Для запуска:

1. Запустить сборку docker-compose командой `docker-compose up`
2. Отправить get-запрос на `http://127.0.0.1:8000` 