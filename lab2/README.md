Это демонстрация простейшего ml-пайплайна для развертывания в Jenkins.

Предположим, что по искусственно сгенерированным параметрам 'Температура', 'Атмосферное давление', 'Геомагнитная активность' и 'Скорость ветра' мы сможем предсказывать, пойдет ли дождь.
  
<br>
Описание файлов:

* data_creation.py - генерирует набор данных
* model_preprocessing.py - предобработка данных
* model_preparation.py - обучение модели на тренировочной выборке
* model_testing.py - проверка модели на тестовой выборке
* requirements.txt - файл с зависимостями
* Jenkinsfile - файл пайплайна для Jenkins

<br>
Для запуска:

1. Создать pipeline в Jenkins.
2. Скопировать ссылку на этот репозиторий в pipeline.
3. Указать в pipeline путь до Jenkinsfile.
4. Запустить сборку.