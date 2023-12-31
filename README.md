# Friends-Analysis
## Information

Данный проект реализован в рамках проектной работы "Анализ социальной сети" Университета ИТМО. Для анализа использовалась социальная сеть ВКонтакте.

This project was implemented as part of the project work "Social Network Analysis" of ITMO University. The social network VKontakte was used for the analysis.
---

## Setup

Для начала работы необходимо создать файл `secret.py`, где требуется указать ваш логин и пароль от социальной сети в глоабльных переменных `login` и `password`.

To get started, you need to create a file `secret.py`, where you need to specify your login and password from the social network in the global variables `login` and `password`.


### Инициализация окружения

Выполните команду
```
make setup
```

Будет создано новое виртуальное окружение в папке `.venv`.
В него будут установлены пакеты, перечисленные в файле `pyproject.toml`.

Обратите внимание: если вы один раз выполнили `make setup`, при попытке повторного ее выполнения ничего не произойдет, 
поскольку единственная ее зависимость - директория `.venv` - уже существует.
Если вам по какой-то причине нужно пересобрать окружение с нуля, 
выполните сначала команду `make clean` - она удалит старое окружение.

### Установка/удаление пакетов

Для установки новых пакетов используйте команду `poetry add`, для удаления - `poetry remove`. 
Мы не рекомендуем вручную редактировать секцию с зависимостями в `pyproject.toml`.

# Линтеры, тесты и автоформатирование

### Автоформатирование

Командой `make format` можно запустить автоматическое форматирование вашего кода.

Ее выполнение приведет к запуску [isort](https://github.com/PyCQA/isort) - утилиты 
для сортировки импортов в нужном порядке, и [black](https://github.com/psf/black) - одного из самых популярных форматтеров для `Python`.


### Статическая проверка кода

Командой `make lint` вы запустите проверку линтерами - инструментами для статического анализа кода. 
Они помогают выявить ошибки в коде еще до его запуска, а также обнаруживают несоответствия стандарту 
[PEP8](https://peps.python.org/pep-0008). 
Среди линтеров есть те же `isort` и `black`, только в данном случае они уже ничего не исправляют, а просто проверяют, что код отформатирован правильно.

### Тесты

Командой `make test` вы запустите тесты при помощи утилиты [pytest](https://pytest.org/). 

