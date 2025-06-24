# Два режима разработки

Используются uv workspaces ([док.](https://docs.astral.sh/uv/concepts/projects/workspaces/))

Настроил два режима:

1. Разработка (текущий `pyproject.toml`)  
   • `workspace = true` → пакеты ставятся editable из каталога `packages/*`.

2. Prod-режим — новый файл `pyproject.prod.toml`  
   • В нём только блок `[tool.uv.sources]`, где каждый пакет берётся из своего репозитория по тегу:
 ```toml
 # Каждый пакет живёт в своём репозитории. `subdirectory = "."` если pyproject.toml в корне.
 bench-utils      = { git = "https://github.com/VLMHyperBenchTeam/bench-utils.git",      tag = "v0.1.2", subdirectory = "." }
 model-interface  = { git = "https://github.com/VLMHyperBenchTeam/model-interface.git",  tag = "v0.1.2", subdirectory = "." }
 model-qwen2-5-vl = { git = "https://github.com/VLMHyperBenchTeam/model-qwen2-5-vl.git", tag = "v0.1.2", subdirectory = "." }
 ```
   • При выпуске новой версии меняете `tag = ...` на актуальный.

Как пользоваться  
```bash
# dev-режим
uv lock
uv sync

# prod-режим без переименований (uv ≥ 0.7)
# Помещаем `pyproject.toml` c Git-источниками в отдельную папку, например `prod/`.
# Далее передаём **каталог** через `--project` (а не путь к файлу!)

# Предположим структура:
# ├─ pyproject.toml            # dev-конфиг (workspace)
# └─ prod/pyproject.toml       # prod-конфиг

uv lock --project prod   --frozen          # генерируем/проверяем uv.lock под Git-теги
uv sync --project prod   --frozen          # устанавливаем прод-зависимости

# Проверка без изменения lock-файла:
uv lock --project prod --check

# Поведение `--frozen`: uv завершится ошибкой, если `uv.lock` отсутствует
# или расходится с `pyproject.toml`.  [Док.](https://docs.astral.sh/uv/concepts/projects/sync/#automatic-lock-and-sync)
```

**Объяснение команд prod-режима:**
• `--frozen` — использует версии из существующего `uv.lock` без обновления зависимостей
• Переименование файлов позволяет uv использовать prod-конфигурацию как основную
• Без `--frozen` uv попытается обновить зависимости, что может нарушить воспроизводимость

**Важно:** Для prod-режима обязательно используйте `--frozen`, чтобы гарантировать точное воспроизведение окружения.

**Альтернативный способ (если поддерживается флаг --project):**
```bash
# prod-режим с флагом --project (может не работать в старых версиях)
uv lock --project pyproject.prod.toml   # создаёт/обновит uv.lock под git-теги
uv sync --frozen --project pyproject.prod.toml  # установит пакеты из GitHub
```

Итого:  
• «dev» — режим активной разработки и редактирования проекта и пакетов;  
• «prod» — режим создания сборки для конечного пользователя, с закреплением версий пакетов из тегов на GitHub (без PyPI).

## Добавить зависимость для пакета внутри проекта

Предположим, что в `bench_utils` нужна `numpy>=1.26`.

Открываем `packages/bench_utils/pyproject.toml` и просто добавляем строку в массив `dependencies`:

```toml
[project]
# ... существующие строки ...
dependencies = [
    "pandas>=2.3.0",
    "scikit-learn>=1.7.0",
    "scipy>=1.15.3",
    "model-interface",
    "numpy>=1.26"         # ← новая строка
]
# ... существующие строки ...
```

Сохраняем файл и выполняем:

```bash
# в корене проекта
uv lock 
uv sync
```

После этого:
• все, кто импортирует `bench_utils`, автоматически получат `numpy` как transit-dependency;  
• в корневых скриптах (если они НЕ делают `import numpy` напрямую) ничего менять не нужно!
## Добавить зависимость в сам проект

Если `numpy` требуется не только внутри `bench_utils`, а в скриптах самого проекта 
(например, в `check_classifiication.py` стоит `import numpy as np`), то её нужно внести и в корневой `pyproject.toml`.

1. Откройте `pyproject.toml` в корне и расширьте список зависимостей:

```toml
[project]
# ... существующие строки ...
dependencies = [
    "bench-utils",
    "model-interface",
    "model-qwen2-5-vl",
    "numpy>=1.26"        # ← добавили
]
# ... существующие строки ...
```

(если библиотека уже заявлена транзитивно через `bench-utils`, то технически она установится и без этой строки, но явное объявление избавит от «скрытых» зависимостей и упростит сопровождение).

2. Снова выполните:

```bash
uv lock
uv sync
```

## Работа с зависимостями

Быстрый чек-лист:
1.	Правьте нужный `pyproject.toml` → `[project].dependencies`.  
2.	Запустите `uv lock` (корень) → обновится `uv.lock`.  
3.	`uv sync` → библиотека реально устанавливается.  
4.	Добавьте/обновите импорты в коде, коммитните изменения.

Благодаря work-space-режиму uv «подхватит» изменения во всех подпакетах, а Docker-сборка (или CI) в «prod»-режиме возьмёт зафиксированные версии из `pyproject.prod.toml`.

## Через uv add

`uv add` добавляет зависимости и обновляет `uv.lock`.  
Как именно она сработает зависит от того, где вы её запустите и какие флаги дадите.

## Добавить пакет в «глобальные» зависимости всего workspace

   (корневой `pyproject.toml` → `[project].dependencies`):  
   ```bash
   uv add numpy>=1.26
   ```
   
   Запускать из корня репо. После этого:
   • `numpy` появится в корневом `dependencies = [...]`;  
   • `uv lock && uv sync` выполнятся автоматически, если не отключите `--no-sync`.

### Добавить dev-зависимость (только для разработки)

```bash
uv add ruff --group dev
```
uv положит зависимость в [PEP 735 `dependency-groups`](https://packaging.python.org/en/latest/specifications/dependency-groups/):

```toml
[dependency-groups]
dev = [
    "ruff>=0.4"
]
```

Секция `tool.uv.dev-dependencies` считается *legacy*; использовать её не нужно.  См. «Development dependencies» в оф. документации uv (<https://docs.astral.sh/uv/concepts/projects/dependencies/#development-dependencies>).

## Добавить зависимость в конкретный пакет-участник workspace

   Варианты (эквивалентны):  
   a) перейти в каталог пакета и вызвать:  
   ```bash
   cd packages/bench_utils
   uv add numpy>=1.26
   ```  
   
   б) оставаться в корне, но указать пакет:  
   ```bash
   uv add numpy>=1.26 --package bench-utils
   ```  
   Тогда изменится `packages/bench_utils/pyproject.toml`, а корневой не изменится.

4. Особые случаи  
   • `--optional extra_name` — положит зависимость в `[project.optional-dependencies]`.  
   • `--index`, `--path`, `--git` — сразу заполнят `[tool.uv.sources]`.  
   • `--bounds major|minor|exact` — управляет формой версионного ограничения.

После любой операции можете отключить автоматическую синхронизацию флагами  
`--frozen` (не менять `uv.lock`) или `--no-sync` (не трогать виртуалку).

**Итого:** `uv add` полностью покрывает сценарии ручного редактирования `pyproject.toml`; просто выбирайте каталог/флаги, чтобы направить изменение либо в корень, либо в конкретный пакет.

## Запуск скриптов

В uv workspace есть два основных варианта.

### 1. «Правильный» способ — uv run

Команда `uv run` сама:
• проверит, что `uv.lock` совпадает с `pyproject.toml`;  
• при необходимости создаст/обновит `.venv`;  
• запустит скрипт внутри этой виртуалки.

Примеры:

1. Запустить python-скрипт:
```
uv run python check_classifiication.py
```

2. Если в `check_classifiication.py` есть шебанг (`#!/usr/bin/env python`):
```
uv run ./check_classifiication.py
```

3. Передать аргументы:
```
uv run python check_page_sorting.py --config myconf.json
```

4. Запустить любой shell-скрипт, которому нужны зависимости проекта:
```
uv run bash scripts/do_something.sh
```

5. Одноразово «подмешать» пакет другой версии:
```
uv run --with "torch==2.3.0" python my_bench.py
```

### 2. Классический способ — активировать .venv

Если нужно «жить» в окружении долго:

```
# сначала убедиться, что всё установлено
uv sync            # или uv lock && uv sync

# активируем виртуалку
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell

python check_classifiication.py
```

Но помните: при ручном запуске ответственность за актуальность `.venv` лежит на вас; после изменений зависимостей придётся снова делать `uv lock`/`uv sync`.

Кратко
• Для единичных вызовов предпочитайте `uv run …` — это безопасно и не требует активации окружения.  
• Для интерактивной сессии активируйте `.venv` после `uv sync`.

# Добавление своих пакетов

Ниже приведён базовый алгоритм, позволяющий подключить к вашему uv-workspace новые «собственные» пакеты, хранящиеся в каталоге `packages/`.

## Выбираем пакет, который хотим добавить

добавить пакет» можно двумя способами:
### 1. Добавляем уже существующий пакет

Берем наш пакет, например, `hello_world`.   

Пример структуры:
```
packages/hello_world
├── hello_world
│   └── __init__.py
├── pyproject.toml
└── README.md
```
  
В `packages/hello_world/pyproject.toml` обязательно укажите:

```toml
[project]
name = "hello-world" 
# по стандартам python name пакета через "-",
# а импорт, через "_"
version = "0.1.0"
...

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

2. Пакеты участники uv work space.
В uv нет отдельной подкоманды вида `uv workspace add ….`

Принадлежность к workspace определяется только содержимым секции:
```toml
[tool.uv.workspace]

members = ["packages/*"]
```

Благодаря параметру  `members` uv «видит» все подпапки `packages/*`.
Поэтому пути добавлять не нужно — любой новый под-каталог автоматически попадёт в workspace.

3. Добавьте запись в `[tool.uv.sources]` корневого `pyproject.toml`:

```toml
[tool.uv.sources]
bench-utils      = { workspace = true }
model-interface  = { workspace = true }
model-qwen2-5-vl = { workspace = true }
prompt-handler   = { workspace = true }
print-utils      = { workspace = true }
hello-world      = { workspace = true }   # ← новый пакет
```

Название слева (`hello-world`) должно совпадать с полем `name` в pyproject самого пакета.  
Если нужно несколько пакетов — просто добавляем строки.

4. При необходимости добавьте пакет в раздел `dependencies` там, где он действительно используется (в корне или в других ваших внутренних пакетах):

```toml
dependencies = [
   ...
   "my-cool-lib",
]
```

5. Обновите окружение:
```bash
uv sync --all-packages
```

**Что делает `--all-packages`:**
• Устанавливает **все** пакеты workspace в единое окружение `.venv` в корне проекта
• Создаёт editable install для всех workspace-пакетов (включая их зависимости)
• Позволяет импортировать любой пакет workspace из любого места проекта
• Применяет extras/groups ко всем пакетам workspace одновременно

**Альтернативы:**
```bash
# Установить только конкретный пакет и его зависимости
uv sync --package hello-world

# Обычная синхронизация (только корневые зависимости)
uv sync
```

Пример вывода:
```bash
Resolved 67 packages in 1ms
      Built hello-world @ file:///absolute_path/packages/hello_world
Prepared 1 package in 150ms
Installed 1 package in 0.31ms
 + hello-world==0.0.2 (from file:///absolute_path/packages/hello_world)
```

**Важно:** После `--all-packages` все пакеты workspace доступны для импорта из единого окружения, что удобно для разработки, но может скрыть проблемы с зависимостями между пакетами.

**Рекомендации по использованию:**
• Используйте `--all-packages` для интерактивной разработки и отладки
• В CI/CD пайплайнах предпочитайте `uv sync --package <specific-package>` для точного контроля зависимостей
• Для production-builds указывайте конкретные пакеты, чтобы избежать установки неиспользуемых зависимостей
• При работе с `uv run` в подпакетах dev-dependencies из корневого workspace могут быть недоступны (известное ограничение)

6. Проверка:
```bash
uv run python -c 'import hello_world; print(f"Модуль: {hello_world.__name__}"); print(f"Файл: {hello_world.__file__}"); print("Импорт успешен!")'
```

### 2. Пакета ещё нет, – создаём и сразу включаем в workspace

   ```bash
   uv init packages/my_cool_lib
   ```  
   • `uv init` сработает «внутри» workspace (распознаётся ближайший `pyproject.toml`),  
   • автоматически добавит новый каталог в список `members`, если он туда не попадает,  
   • сгенерирует `pyproject.toml` пакета.

Подробнее в документации ([ссылка](https://docs.astral.sh/uv/concepts/projects/workspaces/)).

## Добавление в pord-сборку

Если в дальнейшем вы захотите сделать prod-сборку, укажите для того же пакета источник-Git в `pyproject.prod.toml`, по аналогии с уже существующими зависимостями:

```toml
[tool.uv.sources]
hello-world = { git = "https://github.com/VLMHyperBenchTeam/hello_world", tag = "0.0.2", subdirectory = "." }
```

Таким образом, dev-среда использует локальный workspace, а prod-среда — выкачивает тегированную версию из GitHub.