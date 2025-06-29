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

### Пример: добавляем зависимости для `model_qwen2.5-vl`

Допустим, пакету понадобились `torch`, `transformers` и утилиты `qwen-vl-utils`.

```bash
# из корня репозитория
uv add torch>=2.3.0 transformers>=4.41.0 qwen-vl-utils>=0.0.10 \
       --package model-qwen2_5-vl

# или эквивалентно из каталога самого пакета
cd packages/model_qwen2.5-vl
uv add torch>=2.3.0 transformers>=4.41.0 qwen-vl-utils>=0.0.10
```

После команды `uv add`:
1. строки с новыми библиотеками будут добавлены в `packages/model_qwen2.5-vl/pyproject.toml`;
2. файл `uv.lock` обновится;
3. зависимости установятся в `.venv`.

> ⚡ Если хотите ускорить инференс, можно добавить *опциональную* зависимость FlashAttention-2:
>
> ```bash
> uv add flash-attn --package model-qwen2_5-vl --optional flash
> ```
>
> Она запишется в `[project.optional-dependencies.flash]`, а установка останется необязательной.

#### Установка optional-extra `flash`

Чтобы подтянуть опциональные зависимости (extras) в окружение, используйте ключ `--extra` (или `--all-extras`). Примеры:

```bash
# запустить скрипт, временно добавив extra «flash»
uv run --extra flash python run_vqa.py

# установить extra «flash» в .venv «на постой»
uv sync --extra flash

# установить сразу все optional-extras проекта
uv sync --all-extras
```

Команда `uv run` автоматически проверит lock-файл, при необходимости обновит его и
подмешает нужные пакеты в текущую сессию. `uv sync` — подходит для постоянного
добавления extra-зависимостей в виртуальное окружение проекта.

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

## Git-подмодули для пакетов в `./packages`

Все внутренние пакеты (например `bench_utils`, `model_interface`, `model_qwen2.5-vl`, `print_utils`, `prompt_handler`, `hello_world`) теперь подключены к репозиторию как **Git submodule**. Это влияет на процесс клонирования и обновления кода.

### Клонирование репозитория вместе с подмодулями

```bash
# Клонируем основной репозиторий
git clone <repo-url>
cd <repo>

# Инициализируем и выкачиваем содержимое всех подмодулей
git submodule update --init --recursive
```

Если пропустить последнюю команду, каталоги в `packages/*` останутся пустыми.

### Обновление подмодулей

Подтянуть последние изменения во **всех** подпакетах:

```bash
git submodule update --remote --merge  # или --rebase
```

Обновить **конкретный** пакет:

```bash
cd packages/model_interface
# переходим на нужную ветку/тег и тянем изменения
git checkout dev_branch            # или main / v0.2.1
git pull --ff-only
cd ../../

# фиксируем новую ревизию подмодуля в родительском репо
git add packages/model_interface
git commit -m "chore: bump model_interface submodule"
```

### Переключение подмодуля на конкретный тег/ветку

```bash
cd packages/model_interface
git checkout v0.2.0   # либо нужная ветка
cd ../../
git add packages/model_interface
git commit -m "chore: freeze model_interface@v0.2.0"
```

### Добавление нового пакета-подмодуля

```bash
git submodule add https://github.com/VLMHyperBenchTeam/new_pkg.git packages/new_pkg
git commit -m "feat: add new_pkg as submodule"
```

### Удаление подмодуля

```bash
git submodule deinit -f packages/old_pkg
git rm -f packages/old_pkg
rm -rf .git/modules/packages/old_pkg

git commit -m "chore: remove old_pkg submodule"
```

### Почему мы используем submodule

• Позволяют версионировать пакеты независимо и подтягивать нужные теги/ветки;
• Исключают дублирование кода (один источник истины для каждого пакета);
• Сохраняют полную историю Git внутри каждого подпакета;
• При этом workflow uv-workspace (editable install) остаётся тем же — достаточно работать внутри подмодуля, и изменения сразу доступны основному проекту.

#\#\# Проверьте lock-файл и окружение

Перед коммитом удобно убедиться, что `uv.lock` свежий, а `.venv` совпадает с ним.

```bash
uv lock --locked      # завершится ошибкой, если lock устарел
uv sync --check       # проверит .venv без установки
```

# prod-сборку
```bash
# prod-режим (каталог prod/ содержит свой pyproject.toml)
uv lock --project prod          # создаём/обновляем prod/uv.lock
uv sync --project prod --frozen # устанавливаем по lock-файлу
```

### Docker: частичные установки

Для слоёв Docker-сборки полезны флаги `uv sync`:

* `--no-install-project`  – не ставить корневой пакет;
* `--no-install-workspace` – не ставить ни один пакет workspace;
* `--no-install-package <pkg>` – исключить конкретный пакет.

Пример минимального слоя с зависимостями **без** самого приложения:

```Dockerfile
COPY uv.lock pyproject.toml ./
RUN uv sync --no-install-project --all-packages --frozen \
    && rm -rf ~/.cache/uv
```

Позже, когда копируете исходники приложения, выполняете обычный `uv sync --frozen` – слой с базовыми зависимостями кэшируется.

## Release Tool как dev-зависимость

`release_tool` подключён как пакет-участник workspace (`members = [ ..., "release_tool" ]`),
но находится в группе зависимостей `dev`.

Это даёт два эффекта:

1. При обычной разработке (`uv lock` / `uv sync`) пакет попадает в окружение
   и его CLI-entrypoints (`release-tool-stage4`, `release-tool-clear`, …) доступны.
2. В прод-сборках вы вызываете `uv sync --no-dev` **или** используете
   `prod/pyproject.toml`, где `release_tool` не перечислен → пакет **не** устанавливается,
   лишних зависимостей и скриптов нет.

Если в CI нужно запустить релизные скрипты отдельно, добавьте флаг `--group dev` или
`--all-groups` при установке окружения:
```bash
uv sync --group dev
```

## Git-хуки и `pre-commit`

В репозитории настроены *Git hooks* через [pre-commit](https://pre-commit.com).  При обычном `git commit` хук запускает выбранные проверки кода **до** сохранения коммита.

### Установка/обновление хуков

```bash
# установить dev-зависимости (ruff, pre-commit, …)
uv sync --group dev

# зарегистрировать git-хуки (создаст .git/hooks/pre-commit)
uv run pre-commit install

# регулярно обновляйте версии линтеров
uv run pre-commit autoupdate && git add .pre-commit-config.yaml
```

### Что именно проверяется

Файл `.pre-commit-config.yaml` подключает два набора правил:

| Хук | Repo | Назначение |
|-----|------|------------|
| `ruff` | `charliermarsh/ruff-pre-commit` | статический анализ Python (PEP8, PyFlakes, isort и др.) |
| `trailing-whitespace` | `pre-commit/pre-commit-hooks` | удаляет лишние пробелы/табуляцию в конце строк |

При необходимости добавляйте свои плагины в `.pre-commit-config.yaml` и запускайте `pre-commit autoupdate`.

### Когда срабатывают

* автоматически при `git commit` (этап *pre-commit*);
* можно запустить вручную для отдельных файлов или всего репозитория:
  ```bash
  uv run pre-commit run --all-files       # проверить все
  uv run pre-commit run ruff --files path/to/file.py
  ```

### Если проверки не пройдены

1. pre-commit выведет список ошибок и прервёт коммит.
2. Исправьте файл(ы) вручную **или** примите авто-фиксы (ruff и trailing-whitespace умеют править код сами).
3. Повторите `git add …` и `git commit`.
4. В крайнем случае временно пропустите проверки:
   ```bash
   git commit --no-verify -m "chore: hot-fix"
   ```
   (используйте только при острой необходимости, чтобы не ломать стиль кода).

## Переключение backend-а PyTorch (CUDA 12.4 / 12.8)

В рабочем пространстве предусмотрены **extras** для разных вариантов бэкэнда PyTorch:

| Extra | Описание | Индекс PyTorch |
|-------|----------|----------------|
| `cu124` | CUDA 12.4 + cuDNN | https://download.pytorch.org/whl/cu124 |
| `cu128` | CUDA 12.8 + cuDNN | https://download.pytorch.org/whl/cu128 |

При необходимости легко добавить новые (`cu118`, `cu128`, …)

### Локальная установка (dev-режим)

```bash
# CUDA 12.4
uv sync --extra cu124

# CUDA 12.8
uv sync --extra cu128

# убедиться, что lock не изменился, но окружение актуально
uv sync --check
```

Для генерации/обновления lock-файла используйте те же extra:

```bash
uv lock --extra cu124     # пересчитать под CUDA 12.4
uv lock --extra cu128     # пересчитать под CUDA 12.8
```

### Prod-режим

В каталоге `prod/` лежит собственный `pyproject.toml`. Логика такая же, только добавляйте флаги `--project prod` и `--frozen`:

```bash
# prod, CUDA 12.4
uv lock   --project prod --extra cu124      # если нужно обновить lock
uv sync   --project prod --extra cu124 --frozen

# prod, CUDA 12.8
uv lock   --project prod --extra cu128      # если нужно обновить lock
uv sync   --project prod --extra cu128 --frozen
```

### Docker: dev и prod образы под разные backend-варианты

`docker/Dockerfile-cu124-uv` умеет переключаться между CUDA-вариантами с помощью build-arg `TORCH_BACKEND`.
Примеры команд:

```bash
# dev-образ: CUDA 12.4
docker build -f docker/Dockerfile-cu124-uv \
  --target dev \
  --build-arg TORCH_BACKEND=cu124 \
  -t project:dev-cu124 .

# dev-образ: CUDA 12.8
docker build -f docker/Dockerfile-cu124-uv \
  --target dev \
  --build-arg TORCH_BACKEND=cu128 \
  -t project:dev-cu128 .

# prod-образ: CUDA 12.4
docker build -f docker/Dockerfile-cu124-uv \
  --target prod \
  --build-arg TORCH_BACKEND=cu124 \
  -t project:prod-cu124 .

# prod-образ: CUDA 12.8
docker build -f docker/Dockerfile-cu124-uv \
  --target prod \
  --build-arg TORCH_BACKEND=cu128 \
  -t project:prod-cu128 .
```

> За подробностями обратитесь к документу [`docker/Docker_builder.md`](../docker/Docker_builder.md).

## Выбор целевой версии Python (3.10 ↔ 3.12)

В `pyproject.toml` задан широкий диапазон `requires-python = ">=3.10, !=3.11.*, <3.13"`,
поэтому `uv lock` по-умолчанию пытается найти **универсальное** решение, подходящее
сразу для 3.10 и 3.12.  Если нужно сгенерировать или установить окружение строго под
конкретный интерпретатор, используйте флаг `--python` (не `--python-version`) — он
поддерживается во всех основных командах (`uv lock`, `uv sync`, `uv run`, …):

```bash
# lock-файл только для CPython 3.10
uv lock --python 3.10 --extra cu124

# lock-файл только для CPython 3.12
uv lock --python 3.12 --extra cu128

# установка окружения под 3.12 + CUDA 12.8
uv sync --python 3.12 --extra cu128
```

Чтобы постоянно работать с одной версией Python и не писать флаг каждый раз,
Можно зафиксировать её в `.python-version`:

```bash
uv python pin 3.12   # создаст/обновит файл .python-version
```

После этого uv автоматически возьмёт прописанную версию при любой операции.
Если pinned-версия нарушает ограничение `requires-python`, uv выдаст ошибку.

> ⚠️  В релизе uv 0.7 флаг называется именно `--python`; параметра
> `--python-version` нет, поэтому команды вида `uv sync --python-version 3.12`
> завершатся ошибкой «unexpected argument».  Проверяйте подсказку `--help` при
> обновлении инструмента.

### Как это устроено

* Внутренние пакеты (например `model-qwen2_5-vl`) содержат лишь абстрактную зависимость `torch>=2.4,<2.8`, без суффиксов `+cu…`. Таким образом один и тот же исходный код подходит и для CPU, и для разных сборок CUDA.
* Конкретный backend (cu124, cu128 или cpu) задаётся ТОЛЬКО в корневом `pyproject.toml` через `[tool.uv.sources]` **или** переменной окружения `UV_TORCH_BACKEND`.
* Значения в `tool.uv.sources` корня переопределяют любые источники, объявленные во вложенных пакетах workspace.
* Lock-файл фиксирует именно те колёса (`torch==…+cu124`, `torchvision==…+cu128` …), которые выбраны во время `uv lock`.  Если параллельно поддерживаются несколько backend-ов, заведите отдельные lock-файлы (`uv-cu124.lock`, `uv-cu128.lock`) или пересчитывайте lock в CI.

#### Быстрый способ переключения backend-а
```bash
export UV_TORCH_BACKEND=cu128   # cpu | cu124 | cu128
uv sync --locked                # возьмёт нужные колёса без правки файлов
```
Этот приём особенно удобен в Dockerfile:
```Dockerfile
ARG TORCH_BACKEND=cu124
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}
RUN uv sync --locked
```

#### Установка опубликованного пакета из PyPI
Если пакет будет загружен на PyPI с optional-extras, пользователь сможет выбрать вариант так:
```bash
pip install model-qwen2-5-vl[cu124] \
  --extra-index-url=https://download.pytorch.org/whl/cu124
```