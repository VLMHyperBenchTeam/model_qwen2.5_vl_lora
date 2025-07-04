# Управление зависимостями

Здесь показаны все способы добавления и обновления зависимостей в `uv workspace` с помощью **`uv`**.

Все описанное ниже будет работать **только** после настройки окружения:
```bash
uv python pin 3.12         # фиксация Python
uv sync --extra cu124       # выбор CUDA и установка
```

> 📖 **Предварительная настройка**: [Настройка окружения](03_environment_setup.md) — Python версии и CUDA backends

## 1. Команда `uv add` (рекомендуемый способ)

Удобная команда для добавления зависимостей.

### 1.1 Глобальная зависимость (наш проект)

```bash
uv add numpy>=1.26
```

Команда:
- добавит строку `numpy>=1.26` в секцию `[project].dependencies` корневого `pyproject.toml` нашего проекта;
- обновит `uv.lock`, подобрав совместимую версию;
- установит пакет в виртуальное окружение нашего проекта.

### 1.2 Dev-зависимость

```bash
uv add ruff --group dev
```

Команда:
- зависимость записывается в группу **dev** внутри `[dependency-groups]` нашего проекта;
- такие пакеты устанавливаются **только** при обычных `uv sync` (по умолчанию dev-группа активна) и исключаются из финальной prod-сборки при `uv sync --no-default-groups`.

### 1.3 Зависимость только для конкретного пакета

```bash
# из каталога пакета
cd packages/bench_utils
uv add numpy>=1.26

# или из корня проекта
uv add numpy>=1.26 --package bench-utils
```

Команда:
- добавляет зависимость **только** в python-пакет `packages/bench_utils/pyproject.toml`;
- workspace нашего проекта останется без изменений;
- lock-файл и окружение будут пересчитаны автоматически.

### 1.4 Особые случаи

- `--optional extra_name` — помещает зависимость в `[project.optional-dependencies]`, доступную через `--extra`;
  пример: `uv add flash-attn --optional flash`.
- `--index`, `--path`, `--git` — позволяют указать альтернативный источник (PyPI-зеркало, локальный путь, Git-репозиторий).
  пример: `uv add mylib --git "https://github.com/user/mylib.git"`.
- `--bounds major|minor|exact` — управляет формой диапазона версий (semver).
  пример: `uv add pandas --bounds minor` добавит `pandas>=2,<3`.

### 1.5 Пример комплексного добавления

```bash
uv add torch>=2.3 transformers>=4.41 qwen-vl-utils>=0.0.10 \
       --package model-qwen2_5-vl
```

Эта команда за раз:
1. добавит три зависимости в `packages/model_qwen2.5-vl/pyproject.toml`;
2. пересчитает `uv.lock`, учитывая новые пакеты;
3. установит их в `.venv`, подготовив окружение к запуску модели.

### 1.6 FlashAttention как optional-extra

```bash
uv add flash-attn --package model-qwen2_5-vl --optional flash
```

Что даёт:
- пакет `flash-attn` будет установлен **только**, если при `uv sync` (или `uv run`) указать `--extra flash`;
- основной набор зависимостей останется без тяжёлого CUDA-расширения, пока оно реально не нужно.

Для установки используйте:

```bash
uv sync --extra flash           # навсегда (до следующего sync)
uv run --extra flash python …   # разово
```

## 2. Ручное редактирование (альтернативный способ)

### 2.1 Добавить зависимость во внутренний пакет

1. Открыть `packages/<pkg>/pyproject.toml`.
1. В секции `[project].dependencies` добавить строку, например:
   ```toml
   numpy>=1.26
   ```
1. Из корня выполнить:
   ```bash
   uv lock
   uv sync
   ```

### 2.2 Добавить зависимость в сам проект (корень)

Если пакет нужен и скриптам верхнего уровня, расширяем корневой `pyproject.toml`:

```toml
[project]
dependencies = [
    ...,
    "numpy>=1.26",
]
```

Далее всё так же: `uv lock && uv sync`.

## 3. Быстрый чек-лист

1. **Рекомендуется**: использовать `uv add` с нужными флагами.
2. **Альтернативно**: изменить `pyproject.toml` вручную.
3. `uv lock` → пересчёт `uv.lock` (если редактировали вручную).
4. `uv sync` → установка (если редактировали вручную).
5. Закоммитить изменения.

## 🎯 Что дальше?

### Разработка
- **[Запуск скриптов](05_running_scripts.md)** — использование `uv run` и активация виртуальной среды
- **[Добавление пакетов](06_adding_packages.md)** — подключение новых workspace-пакетов

### Продакшен
- **[Docker и деплой](07_docker_deployment.md)** — сборка prod-образов и деплой
