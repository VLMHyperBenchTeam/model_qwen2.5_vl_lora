# Prod-сборка и Docker

В этом документе описывается минимальный набор шагов для подготовки production-окружения и сборки контейнера.

## 1. Проверка lock-файла и окружения

```bash
uv lock --locked    # ошибка, если lock устарел
uv sync --check     # убедиться, что .venv соответствует lock
```

## 2. Prod-режим

```bash
uv lock --project prod          # создать или пересчитать prod/uv.lock
uv sync --project prod --frozen # установка по lock-файлу
```

Используйте `--frozen`, чтобы гарантировать воспроизводимость.

## 3. Docker

Для базовых слоёв Docker-сборки полезны опции `uv sync`:

- `--no-install-project` — не ставить корневой пакет;
- `--all-packages` — установить все workspace-пакеты;
- `--frozen` — работать строго по lock-файлу.

Минимальный слой зависимостей без исходников:

```Dockerfile
COPY uv.lock pyproject.toml ./
RUN uv sync --no-install-project --all-packages --frozen \
    && rm -rf ~/.cache/uv
```

После копирования исходников выполняйте обычный `uv sync --frozen` — слой с deps кэшируется.

______________________________________________________________________

**Полная инструкция со всеми аргументами, параметрами CUDA и FAQ находится в** [`07_docker_builder_full.md`](07_docker_builder_full.md).
