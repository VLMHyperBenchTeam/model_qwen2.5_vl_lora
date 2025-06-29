# Настройка окружения

Базовая настройка Python-интерпретатора и PyTorch-бэкендов для корректной работы проекта.

## 1. Фиксация версии Python (обязательно)

Проект поддерживает диапазон `>=3.10, !=3.11.*, <3.13`, но в реальной работе **всегда** фиксируется одна конкретная версия для воспроизводимости.

```bash
uv python pin 3.12   # создает/обновляет .python-version
```

Файл `.python-version` хранится в корне репозитория и читается `uv`, `pyenv`, `asdf` и другими инструментами.

### Смена версии Python

```bash
# Обновляем pin
uv python pin 3.10

# Пересчитываем lock и переустанавливаем окружение
uv lock --extra cu124
uv sync --extra cu124
```

## 2. Выбор PyTorch backend (обязательно)

В проекте предусмотрены два CUDA-бэкенда:

| Extra | CUDA | Описание |
|-------|------|----------|
| `cu124` | 12.4 + cuDNN | Стабильная версия, рекомендуется |
| `cu128` | 12.8 + cuDNN | Новейшая версия |

### Установка с выбранным backend

```bash
uv sync --extra cu124   # CUDA 12.4
uv sync --extra cu128   # CUDA 12.8
```

> ⚠️ **Важно**: обычный `uv sync` без `--extra` завершится ошибкой. PyTorch загружается только из официальных репозиториев `https://download.pytorch.org/whl`.

### Проверка установки

```bash
# Проверить без изменения lock-файла
uv sync --check

# Проверить версии PyTorch
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

## 3. Быстрое переключение через переменную окружения

```bash
export UV_TORCH_BACKEND=cu128
uv sync --locked
```

Удобно для экспериментов, но для стабильной работы лучше использовать явный `--extra`.

## 4. Работа с разными режимами

### Dev-режим
```bash
uv sync --extra cu124
```

### Staging/Prod-режимы
```bash
uv sync --project staging --extra cu124 --frozen
uv sync --project prod --extra cu124 --frozen
```

### Docker-сборка
```bash
docker build -f docker/Dockerfile-uv \
  --target dev \
  --build-arg TORCH_BACKEND=cu128 \
  -t project:dev-cu128 .
```

## 5. Генерация lock-файлов под разные backend

Для разных CUDA-версий можно создавать отдельные lock-файлы:

```bash
# Отдельные lock-файлы для кэширования в CI
uv lock --extra cu124 -o uv-cu124.lock
uv lock --extra cu128 -o uv-cu128.lock

# Для prod-режима
uv lock --project prod --extra cu124 -o prod/uv-cu124.lock
```

## 6. Чек-лист первичной настройки

1. **Клонируем с submodules**:
   ```bash
   git clone --recursive https://github.com/org/project.git
   ```

2. **Фиксируем Python**:
   ```bash
   uv python pin 3.12
   ```

3. **Выбираем CUDA и устанавливаем**:
   ```bash
   uv sync --extra cu124
   ```

4. **Проверяем работу**:
   ```bash
   uv run python -c "import torch; print('OK')"
   ```

## 7. Генерация lock-файла для PyTorch wheels

Иногда нужно собрать _несколько_ окружений (разные CUDA). Вместо проекта можно скомпилировать отдельный lock-файл и потом ставить использовать его.

```bash
CUDA="cu124"
PYTORCH_URL="https://download.pytorch.org/whl/${CUDA}"

# 1) Создаём lock-файл (Python 3.10 + backend cu124)
uv pip compile pyproject.toml \
  --python 3.10 \
  --extra ${CUDA} \
  --output-file py310_${CUDA}.lock \
  --index-url "${PYTORCH_URL}" \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match

# 2) Синхронизируем любое окружение строго по lock-файлу
uv pip sync py310_${CUDA}.lock \
  --index-url "${PYTORCH_URL}" \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match
```

Ключевые флаги:

| Опция | Что делает |
|-------|------------|
| `--index-url` | Указывает основной репозиторий (PyTorch wheels). |
| `--extra-index-url` | Добавляет PyPI вторым источником. |
| `--index-strategy unsafe-best-match` | Разрешает искать пакеты во **всех** индексах, а не только в первом найденном. |

> 🔗 Официальная документация: [uv — pip interface / alternative indexes](https://docs.astral.sh/uv/guides/projects/) и [locking & syncing](https://docs.astral.sh/uv/concepts/projects/sync/).

После этого `py310_cu124.lock` можно положить в репозиторий и использовать в Docker-сборке (см. главу [«Docker и деплой»](07_docker_deployment.md)).

## 🎯 Что дальше?

- **[Управление зависимостями](04_dependency_management.md)** — добавление и обновление пакетов
- **[Запуск скриптов](05_running_scripts.md)** — `uv run` vs активация venv

---

## 💡 Частые проблемы:

- Запустили `uv sync` без `--extra cuXXX` → PyTorch не найдёт бинарники CUDA и сборка упадёт. Всегда добавляйте `--extra cu124` или `cu128` в **каждую** команду `uv lock / sync / pip compile / pip sync`.
- Забыли `uv python pin` → разные версии Python на машинах разработчиков, несовместимые lock-файлы.
- Не выполнили `git submodule update --init --recursive` → отсутствуют внутренние пакеты (import error).
- Собрали окружение по prod-lock (или кастомному `py310_cu124.lock`) без dev-группы → команда `pre-commit` не установлена, git-хуки падают.

Решение:
1. установить стандартное dev-окружение: `uv sync --extra cu124`, **или**
2. добавить только dev-зависимости: `uv pip sync <lock>.lock --group dev`.