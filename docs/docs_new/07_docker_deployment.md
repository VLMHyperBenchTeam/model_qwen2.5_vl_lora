# Docker и деплой

Полное руководство по сборке Docker-образов и подготовке к продакшен-деплою.

## 1. Подготовка к prod-сборке

### Проверка окружения
```bash
uv lock --locked --project prod    # проверить актуальность lock-файла
uv sync --check --project prod     # убедиться что .venv соответствует lock
```

### Prod-режим
```bash
uv lock --project prod --extra cu124      # создать/пересчитать prod/uv.lock
uv sync --project prod --frozen --extra cu124  # установка строго по lock-файлу
```

Используйте `--frozen`, чтобы гарантировать воспроизводимость в продакшене.

## 2. Универсальный Dockerfile-uv

`docker/Dockerfile-uv` — универсальный Docker-рецепт для dev и prod окружений.

### Основные принципы
- **Один Dockerfile** для dev и prod режимов
- **Многоступенчатая сборка**: `base` → `deps` → `wheel` → `runtime`
- **Гибкие build-аргументы**: CUDA-версия, базовый образ, CMake
- **Кэширование**: эффективное переиспользование слоев Docker

### Архитектура слоев

| Слой | Назначение | Содержимое |
|------|------------|------------|
| **base** | Системные зависимости | Ubuntu + CUDA + Python + статический `uv` |
| **deps-dev/prod** | Зависимости | `uv sync --locked` по соответствующему lock-файлу |
| **wheel-dev/prod** | ML-библиотеки | Сборка `flash-attn` под нужную CUDA-версию |
| **dev/prod** | Финальные образы | Минимальные runtime-слои |

### Build-аргументы

| ARG | По умолчанию | Описание |
|-----|--------------|----------|
| `CUDA_VERSION` | `12.4.1` | Версия CUDA |
| `CUDA_VARIANT` | `cudnn` | Вариант образа (`runtime`, `cudnn`, `base`) |
| `UBUNTU_VERSION` | `22.04` | Версия Ubuntu |
| `CMAKE_VERSION` | `3.26.1` | Версия CMake для сборки flash-attn |

## 3. Примеры сборки

### Dev-образ
```bash
docker build -f docker/Dockerfile-uv \
  --target dev \
  --build-arg CUDA_VERSION=12.4.1 \
  --build-arg CUDA_VARIANT=cudnn \
  -t myproject:dev-cu124 .
```

### Prod-образ
```bash
docker build -f docker/Dockerfile-uv \
  --target prod \
  --build-arg CUDA_VERSION=12.4.1 \
  -t myproject:prod-cu124 .
```

### Разные CUDA-версии
```bash
# CUDA 12.8
docker build -f docker/Dockerfile-uv \
  --target prod \
  --build-arg CUDA_VERSION=12.8.0 \
  --build-arg CMAKE_VERSION=3.22.6 \
  -t myproject:prod-cu128 .
```

## 4. ML-специализация

### Автоматическая сборка flash-attn
- Компилируется под нужную CUDA-версию
- Результат кэшируется в отдельном слое
- Переиспользуется в dev и prod образах

### Lock-файл воспроизводимость
```dockerfile
COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project --all-packages
```

### Минимальные runtime-образы
Prod-образы не содержат:
- Исходные коды проекта
- Dev-зависимости
- Кэши компилятора

## 5. Использование образов

### Dev-режим (с исходниками)
```bash
docker run --gpus=all -it \
  -v $(pwd):/workspace \
  myproject:dev-cu124 \
  bash
```

### Prod-режим (готовое приложение)
```bash
docker run --gpus=all --rm \
  myproject:prod-cu124 \
  uv run python -m my_service
```

## 6. Оптимизация сборки

### Отдельные lock-файлы для разных CUDA
```bash
# Генерация
uv lock --extra cu124 -o uv-cu124.lock
uv lock --extra cu128 -o uv-cu128.lock

# Использование в Dockerfile
ARG LOCK_FILE=uv.lock
COPY ${LOCK_FILE} uv.lock
COPY pyproject.toml ./
RUN uv sync --locked
```

### Кэширование dependencies
```dockerfile
# Сначала копируем только конфигурацию
COPY uv.lock pyproject.toml ./
RUN uv sync --no-install-project --all-packages --frozen

# Потом исходники (кэш deps не сбрасывается)
COPY . .
RUN uv sync --frozen
```

## 7. Отладка и проверка

### Smoke-тест
```bash
docker run --gpus all --rm -it myproject:dev-cu124 \
  python -c "import torch, flash_attn; print('CUDA:', torch.version.cuda, 'FlashAttn:', flash_attn.__version__)"
```

### Извлечение wheel на хост
```bash
id=$(docker create myproject:dev-cu124)
docker cp "$id":/wheelhouse/flash_attn*.whl .
docker rm "$id"
```

### Отладка ошибок сборки
```bash
# Запуск на промежуточном слое
docker build -f docker/Dockerfile-uv --target deps-dev .
docker run -it <image_id> bash
```

## 8. CI/CD интеграция

### GitHub Actions пример
```yaml
- name: Build Docker images
  run: |
    docker build -f docker/Dockerfile-uv \
      --target prod \
      --build-arg CUDA_VERSION=12.4.1 \
      --cache-from myproject:cache \
      --cache-to myproject:cache \
      -t myproject:${{ github.sha }} .
```

### Multi-platform сборка
```bash
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile-uv \
  --target prod \
  -t myproject:prod-cu124 .
```

## 🎯 Что дальше?

- **[Инструменты и автоматизация](08_tooling_automation.md)** — `release_tool`, CI/CD, автоматизация релизов

---

> 💡 **Лучшие практики**:
> - Используйте `--frozen` для воспроизводимости
> - Кэшируйте слои зависимостей
> - Тестируйте образы с `--gpus all`
> - Генерируйте отдельные lock-файлы для разных CUDA-версий