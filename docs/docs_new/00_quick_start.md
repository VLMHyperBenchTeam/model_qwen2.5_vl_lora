# Быстрый старт

**Цель:** за 5 минут от `git clone` до первого запуска.

## 1. Клонирование проекта

```bash
# Клонируем с submodules (наши python-пакеты)
git clone --recursive https://github.com/your-org/your-project.git
cd your-project

# Если забыли --recursive
git submodule update --init --recursive
```

## 2. Настройка окружения

```bash
# Фиксируем версию Python (обязательно!)
uv python pin 3.12

# Выбираем CUDA-версию и устанавливаем все зависимости
uv sync --extra cu124   # CUDA 12.4 + cuDNN
# или
uv sync --extra cu128   # CUDA 12.8 + cuDNN
```

> ⚠️ **Важно**: обычный `uv sync` завершится ошибкой - нужно обязательно указать `--extra cu124` или `cu128`

## 3. Проверка работы

```bash
# Запуск тестового скрипта
uv run python check_classification.py

# Проверка импорта пакетов
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

## 4. Структура проекта

```
your-project/
├── packages/                   # Наши python-пакеты (git submodules)
│   ├── bench-utils/           # Утилиты для бенчмаркинга
│   ├── model-interface/       # Интерфейсы для моделей
│   └── model-qwen2-5-vl/     # Конкретная модель
├── pyproject.toml             # Dev-конфигурация
├── staging/pyproject.toml     # Staging-конфигурация
├── prod/pyproject.toml        # Prod-конфигурация
└── docker/Dockerfile-uv       # Универсальный Docker
```

## 5. Основные команды

```bash
# Запуск скриптов
uv run python your_script.py

# Добавление зависимостей
uv add numpy>=1.26

# Обновление пакетов
git submodule update --remote --merge

# Переключение режимов
uv sync --project staging --frozen --extra cu124  # staging
uv sync --project prod --frozen --extra cu124     # prod
```

## 🎯 Что дальше?

### Понимание архитектуры
- **[Концепция архитектуры](01_architecture_concept.md)** — зачем микропакеты и три режима работы

### Настройка под себя
- **[Настройка окружения](03_environment_setup.md)** — Python, CUDA backends, детальная настройка
- **[Управление зависимостями](04_dependency_management.md)** — `uv add`, extras, обновления

### Ежедневная работа
- **[Запуск скриптов](05_running_scripts.md)** — `uv run` vs активация venv
- **[Добавление пакетов](06_adding_packages.md)** — git submodules, workspace

### Продакшен
- **[Docker и деплой](07_docker_deployment.md)** — сборка образов, CI/CD

---

> 💡 **Совет**: если что-то не работает, проверьте:
> 1. Зафиксирована ли версия Python через `uv python pin`?
> 2. Инициализированы ли git submodules?
> 3. Указали ли `--extra cu124` или `cu128`?