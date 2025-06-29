# Вспомогательные инструменты и Git-хуки

## 1. `release_tool`

`release_tool` подключён к workspace как dev-зависимость.

- Доступен в окружении по команде `release-tool-*` после `uv sync`.
- В prod-сборке отсутствует (используйте `uv sync --no-dev`).
- Для CI можно установить только dev-группу:
  ```bash
  uv sync --group dev
  ```

## 2. Git-хуки через `pre-commit`

Файл `.pre-commit-config.yaml` включает:
| Хук | Repo | Назначение |
|-----|------|------------|
| `ruff` | `charliermarsh/ruff-pre-commit` | статический анализ Python |
| `trailing-whitespace` | `pre-commit/pre-commit-hooks` | удаляет пробелы в конце строк |

### Установка

```bash
uv sync --group dev          # установить ruff, pre-commit …
uv run pre-commit install    # добавить .git/hooks/pre-commit

# или сразу все возможные хуки
uv run pre-commit install --all-hooks
```

### Ручной запуск

```bash
uv run pre-commit run --all-files          # проверить всё
uv run pre-commit run ruff --files file.py # выбрать хук/файлы
```

### Обход проверок

```bash
git commit --no-verify -m "chore: hot-fix"
```

Используйте только в крайнем случае.

## 🎯 Что дальше?

### Архитектура и настройка
- **[Концепция архитектуры](01_architecture_concept.md)** — понимание общих принципов проекта
- **[Настройка окружения](03_environment_setup.md)** — Python версии и CUDA backends

### Продакшен
- **[Docker и деплой](07_docker_deployment.md)** — сборка образов и CI/CD интеграция

---

> 💡 **Совет**: настройте `pre-commit` с самого начала проекта, чтобы автоматически поддерживать качество кода.
