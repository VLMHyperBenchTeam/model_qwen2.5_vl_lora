# Dev- vs Prod-режимы

Документ объясняет, как в проекте настроены два независимых окружения с помощью `uv workspaces`.

## 1. Структура конфигураций

- **dev** — текущий `pyproject.toml` (`workspace = true` → пакеты устанавливаются editable из `packages/*`).
- **prod** — отдельный файл `pyproject.prod.toml`, где все внутренние пакеты берутся по `git tag`.

Пример блока `[tool.uv.sources]` в prod-конфигурации:

```toml
bench-utils      = { git = "https://github.com/VLMHyperBenchTeam/bench-utils.git",      tag = "v0.1.2", subdirectory = "." }
model-interface  = { git = "https://github.com/VLMHyperBenchTeam/model-interface.git",  tag = "v0.1.2", subdirectory = "." }
model-qwen2-5-vl = { git = "https://github.com/VLMHyperBenchTeam/model-qwen2-5-vl.git", tag = "v0.1.2", subdirectory = "." }
```

## 2. Базовые команды

```bash
# dev-режим
uv lock
uv sync

# prod-режим (uv ≥ 0.7)
uv lock   --project prod   --frozen   # генерируем/проверяем prod/uv.lock
uv sync   --project prod   --frozen   # устанавливаем зависимости из Git-тегов
```

Ключ `--frozen` запрещает обновлять lock-файл и гарантирует идентичную установку.

Если uv поддерживает прямой путь к файлу:

```bash
uv lock  --project pyproject.prod.toml
uv sync  --project pyproject.prod.toml --frozen
```

## 3. Итог

- **dev** — активная разработка, editable-пакеты.
- **prod** — воспроизводимая сборка из зафиксированных Git-тегов без PyPI.
