# Dev-, Staging- и Prod-режимы

В проекте настроены три независимых окружения с помощью `uv workspaces`.

## 1. Структура конфигураций

- **dev** — текущий `pyproject.toml` (`workspace = true` → пакеты устанавливаются editable из `packages/*`).
- **staging** — `staging/pyproject.toml`, где внутренние пакеты фиксируются на *dev*-теги, готовые к ручному тестированию.
- **prod** — `prod/pyproject.toml`, где все внутренние пакеты берутся по стабильному `git tag`.

Пример блока `[tool.uv.sources]` в prod-конфигурации:

Примеры блоков `[tool.uv.sources]`.

### staging-конфигурация

```toml
bench-utils      = { git = "https://github.com/VLMHyperBenchTeam/bench-utils.git",      tag = "v0.1.2.dev0", subdirectory = "." }
model-interface  = { git = "https://github.com/VLMHyperBenchTeam/model-interface.git",  tag = "v0.1.2.dev0", subdirectory = "." }
model-qwen2-5-vl = { git = "https://github.com/VLMHyperBenchTeam/model-qwen2-5-vl.git", tag = "v0.1.2.dev0", subdirectory = "." }
```

### prod-конфигурация

```toml
bench-utils      = { git = "https://github.com/VLMHyperBenchTeam/bench-utils.git",      tag = "v0.1.2", subdirectory = "." }
model-interface  = { git = "https://github.com/VLMHyperBenchTeam/model-interface.git",  tag = "v0.1.2", subdirectory = "." }
model-qwen2-5-vl = { git = "https://github.com/VLMHyperBenchTeam/model-qwen2-5-vl.git", tag = "v0.1.2", subdirectory = "." }
```

## 2. Базовые команды

### dev-режим

```bash
uv lock
# Выбор backend-а PyTorch обязателен (см. 09_torch_backends.md)
uv sync --extra cu124   # или cu128
```

> Обычный `uv sync` без указания `--extra` завершится ошибкой, так как нужно выбрать CUDA-бэкенд PyTorch. Подробности — в файле «09_torch_backends.md».

### staging-режим

```bash
# сформировать или обновить lock-файл
uv lock --project staging

# проверить актуальность lock-файла
uv lock --project staging --locked

# установить окружение строго по lock-файлу
uv sync --project staging --frozen
```

### prod-режим

```bash
# сформировать или обновить lock-файл
uv lock --project prod

# проверить актуальность lock-файла
uv lock --project prod --locked

# установить окружение строго по lock-файлу
uv sync --project prod --frozen
```

> Примечание: вместо каталога (`staging` или `prod`) можно передать путь к конкретному файлу:
>
> ```bash
> uv lock  --project staging/pyproject.toml
> uv sync  --project staging/pyproject.toml --frozen
> ```

## 3. Итог

- **dev** — активная разработка, editable-пакеты;
- **staging** — предрелизное тестирование из *dev*-тегов;
- **prod** — воспроизводимая сборка из стабильных Git-тегов без PyPI.