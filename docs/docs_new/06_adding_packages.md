# Добавление собственных пакетов

Подключение нового или существующего пакета в **трёх workspace**: *dev*, *staging* и *prod*.

> Перед выполнением команд убедитесь, что окружение создано с выбранным CUDA-бэкендом PyTorch (см. [Настройка окружения](03_environment_setup.md)):
> ```bash
> uv python pin 3.12
> uv sync --extra cu124   # или cu128
> ```

## 1. Git submodules

Все внутренние пакеты подключаются в репозиторий как **git submodule**. Это позволяет:

- фиксировать пакет на конкретной ревизии;
- обновлять его командой `git submodule update --remote`;
- иметь отдельную историю коммитов.

### Основные команды

```bash
git submodule update --init --recursive          # git clone + инициализация

git submodule update --remote --merge            # обновить все пакеты

# добавить новый пакет
git submodule add https://github.com/ORG/new_pkg.git packages/new_pkg

# удалить пакет
git submodule deinit -f packages/old_pkg && git rm -f packages/old_pkg
```

> После добавления нового submodule не забудьте выполнить:
> ```bash
> git submodule update --init
> ```
> чтобы подтянуть исходники и корректно зафиксировать хеш в родительском репозитории.

## 2. dev-сборка (packages/* в режиме workspace)

1. Убедитесь, что пакет присутствует в каталоге `packages/<pkg>` (добавлен как submodule) и содержит `pyproject.toml`.
1. Убедиться, что в корневом `pyproject.toml` пакет объявлен в `[tool.uv.sources]`:
   ```toml
   [tool.uv.sources]
   hello-world = { workspace = true }
   ```
1. При необходимости добавить пакет в `[project].dependencies` там, где он используется.
1. Обновить окружение dev:
   ```bash
   uv sync --extra cu124 --all-packages   # или cu128
   ```

## 3. staging-сборка (фиксация dev-тегов)

После того как пакет получил dev-тег (например `v0.1.0.dev0`):

1. Откройте `staging/pyproject.toml` и добавьте источник Git:
   ```toml
   [tool.uv.sources]
   hello-world = { git = "https://github.com/USER/hello_world", tag = "v0.1.0.dev0", subdirectory = "." }
   ```
2. Пересчитайте и установите окружение staging:
   ```bash
   uv lock --project staging
   uv sync --project staging --frozen --extra cu124   # или cu128
   ```

## 4. prod-сборка (стабильный релиз)

В `prod/pyproject.toml` укажите стабильный Git-тег:

```toml
[tool.uv.sources]
hello-world = { git = "https://github.com/USER/hello_world", tag = "v0.1.0", subdirectory = "." }
```

Далее:
```bash
uv lock --project prod
uv sync --project prod --frozen --extra cu124   # или cu128
```

## 🎯 Что дальше?

### Продакшен и деплой
- **[Docker и деплой](07_docker_deployment.md)** — сборка Docker-образов для всех режимов
- **[Инструменты и автоматизация](08_tooling_automation.md)** — release_tool, автоматизация релизов

### Настройка
- **[Управление зависимостями](04_dependency_management.md)** — добавление внешних зависимостей в пакеты
- **[Dev/Staging/Prod режимы](02_dev_staging_prod.md)** — детали работы с разными режимами
