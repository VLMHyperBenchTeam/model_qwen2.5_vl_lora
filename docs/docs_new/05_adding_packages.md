# Добавление собственных пакетов

Подключение нового или существующего пакета в **трёх workspace**: *dev*, *staging* и *prod*.

> Перед выполнением команд убедитесь, что окружение создано с выбранным CUDA-бэкендом PyTorch (см. 09_torch_backends.md):
> ```bash
> uv sync --extra cu124   # или cu128
> ```

## 1. dev-сборка (packages/* в режиме workspace)

1. Скопировать пакет в `packages/<pkg>` (должен содержать `pyproject.toml`).
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

## 2. staging-сборка (фиксация dev-тегов)

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

## 3. prod-сборка (стабильный релиз)

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

## 4. Git submodules

Все внутренние пакеты подключаются как submodule, что позволяет:

- фиксировать пакет на конкретной ревизии;
- обновлять одним `git submodule update --remote`;
- иметь отдельную историю коммитов.

### Основные команды

```bash
git submodule update --init --recursive          # клон + инициализация

git submodule update --remote --merge            # обновить все пакеты

git submodule add https://github.com/ORG/new_pkg.git packages/new_pkg   # добавить

git submodule deinit -f packages/old_pkg && git rm -f packages/old_pkg  # удалить
```

> После добавления нового submodule не забудьте выполнить:
> ```bash
> git submodule update --init
> ```
> чтобы подтянуть исходники и корректно зафиксировать хеш в родительском репозитории.
