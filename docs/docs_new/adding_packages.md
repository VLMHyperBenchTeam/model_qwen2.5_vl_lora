# Добавление собственных пакетов

Алгоритм подключения нового или существующего пакета в workspace.

## 1. Существующий пакет в каталоге `packages/*`

1. Скопировать пакет в `packages/<pkg>` (должен содержать `pyproject.toml`).
1. Убедиться, что в корневом `pyproject.toml` пакет объявлен в `[tool.uv.sources]`:
   ```toml
   [tool.uv.sources]
   hello-world = { workspace = true }
   ```
1. При необходимости добавить пакет в `[project].dependencies` там, где он используется.
1. Обновить окружение:
   ```bash
   uv sync --all-packages
   ```

## 2. Создать новый пакет с нуля

```bash
uv init packages/my_cool_lib
```

Команда сгенерирует структуру проекта и автоматически добавит каталог в workspace.

## 3. Проверка установки

```bash
uv run python -c 'import hello_world; print("Импорт успешен!")'
```

## 4. Подключение пакета в prod-сборку

В `pyproject.prod.toml` задайте источник Git:

```toml
[tool.uv.sources]
hello-world = { git = "https://github.com/USER/hello_world", tag = "0.0.2", subdirectory = "." }
```

## 5. Git submodules

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
