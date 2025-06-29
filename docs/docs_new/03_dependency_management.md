# Управление зависимостями

Документ объединяет все рецепты добавления и обновления зависимостей в workspace с помощью **`uv`**.

## 1. Добавить зависимость во внутренний пакет

1. Открыть `packages/<pkg>/pyproject.toml`.
1. В секции `[project].dependencies` добавить строку, например:
   ```toml
   numpy>=1.26
   ```
1. Из корня выполнить:
   ```bash
   uv lock
   uv sync
   ```

## 2. Добавить зависимость в сам проект (корень)

Если пакет нужен и скриптам верхнего уровня, расширяем корневой `pyproject.toml`:

```toml
[project]
dependencies = [
    ...,
    "numpy>=1.26",
]
```

Далее всё так же: `uv lock && uv sync`.

## 3. Быстрый чек-лист

1. Изменить `pyproject.toml`.
1. `uv lock` → пересчёт `uv.lock`.
1. `uv sync` → установка.
1. Закоммитить изменения.

## 4. Команда `uv add`

Удобная альтернатива ручному редактированию.

### 4.1 Глобальная зависимость (корень)

```bash
uv add numpy>=1.26
```

### 4.2 Dev-зависимость

```bash
uv add ruff --group dev
```

Запись попадёт в `[dependency-groups].dev`.

### 4.3 Зависимость только для конкретного пакета

```bash
# из каталога пакета
cd packages/bench_utils
uv add numpy>=1.26

# или из корня
uv add numpy>=1.26 --package bench-utils
```

### 4.4 Особые случаи

- `--optional extra_name` → `[project.optional-dependencies]`.
- `--index`, `--path`, `--git` задают альтернативные источники.
- `--bounds major|minor|exact` формирует ограничение версии.

## 5. Пример комплексного добавления

```bash
uv add torch>=2.3 transformers>=4.41 qwen-vl-utils>=0.0.10 \
       --package model-qwen2_5-vl
```

После этого:

1. строки добавятся в `packages/model_qwen2.5-vl/pyproject.toml`;
1. `uv.lock` обновится;
1. пакеты установятся в `.venv`.

### FlashAttention как optional-extra

```bash
uv add flash-attn --package model-qwen2_5-vl --optional flash
```

Для установки используйте:

```bash
uv sync --extra flash           # навсегда
uv run --extra flash python …   # разово
```
