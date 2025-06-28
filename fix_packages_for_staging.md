Ошибка в `uv lock --project prod` возникает, потому что в тегах пакетов (например `model_qwen2.5-vl@v0.1.2.dev3`) в их `pyproject.toml` до сих пор есть записи

```toml
[tool.uv.sources]
model-interface = { workspace = true }
```

uv-workspace-ссылка допустима только в dev-ветке.
При подготовке релиза её нужно удалить.
В `release_tool` эта очистка уже встроена в Stage 4 — функция `_clean_workspace_sources`.

Что нужно сделать ⬇️

───────────────────────────────────────────
1. Перейти в корень монорепы
2. Запустить Stage 4 для bump-коммита пакетов
```bash
uv run release-tool-stage4 \
       --bump patch        # или minor/major
       --push              # сразу пушим в dev_branch
```

Stage 4:

* повышает версию пакета;
* удаляет `workspace = true` из `[tool.uv.sources]`;
* делает commit «prepare release …» в dev-ветке пакета.

───────────────────────────────────────────
3. Открыть PR `dev_branch → main`, пройти ревью и смёржить.

───────────────────────────────────────────
4. После merge выполнить Stage 5 — создание тега и
автоматическое обновление `prod/pyproject.toml`:

```bash
uv run release-tool-stage5 --push
```

Stage 5:

* создаёт аннотированный тег `vX.Y.Z`;
* в корневом `prod/pyproject.toml` проставляет
  `tag = "vX.Y.Z"` для зависимости соответствующего пакета;
* коммитит изменённый `prod/pyproject.toml` (и пушит, если указан `--push`).

───────────────────────────────────────────
5. Теперь можно собрать prod-lock:

```bash
uv lock --project prod           # создаст prod/uv.lock
uv sync --project prod --frozen  # проверка установки по lock-файлу
```

uv больше не встретит `workspace = true`, так как Stage 4 уже
«очистил» релизные `pyproject.toml`, а Stage 5 зафиксировал
нужные теги в `prod/pyproject.toml`.

Если нужно исправить только один-два пакета без полного цикла, можно:

```bash
# пример для model_qwen2.5-vl
cd packages/model_qwen2.5-vl
uv run release_tool/stages/stage4.py --bump patch
git push origin dev_branch
```

Но предпочтительнее пользоваться стандартным
workflow Stage 4 → PR → Stage 5, чтобы версии,
теги и prod-конфигурация всегда были согласованы.