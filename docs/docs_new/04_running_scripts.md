# Запуск скриптов

В workspace есть два сценария запуска кода.

## 1. «Правильный» способ — `uv run`

`uv run` гарантирует, что:

- `uv.lock` совпадает с `pyproject.toml`;
- `.venv` актуальна (создаётся/обновляется при необходимости);
- команда выполняется внутри виртуального окружения.

Примеры:

```bash
uv run python check_classification.py               # обычный скрипт
uv run ./check_classification.py                    # если есть shebang
uv run python check_page_sorting.py --config conf.json
uv run bash scripts/do_something.sh                  # shell-скрипт
uv run --with "torch==2.3.0" python my_bench.py    # временно другая версия пакета
```

## 2. Классический способ — активировать `.venv`

```bash
uv sync                       # убедиться, что всё установлено
source .venv/bin/activate     # Linux/macOS
# .venv\Scripts\activate     # Windows PowerShell
python check_classifiication.py
```

При таком подходе следить за актуальностью окружения придётся вручную.

### Итог

- Для одиночных запусков и CI используйте `uv run` — безопаснее.
- Для интерактивной работы допустимо активировать `.venv` после `uv sync`.
