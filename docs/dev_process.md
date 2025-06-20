# Два режима разработки

Используется uv work spaces

Настроил два режима:

1. Разработка (текущий `pyproject.toml`)  
   • `workspace = true` → пакеты ставятся editable из каталога `packages/*`.

2. Прод-режим — новый файл `pyproject.prod.toml`  
   • В нём только блок `[tool.uv.sources]`, где каждый пакет берётся из этого же репозитория по тегу и подпапке:  
     ```toml
     bench-utils      = { git = ".../vlm-hyperbench.git", tag = "bench-utils-v0.1.2", subdirectory = "packages/bench_utils" }
     model-interface  = { git = ".../vlm-hyperbench.git", tag = "model-interface-v0.1.2", subdirectory = "packages/model_interface" }
     model-qwen2-5-vl = { git = ".../vlm-hyperbench.git", tag = "model-qwen2-5-vl-v0.1.2", subdirectory = "packages/model_qwen2.5-vl" }
     ```
   • При выпуске новой версии меняете `tag = ...` на актуальный.

Как пользоваться  
```
# dev-режим
uv lock
uv sync

# prod-режим
uv lock --project pyproject.prod.toml   # создаёт/обновит uv.lock по git-тегам
uv sync --production                    # установит пакеты из GitHub
```

(`uv >= 0.7` должен быть, иначе эта функция не работает!)

**Итого:** 
• «dev» — мгновенное редактирование;  
• «prod» — чистые сборки из тегов GitHub без PyPI.