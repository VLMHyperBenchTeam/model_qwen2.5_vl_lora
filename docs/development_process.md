# Обзор процесса разработки

1. [Общая концепция](docs_new/concept_overview.md) — микропакетная архитектура, режимы dev/prod и релизный пайплайн.
1. [Dev ↔ Prod](docs_new/dev_vs_prod.md) — переключение окружений и базовые команды `uv`.
1. [Управление зависимостями](docs_new/dependency_management.md) — добавление/обновление пакетов, `uv add`, extras.
1. [Запуск скриптов](docs_new/running_scripts.md) — использование `uv run` и активация `.venv`.
1. [Добавление пакетов](docs_new/adding_packages.md) — подключение workspace-пакетов и работа с Git-submodules.
1. [Prod-сборка & Docker](docs_new/prod_build_and_docker.md) — минимальный рецепт prod-режима и сборка контейнера.
1. [Полный гайд Docker](docs_new/docker_builder_full.md) — детальное описание `Dockerfile-uv`, ARG-параметры, FAQ.
1. [Инструменты и Git-хуки](docs_new/tooling_and_hooks.md) — `release_tool`, `pre-commit`, `ruff`.
1. [Torch backend](docs_new/torch_backends.md) — выбор CUDA extras, переменная `UV_TORCH_BACKEND`, отдельные lock-файлы.
1. [Версии Python](docs_new/python_versions.md) — фиксация интерпретатора (`--python`) и файл `.python-version`.
