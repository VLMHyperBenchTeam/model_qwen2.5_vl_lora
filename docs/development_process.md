# Обзор процесса разработки

1. [Общая концепция](docs_new/01_concept_overview.md) — микропакетная архитектура, режимы dev/prod и релизный пайплайн.
2. [Dev / Staging / Prod](docs_new/02_dev_staging_prod.md) — переключение окружений и базовые команды `uv`.
3. [Управление зависимостями](docs_new/03_dependency_management.md) — добавление/обновление пакетов, `uv add`, extras.
4. [Запуск скриптов](docs_new/04_running_scripts.md) — использование `uv run` и активация `.venv`.
5. [Добавление пакетов](docs_new/05_adding_packages.md) — подключение workspace-пакетов и работа с Git-submodules.
6. [Prod-сборка & Docker](docs_new/06_prod_build_and_docker.md) — минимальный рецепт prod-режима и сборка контейнера.
7. [Полный гайд Docker](docs_new/07_docker_builder_full.md) — детальное описание `Dockerfile-uv`, ARG-параметры, FAQ.
8. [Инструменты и Git-хуки](docs_new/08_tooling_and_hooks.md) — `release_tool`, `pre-commit`, `ruff`.
9. [Torch backend](docs_new/09_torch_backends.md) — выбор CUDA extras, переменная `UV_TORCH_BACKEND`, отдельные lock-файлы.
10. [Версии Python](docs_new/10_python_versions.md) — фиксация интерпретатора (`--python`) и файл `.python-version`.
