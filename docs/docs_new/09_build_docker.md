# 9. Сборка Docker-образов по JSON-конфигу (`build_docker.py`)

Этот скрипт позволяет описывать параметры Docker-сборки в декларативном JSON-файле и запускать **одну или несколько** сборок одной командой.

> **Зачем?**
> • Избавиться от громоздких `bash`-строк в CI.
> • Хранить разные пресеты (dev, prod, CUDA-версии) в VCS.
> • Легко комбинировать аргументы и кеш-слои.

---
## 1. Быстрый старт

1. Опишите сборку в `docker/build_config.json`:
   ```jsonc
   {
     "defaults": {
       "dockerfile": "docker/Dockerfile-uv",   // путь к Dockerfile
       "context": "."                          // контекст сборки
     },
     "builds": [
       {
         "tag": "myproject:prod-cu124",        // итоговый тег
         "mode": "prod",                       // режим (dev|prod)
         "base_image": "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", // базовый FROM
         "uv_version": "0.7.18",               // версия uv
         "cmake_version": "3.29.2",            // версия CMake
         "toml_path": "prod/pyproject.toml",   // pyproject для prod
         "lock_path": "py310_cu124.lock",      // сгенерированный lock-файл
         "torch_index": "https://download.pytorch.org/whl/cu124", // репозиторий PyTorch wheels
         "platforms": "linux/amd64",           // multi-arch (опц.)
         "load": true                           // загрузить в локальный докер
       }
     ]
   }
   ```

2. Запустите скрипт:
   ```bash
   ./build_docker.py docker/build_config.json        # полноценная сборка через buildx
   ./build_docker.py docker/build_config.json --dry-run  # только вывести команды
   ```

> ⚙️  Скрипт требует установленный **docker buildx** (входит в Docker CLI ≥20.10).

---
## 2. Формат конфигурации

| Ключ                | Тип          | Описание |
|---------------------|--------------|----------|
| `defaults.context`  | `string`     | Контекст сборки (`.` по умолчанию) |
| `defaults.dockerfile` | `string`   | Путь к Dockerfile (по умолчанию `docker/Dockerfile-uv`) |
| **Описание одного build-объекта** | | |
| `tag`               | `string`     | Итоговый тег образа (`myproject:dev-cu124`) |
| `mode`              | `string`     | Режим сборки (`dev` / `prod`) |
| `base_image`*       | `string`     | Полный базовый образ (обязательный) |
| `uv_version`        | `string`     | Версия uv, передаётся как ARG `UV_VERSION` |
| `cmake_version`     | `string`     | Версия CMake, передаётся как ARG `CMAKE_VERSION` |
| `toml_path`*        | `string`     | Путь к `pyproject.toml` (обязательный, ARG `TOML_PATH`) |
| `lock_path`*        | `string`     | Путь к lock-файлу (обязательный, ARG `LOCK_PATH`) |
| `torch_index`*      | `string`     | URL репозитория PyTorch wheels (обязательный, ARG `TORCH_INDEX`) |
| `platforms`         | `string`     | Значение `--platform` (пример `linux/amd64,linux/arm64`) |
| `push`              | `bool`       | Добавляет `--push` для публикации в registry |
| `load`              | `bool`       | Добавляет `--load` для загрузки в локальный докер |
| `no_cache`          | `bool`       | Принудительная сборка без кеша |
| `cache_from`        | `string`     | `--cache-from` источник кеша |
| `cache_to`          | `string`     | `--cache-to` приёмник кеша |
| `dockerfile`        | `string`     | Переопределяет `defaults.dockerfile` |
| `context`           | `string`     | Переопределяет `defaults.context` |
| `args`              | `object`     | Произвольные дополнительные `--build-arg` (опц.) |

`builds` может быть массивом или одним объектом.  Если файл содержит **массив** верхнего уровня — считается списком сборок, а секция `defaults` не используется.

---
## 3. Несколько сборок в одном файле

```jsonc
[
  {
    "tag": "myproject:dev-cu124",
    "mode": "dev",
    "base_image": "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
    "toml_path": "pyproject.toml",
    "lock_path": "py310_cu124.lock",
    "torch_index": "https://download.pytorch.org/whl/cu124",
  },
  {
    "tag": "myproject:prod-cu124",
    "mode": "prod",
    "base_image": "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
    "toml_path": "prod/pyproject.toml",
    "lock_path": "prod/py310_cu124.lock",
    "torch_index": "https://download.pytorch.org/whl/cu124",
  }
]
```

Запуск выполнит обе сборки последовательно.

---
## 4. Интеграция с CI/CD

Пример шага **GitHub Actions**:
```yaml
- name: Build & push Docker images
  run: |
    ./build_docker.py docker/build_config.json
```

В Jenkins/GitLab CI аналогично: скрипт читается как обычный CLI.

---
## 5. Отладка

1. Добавьте флаг `--dry-run`, чтобы увидеть точные команды `docker buildx`.
2. Задайте `no_cache: true`, если нужно принудительно пересобрать слой.
3. Чтобы войти во временный слой, укажите `--target deps-dev` и добавьте `load: true`, затем:
   ```bash
   docker run -it <image_id> bash
   ```

---
## 6. Связь с предыдущей документацией

Скрипт дополняет гайд из «[07_docker_deployment](07_docker_deployment.md)» и не заменяет `docker build` напрямую — он лишь автоматизирует передачу аргументов и тегов.  Логика слоёв `Dockerfile-uv` остаётся неизменной.

---
## 7. Чек-лист перед использованием

- [ ] Установлен Docker с поддержкой **BuildKit/Buildx**.
- [ ] Файл `docker/build_config.json` валиден (проверить `jq .`).
- [ ] Настроен логин в нужный Docker-registry, если используется `push`.
- [ ] Есть кеш-репозиторий, если указываете `cache_to` / `cache_from`.

> **ℹ️ Примечание**. Начиная с Docker Engine 20.10, Buildx и BuildKit поставляются «из коробки» — дополнительная установка не требуется.
> Команда `docker build` уже использует Buildx под капотом, но мы вызываем `docker buildx build`, чтобы получить расширенные возможности (`--platform`, `--load/--push`, кеш-слои и т. д.).  См. официальную справку [docker buildx build](https://docs.docker.com/reference/cli/docker/buildx/build/) и обзор [Docker Build Overview](https://docs.docker.com/build/concepts/overview/).

Теперь сборка Docker-образов описывается одной JSON-строкой! 🎉