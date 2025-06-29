# Документация по `docker/Dockerfile-cu124-uv`

## Общая идея

`Dockerfile-cu124-uv` — это **универсальный Docker-рецепт**, который позволяет собирать как *разработческое* (dev), так и *продакшен* (prod) окружения из одного файла. Он решает сразу несколько задач:

1. **Повторяемость окружения**: сборка выполняется строго по `uv.lock`, что гарантирует идентичные версии библиотек.
2. **Гибкость**: CUDA-версия, вариант базового образа (`runtime | cudnn | base`) и версия CMake передаются через `--build-arg`, поэтому образ можно быстро адаптировать под разные GPU-серверы.
3. **Минимизация лишнего веса**: продакшен-слой собирается без исходников проекта и без dev-зависимостей, а wheel-файл `flash-attn` переиспользуется.
4. **Единая точка поддержки**: вместо двух Dockerfile поддерживаем один, уменьшая трудозатраты.

## Для чего сделан

В проекте используется пакетный менеджер **`uv`** (аналог `pip-tools`/`poetry`) и необязательная зависимость **`flash-attn`**, которую зачастую нужно собирать из исходников под конкретную версию CUDA/torch. Задача Dockerfile:

* предоставить разработчикам готовую среду с установленными всеми пакетами, включая кастомный wheel `flash-attn`;
* дать продакшен-образ с минимальным набором зависимостей (без исходников, тестовых фреймворков и т. д.);
* избавить CI/CD от необходимости пересобирать `flash-attn` при каждой сборке: wheel компилируется один раз на промежуточном слое и кэшируется.

## Структура слоёв

| Слой | База | Назначение |
|------|------|------------|
| **0. `base`** | `nvidia/cuda:${CUDA_VERSION}-${CUDA_VARIANT}-devel-ubuntu${UBUNTU_VERSION}` | Устанавливаются системные зависимости (gcc, git, Python, …) и статический бинарь `uv`. |
| **2.a `deps-dev`** | `base` | `uv sync --locked` по корневому `uv.lock` + копия исходников проекта. |
| **2.b `deps-prod`** | `base` | Минимальное окружение по `prod/uv.lock`, без исходников. |
| **3 `wheel-dev`** | `deps-dev` | Сборка колеса `flash-attn` в dev-контексте. |
| **3.b `wheel-prod`** | `deps-prod` | Аналогичная сборка, но против minimal-env; wheel идентичен и может быть заимствован из кэша. |
| **4.a `dev`** | `deps-dev` + wheel | Финальный dev-образ: весь исходный код + готовый wheel. |
| **4.b `prod`** | `base` + wheel | Финальный prod-образ: только скомпилированные зависимости и wheel, без исходников. |

### Почему именно так

* **Раздельные слои `deps-*`**.  Меняем `pyproject.toml` — пересчитываются лишь зависимости, а не заново всё.
* **Сборка wheel отдельным RUN**.  Колесо `flash-attn` сильно зависит от CUDA: его кэшируется Docker и можно смонтировать в любой следующий слой.
* **`ENV UV_NO_BUILD_ISOLATION=1`**.  Ускоряет сборку `flash-attn`, заставляя использовать уже установленный `torch` вместо повторной сборки в изоляции.
* **Переиспользование wheel в финальных слоях** снижает вес образа и экономит время на CI.

## Аргументы сборки

| ARG | Значение по умолчанию | Описание |
|-----|-----------------------|----------|
| `CUDA_VERSION`  | `12.4.1` | Версия CUDA. При смене нужно убедиться, что существует соответствующий базовый образ NVIDIA. |
| `CUDA_VARIANT`  | `cudnn`  | Вариант базового образа (`runtime`, `cudnn`, `base`). |
| `UBUNTU_VERSION`| `22.04`  | Версия Ubuntu в базе. |
| `CMAKE_VERSION` | `3.26.1` | Версия CMake для сборки `flash-attn`. |

## Базовая команда сборки со всеми аргументами

Ниже приведён пример *полной* команды `docker build`, в которой явно указаны все поддерживаемые `ARG`-параметры со значениями по умолчанию. При необходимости вы можете переопределить любой из них.

```bash
# сборка dev-образа с параметрами по умолчанию

docker build \
  -f docker/Dockerfile-cu124-uv \
  --build-arg CUDA_VERSION=12.4.1 \
  --build-arg CUDA_VARIANT=cudnn \
  --build-arg UBUNTU_VERSION=22.04 \
  --build-arg CMAKE_VERSION=3.26.1 \
  --target dev \
  -t myproj:dev-cu124 .
```

> Замените `--target dev` на `--target prod`, если нужен продакшен-образ. Все остальные `ARG` остаются теми же, либо задайте собственные значения.

## Примеры сборки

### 1. Dev-образ под CUDA 12.4 + cuDNN
```bash
# соберёт dev-слой и финальный dev-образ
docker build \
  -f docker/Dockerfile-cu124-uv \
  --target dev \
  -t myproj:dev-cu124 \
  --build-arg CUDA_VERSION=12.4.1 \
  --build-arg CUDA_VARIANT=cudnn \
  .
```

### 2. Prod-образ под CUDA 12.4 без cuDNN
```bash
docker build \
  -f docker/Dockerfile-cu124-uv \
  --target prod \
  -t myproj:prod-cu124-runtime \
  --build-arg CUDA_VARIANT=runtime \
  .
```

### 3. Образ под CUDA 12.8 (cu128)
```bash
docker build \
  -f docker/Dockerfile-cu124-uv \
  --target dev \
  -t myproj:dev-cu128 \
  --build-arg CUDA_VERSION=12.8.0 \
  --build-arg CMAKE_VERSION=3.22.6 \
  .
```
> **Важно:** убедитесь, что `flash-attn` выпускает wheels/исходники под требуемую версию CUDA; при необходимости воспользуйтесь веткой с поддержкой CUDA 12.8.

## Использование контейнера

### Dev-режим
```bash
# Запуск с доступом к GPU и монтированием исходников
 docker run --gpus=all -it \
   -v $(pwd):/workspace \
   myproj:dev-cu124
# внутри контейнера будут ваши источники, активированное venv и все dev-зависимости
```

### Prod-режим
```bash
# Запуск приложения (пример)
 docker run --gpus=all -it --rm myproj:prod-cu124-runtime \
   uv run python -m my_service
```
*Исходников внутри нет — только скомпилированные зависимости, wheel `flash-attn` и ваши пакет(ы) из `prod/pyproject.toml`.*

## Проверка и публикация wheel FlashAttention

После слоёв `wheel-dev` / `wheel-prod` внутри образа появляется собранный wheel `flash_attn`. Ниже приведены быстрые приёмы, как убедиться, что он рабочий, и при необходимости выгрузить его для повторного использования.

### Smoke-тест
```bash
docker run --gpus all --rm -it myproj:dev-cu124 \
  python - <<'PY'
import torch, flash_attn
print('CUDA:', torch.version.cuda)
print('Flash-Attn:', flash_attn.__version__)
PY
```

### Извлечь wheel на хост
```bash
id=$(docker create myproj:dev-cu124)
docker cp "$id":/wheelhouse/flash_attn*.whl .
docker rm "$id"
```
Колесо можно прикрепить к релизу GitHub и затем ссылаться на него через прямой URL в `[project.optional-dependencies.flash]` (см. docs/development_process.md).

### Отладка ошибок сборки
Если шаг сборки `flash-attn` завершается ошибкой, запустите интерактивную оболочку на предыдущем слое:
```bash
docker build -f docker/Dockerfile-cu124-uv --target wheel-dev --progress=plain .
# взять ID образа шага перед ошибкой
docker run -it <imageID> bash
```
Полезные советы приведены в статье Docker «How to debug build failures».

## FAQ

**Q: Зачем два отдельных `uv.lock` (корневой и `prod/uv.lock`)?**
A: Prod-lock исключает dev-библиотеки (`pytest`, `black` и т. д.) и тем самым уменьшает размер образа и время сборки.

**Q: Можно ли отключить сборку `flash-attn`?**
A: Да. Удалите шаги `wheel-*` и строки `COPY --from=wheel-*`, либо задайте `--build-arg FLASH_ATTN_VERSION=none` и обрабатывайте условие в Dockerfile.

**Q: Почему не multistage-ONBUILD?**
A: Нам важно максимальное переиспользование кеша на разных CI-вакансиях. Явное указание `--target` даёт больше контроля.

## Lock-файлы под разные версии CUDA

`uv` создаёт **один** `uv.lock`, привязанный к текущему набору extras. Если вам нужны образы под несколько версий CUDA (например 12.4 и 12.8), удобно держать *отдельные* lock-файлы — это исключит лишние пересчёты зависимостей и ускорит кэширование Docker-слоёв.

### Как сгенерировать

```bash
# CUDA 12.4
uv lock --extra cu124 -o uv-cu124.lock

# CUDA 12.8
uv lock --extra cu128 -o uv-cu128.lock
```

Имена файлов произвольны; удобно придерживаться конвенции `uv-<backend>.lock`.

> ⚠️ Убедитесь, что `pyproject.toml` (или `prod/pyproject.toml`) *соответствует* тому extra, под который рассчитан lock-файл, иначе `uv sync --locked` завершится ошибкой.

### Использование в Docker

В Dockerfile ожидается файл `uv.lock` рядом с `pyproject.toml`. Чтобы указать нужный lock-файл, достаточно скопировать его под этим именем перед вызовом `uv sync`.

```bash
# Пример для CUDA 12.4
docker build -f docker/Dockerfile-cu124-uv \
  --target dev \
  -t myproj:dev-cu124 \
  --build-arg TORCH_BACKEND=cu124 \
  --build-arg LOCK_FILE=uv-cu124.lock \   # см. ниже
  .
```

В самом `Dockerfile` должен быть аргумент и COPY:

```dockerfile
ARG LOCK_FILE=uv.lock   # значение по умолчанию
COPY ${LOCK_FILE} uv.lock
COPY pyproject.toml ./
RUN uv sync --locked
```

### Генерация prod-lock

Для продакшен-окружения процесс аналогичен, но указывайте каталог `prod/`:

```bash
uv lock --project prod --extra cu124 -o prod/uv-cu124.lock
```

Затем в Docker-сборке копируйте соответствующий файл вместо стандартного `prod/uv.lock`.

Такой подход гарантирует:

1. Репродуктивные сборки — зависимости фиксируются раз и навсегда.
2. Минимум «шумных» пересборок Docker-слоёв при смене версии CUDA.
3. Прозрачность в CI/CD: легко понять, под какой backend собран образ.

---

> Последнее обновление: $(date +"%Y-%m-%d")