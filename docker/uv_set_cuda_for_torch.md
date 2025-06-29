Да. `uv` позволяет явно задать требуемый вариант PyTorch-wheels (CPU либо конкретную версию CUDA) двумя способами.

1. Быстрый вариант через интерфейс `uv pip`
   ```
   # Установка Torch/Torchvision под CUDA 12.4
   uv pip install torch torchvision --torch-backend=cu124
   ```
   или через переменную окружения
   ```bash
   UV_TORCH_BACKEND=cu124 uv pip install torch torchvision
   ```
   Вместо `cu124` можно указывать: `cpu`, `cu121`, `cu126`, `cu128`, `rocm63` и т. д.
   Есть и режим автодетекции драйвера: `--torch-backend=auto` (при его выборе uv попытается подобрать подходящий индекс сам) [документация](https://docs.astral.sh/uv/guides/integration/pytorch/#the-uv-pip-interface).

2. Конфигурация в `pyproject.toml` (для `uv lock / uv sync`)

   ```toml
   [project]
   dependencies = ["torch>=2.4.0", "torchvision>=0.19.0"]

   [tool.uv.sources]
   torch       = [{ index = "pytorch-cu124", marker = "sys_platform == 'linux'" }]
   torchvision = [{ index = "pytorch-cu124", marker = "sys_platform == 'linux'" }]

   [[tool.uv.index]]
   name     = "pytorch-cu124"
   url      = "https://download.pytorch.org/whl/cu124"
   explicit = true
   ```
   Для других платформ или CPU-сборок просто добавьте ещё один источник (например, PyPI) с соответствующими `marker`.

При необходимости можно описать несколько вариантов (CPU, `cu128`, …) как extras и переключать их командой `uv sync --extra cu128`.

Таким образом, указать конкретную версию CUDA в `uv` можно либо опцией `--torch-backend`, либо фиксацией нужного индекса в файле конфигурации проекта.

Нет. На сегодня (uv ≥ 0.5.3) флаг `--torch-backend=…` реализован только в подкоманде `uv pip`.
`uv add` такого параметра пока не принимает. Для расширения этой функциональности открыт запрос [Issue #12994](https://github.com/astral-sh/uv/issues/12994), но он ещё не влит.

Чтобы добавить PyTorch под нужную версию CUDA через `uv add`, придётся задать собственный индекс в `pyproject.toml` и воспользоваться обычной командой `uv add`:

```toml
# pyproject.toml
[[tool.uv.index]]
name     = "pytorch-cu124"
url      = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch       = [{ index = "pytorch-cu124" }]
torchvision = [{ index = "pytorch-cu124" }]
```

затем:

```bash
uv add torch torchvision         # или любые другие зависимости
uv lock
uv sync
```

Либо обойтись без `uv add`, используя:

```bash
uv pip install torch torchvision --torch-backend=cu124
```

или

```bash
UV_TORCH_BACKEND=cu124 uv pip install torch torchvision
```

Документация о механизме `--torch-backend` приведена в руководстве «Using uv with PyTorch» [docs.astral.sh](https://docs.astral.sh/uv/guides/integration/pytorch/#the-uv-pip-interface).

# Как поступить?

Да, так можно и даже удобнее:

1. Внутренний пакет `model-qwen2_5-vl`
 • содержит лишь «абстрактную» зависимость `torch>=2.7` (без индексов, без суффикса `+cu…`);
 • не указывает `tool.uv.sources` для PyTorch-семейства.
 Таким образом пакет остаётся универсальным: его можно ставить в CPU-среду, в CUDA 11.8, 12.4, ROCm и т. д.

2. На уровне **проекта / образа** (корневой `pyproject.toml`)
 • прописываете нужный бекенд через `tool.uv.sources` или флаг `--torch-backend`:

```toml
# корневой pyproject.toml
[[tool.uv.index]]
name = "pytorch-cu124"
url  = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch        = [{ index = "pytorch-cu124" }]
torchvision  = [{ index = "pytorch-cu124" }]
pytorch-triton = [{ index = "pytorch-cu124" }]
```

или короче в Dockerfile:

```dockerfile
ENV UV_TORCH_BACKEND=cu124   # 0.7+  умеет auto, cu126, cpu, …
RUN uv sync --locked
```

3. При сборке разных образов вы просто меняете переменную/аргумент:

```dockerfile
# build-arg приходит из CI («cu121», «cu128», «cpu» …)
ARG TORCH_BACKEND
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}
RUN uv sync --locked
```

Так одна и та же версия вашего пакета работает с любым CUDA, а
конкретный проект (или Docker-образ) «заливает» именно тот билд PyTorch,
который совместим с базовым образом `nvidia/cuda:…`.

Что важно знать

• `tool.uv.sources` из корня **переопределяет** вложенные пакеты-участники
  workspace, поэтому отсутствие фиксированного backend-индекса внутри
  самого `model-qwen2_5-vl` не мешает вам задать его сверху.
• Lock-файл будет включать именно те колёса (`+cu124` и т. п.),
  которые выбраны на момент `uv lock`.
  Если нужен второй backend — заведите отдельный lock-файл
  (`uv.lock-cu124`, `uv.lock-cu128`) или пересчитывайте lock
  в CI для каждого варианта.

Итого: пакет остаётся универсальным, а выбор CUDA-варианта полностью
контролируется на уровне проекта/контейнера (где и так известна
версия CUDA).

# Итоговый рекомендованнный вариант

Можно и так, но в большинстве случаев это «тяжёлая артиллерия».
Сравним оба подхода.

────────────────────────────────────────
1. Несколько workspace-директорий
────────────────────────────────────────
• Для каждого back-end (cpu, cu121, cu124 …) заводите клон корня
  `workspace-cu124/pyproject.toml`, где меняется только секция

```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url  = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch       = [{ index = "pytorch-cu124" }]
torchvision = [{ index = "pytorch-cu124" }]
```

• Лок-файл получается свой (`uv.lock` в каталоге-клоне).
• В CI собираете образ:

```bash
cd workspace-cu124
uv sync --locked           # либо uv run …
```

Плюсы
– полная изоляция: разные lock-файлы → 100 % репродуцируемость.
Минусы
– дублирование кода (нужно следить, чтобы версии пакетов, README и т.д. совпадали);
– прибавится N virtualenv-ов, N Dockerfile-ов, N pipelines.

────────────────────────────────────────
2. Один workspace + «переключатель» back-енда
────────────────────────────────────────
Рекомендованный вариант из официальной документации
[Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).

a) В пакете не фиксируем индекс PyTorch вовсе
   (оставляем просто `torch>=2.7`).

b) В корневом `pyproject.toml` прописываем backend-варианты через extras:

```toml
[project.optional-dependencies]
cpu   = ["torch>=2.7", "torchvision>=0.19"]
cu124 = ["torch>=2.7", "torchvision>=0.19"]

[[tool.uv.index]]
name = "pytorch-cpu"
url  = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu124"
url  = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu",   extra = "cpu" },
  { index = "pytorch-cu124", extra = "cu124" },
]
torchvision = [
  { index = "pytorch-cpu",   extra = "cpu" },
  { index = "pytorch-cu124", extra = "cu124" },
]
```

c) При сборке образа выбираем extra:

```dockerfile
# build-arg приходит из CI: cpu | cu124 | cu128 …
ARG TORCH_BACKEND=cu124
RUN uv sync --locked --extra ${TORCH_BACKEND}
```

или на локальной машине:

```bash
uv sync --extra cpu
uv sync --extra cu124
```

Плюсы
– один исходник, один workspace, один CI-скрипт;
– легко добавить новый backend (cu128) единожды в конфигурацию;
– пользователи могут установить нужный вариант:
  `pip install model-qwen2_5-vl[cu124] --extra-index-url …`.

Минусы
– lock-файл придётся пересчитывать для каждого backend
  (но это можно автоматизировать: `uv lock --extra cu124 -o uv-cu124.lock`).

────────────────────────────────────────
Что выбрать?
────────────────────────────────────────
• Если у вас 2–3 варианта CUDA и одна команда разработчиков —
  подход с extras проще поддерживать.
• Если нужна жёсткая изоляция зависимостей (регуляторика,
  долгоживущие LTS-ветки) — отдельные workspaces/репозитории с
  собственными lock-файлами могут быть оправданы.

Большинство open-source библиотек (PyTorch Lightning, BitsAndBytes и т.д.)
идут по пути № 2: «один пакет + extras», а образы под конкретный CUDA
строятся на уровне проекта или CI/CD.

# Параметризация образа от nvidia

Да, «тег» базового образа можно сделать переменным так же, как мы уже делаем с `TORCH_BACKEND`. В Dockerfile это решается обычными build-args, объявленными ПЕРЕД первой директивой `FROM`.
Схема:

```dockerfile
# syntax=docker/dockerfile:1.4          # (не обязательно, но полезно)

###############################################################################
# 1. Параметризуем выбор базового CUDA-образа
###############################################################################
ARG CUDA_VERSION=12.4.1                 # 12.6.0, 11.8.0 … — что угодно
ARG VARIANT=cudnn                       # или runtime / base
ARG UBUNTU_VERSION=22.04                # можно оставить 20.04 и т.д.

# Формируем тег. Docker подставит ARG-ы при разборе строки:
FROM nvidia/cuda:${CUDA_VERSION}-${VARIANT}-devel-ubuntu${UBUNTU_VERSION} AS base
```

Дальше Dockerfile остаётся тем же. На сборку передаём нужные аргументы:

```bash
# CUDA 12.4  + cudnn + Ubuntu 22.04 (текущий вариант)
docker build \
  --build-arg CUDA_VERSION=12.4.1 \
  --build-arg VARIANT=cudnn \
  --build-arg UBUNTU_VERSION=22.04 \
  --build-arg TORCH_BACKEND=cu124 \
  -t qwen2.5-vl:cu124 .

# CUDA 11.8 runtime-вариант, чтобы собрать «cpu»-PyTorch
docker build \
  --build-arg CUDA_VERSION=11.8.0 \
  --build-arg VARIANT=runtime \
  --build-arg UBUNTU_VERSION=22.04 \
  --build-arg TORCH_BACKEND=cpu \
  -t qwen2.5-vl:cpu .
```

Под капотом:

1. `ARG`-и, объявленные ДО `FROM`, видны именно в теге базового образа.
   Если объявить их ниже, Docker не сможет подставить переменную в `FROM`.

2. Можно сохранить один Dockerfile и управлять комбинациями CUDA/Ubuntu/cudnn-вариантов только через `docker build --build-arg …`.

3. Нужный backend PyTorch мы уже переключаем флагом `TORCH_BACKEND` (`uv sync --extra …`), так что обе системы параметров не конфликтуют.

Если хотите полностью автоматизировать матчинг
«TORCH_BACKEND → образ», вынесите маппинг в CI-скрипт:

```bash
case "$TORCH_BACKEND" in
  cpu)   CUDA_VERSION=11.8.0; VARIANT=runtime ;;
  cu124) CUDA_VERSION=12.4.1; VARIANT=cudnn   ;;
  cu128) CUDA_VERSION=12.8.0; VARIANT=cudnn   ;;
esac
docker build --build-arg CUDA_VERSION --build-arg VARIANT \
             --build-arg UBUNTU_VERSION=22.04 \
             --build-arg TORCH_BACKEND \
             -t "qwen2.5-vl:${TORCH_BACKEND}" .
```

Таким образом одним Dockerfile’ом покрываются любые комбинации версий CUDA и наборов библиотек NVIDIA.

# Единый Dockerfile для dev и prod

Да, оба Dockerfile отличаются лишь несколькими деталями, поэтому их удобно свести к одному многостадийному Dockerfile с двумя «финальными» целями — `dev` и `prod`. Переключение делается обычным флагом `--target` (или `--build-arg MODE=…`, если хочется именно аргумент). Общая схема:

1. Общие слои
   • `base` – выбор образа CUDA через `ARG CUDA_VERSION/VARIANT/UBUNTU_VERSION`.
   • `deps` – установка зависимостей из `uv.lock`.
   • `wheel` – сборка `flash-attn`.

2. Финальные слои
   • `dev` наследуется от `deps`, копирует исходники workspace и сразу готово к разработке.
   • `prod` наследуется от `base` (или даже `deps`), копирует только «чистые» файлы (например, `prod/pyproject.toml`), подтягивает lock, кладёт собранный wheel.

Пример каркаса (упрощённо):

```dockerfile
#################### 0. Параметры ################################################
ARG CUDA_VERSION=12.4.1
ARG VARIANT=cudnn
ARG UBUNTU_VERSION=22.04
ARG CMAKE_VERSION=3.26.1

#################### 1. Базовый слой ############################################
FROM nvidia/cuda:${CUDA_VERSION}-${VARIANT}-devel-ubuntu${UBUNTU_VERSION} AS base
# … apt/uv одинаковые …

#################### 2. deps: uv sync (общий) ###################################
FROM base AS deps
WORKDIR /build
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY release_tool ./release_tool           # если нужен в рантайме
RUN uv sync --locked

#################### 3. wheel: flash-attn #######################################
FROM deps AS wheel
WORKDIR /build
# CMake, UV_NO_BUILD_ISOLATION – как раньше
RUN uv pip install --no-cache-dir pip packaging psutil pybind11 && \
    mkdir -p /wheelhouse && \
    uv run python -m pip wheel flash-attn==2.6.1 \
        --no-deps --no-build-isolation -w /wheelhouse

#################### 4a. Финальная среда «dev» ###################################
FROM deps AS dev
WORKDIR /workspace
# копируем ВСЮ рабочую область для интерактива
COPY --from=wheel /wheelhouse/flash_attn*.whl /tmp/
RUN uv pip install --no-deps /tmp/flash_attn*.whl
ENV PATH="/workspace/.venv/bin:${PATH}"
CMD ["/bin/bash"]

#################### 4b. Финальная среда «prod» ##################################
FROM base AS prod
WORKDIR /workspace
# minimal copy (можно заменить на prod/pyproject.toml + prod/uv.lock)
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN uv sync --locked
COPY --from=wheel /wheelhouse/flash_attn*.whl /tmp/
RUN uv pip install --no-deps /tmp/flash_attn*.whl
ENV PATH="/workspace/.venv/bin:${PATH}"
CMD ["/bin/bash"]
```

Сборка:

```bash
# образ для разработки
docker build --target dev  -t qwen2.5-vl:dev  .

# минимальный прод-рантайм
docker build --target prod -t qwen2.5-vl:prod .
```

При необходимости добавить «матрицу» разных CUDA-версий + back-ендов Torch:

```bash
docker build \
  --build-arg CUDA_VERSION=12.6.0 \
  --build-arg VARIANT=cudnn \
  --build-arg TORCH_BACKEND=cu126 \
  --target prod \
  -t qwen2.5-vl:cu126 .
```

Такой единый Dockerfile упрощает поддержку: все общие слои кешируются, различия сведены к минимуму и управляются параметрами сборки.