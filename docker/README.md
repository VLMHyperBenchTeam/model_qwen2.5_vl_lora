Ниже «каркас» много-стадийного Dockerfile, который

• использует uv как единственный менеджер Python,
• устанавливает все зависимости из `uv.lock`, кроме `flash-attn`,
• собирает wheel `flash_attn` внутри билд-стадии,
• кладёт готовый wheel в artefacts (или во вторую стадию),
• а в прод-стадии подключает wheel явной ссылкой (или локальным файлом).

Идея: объявляем `flash-attn` как optional-extra `flash` (что мы уже записали в `pyproject.toml`). Тогда `uv sync` без `--extra flash` ставит всё, кроме самого `flash-attn`.

```dockerfile
###############################################################################
# 1)  Базовый образ и общие тулзы
###############################################################################
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential git git-lfs \
        python3 python3-distutils wget curl pkg-config \
        libsndfile1 ninja-build && \
    rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN git lfs install

# ---- ставим статический бинарь uv (без Python-зависимостей) ----
ENV UV_VERSION=0.7.16
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --version ${UV_VERSION}
ENV PATH=/root/.local/bin:$PATH

###############################################################################
# 2)  Стадия «deps»: ставим всё из uv.lock, КРОМЕ flash-attn
###############################################################################
FROM base AS deps
WORKDIR /build

# Копируем только описания проекта (чтоб не перетягивать исходники дважды)
COPY pyproject.toml uv.lock ./

# Синхронизируем окружение без optional-extra «flash»
RUN uv sync --locked --no-dev --no-default-groups

###############################################################################
# 3)  Стадия «wheel»: собираем flash-attn
###############################################################################
FROM deps AS wheel
WORKDIR /build

# Ставим CMake, если нужен конкретный >=3.25
ARG CMAKE_VERSION=3.26.1
RUN curl -LO https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-x86_64.sh \
    && chmod +x cmake-$CMAKE_VERSION-linux-x86_64.sh \
    && ./cmake-$CMAKE_VERSION-linux-x86_64.sh --skip-license --prefix=/usr/local \
    && rm cmake-$CMAKE_VERSION-linux-x86_64.sh

# Собираем wheel в /wheelhouse, используя уже установленный torch
ENV UV_NO_BUILD_ISOLATION=1       # пусть юзает текущее окружение
RUN python -m pip wheel "flash-attn==2.6.1"           \
      --no-deps --no-binary ':all:' --wheel-dir /wheelhouse

###############################################################################
# 4)  Прод-стадия: лёгкий рантайм + наш wheel
###############################################################################
FROM base AS prod
WORKDIR /workspace

# (а) Ставим зависимости из lock-файла, опять без extra=flash
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-default-groups

# (б) Копируем собранный wheel и добавляем его в sources
COPY --from=wheel /wheelhouse/flash_attn*.whl /tmp/

# Добавляем запись в tool.uv.sources  ─ делаем в один шаг sed-ом
RUN WHEEL=$(ls /tmp/flash_attn*.whl | head -n1) && \
    uv add "flash_attn @ file://${WHEEL}"            \
           --package model-qwen2_5-vl                \
           --optional flash --raw --no-sync

# (в) До-устанавливаем сам wheel (+зависимость optional-extra ‘flash’)
RUN uv sync --locked --extra flash --no-dev

CMD ["/bin/bash"]
```

Как это работает
────────────────
1. `deps`-стадия воспроизводит окружение из `uv.lock` быстрее, чем привычные
   `pip install`, — `flash-attn` туда не попадает.
2. В `wheel` мы используем уже установленный `torch==2.4.0` (+ CUDA 12.4) и без-изоляционную сборку (`UV_NO_BUILD_ISOLATION=1`) — FlashAttention видит PyTorch и собирается за 3-5 мин вместо 40-60.
3. Готовый wheel переносится во `prod`.
   Через `uv add ... --raw` мы дописываем ссылку в `[tool.uv.sources]`
   и отдельно вызываем `uv sync --extra flash`, чтобы поставить wheel
   (теперь уже без пересборки).
4. Если wheel публикуется на GitHub/Minio — записываем не `file://…`,
   а прямую `https://…`-ссылку; `uv` умеет скачивать.

После выхода новой версии Torch/CUDA
─────────────────────────────────────
• запустите тот же Docker-build — в стадии `wheel` соберётся wheel для новой связки;
• залейте `flash_attn-*.whl` в свой release и поменяйте URL в шаге `uv add`.

Проверка
────────
```bash
docker build -f docker/Dockerfile-cu124-uv -t ghcr.io/vlmhyperbenchteam/qwen2.5-vl:ubuntu22.04-cu124-torch2.4.0_uv_v0.1.0 .

docker build -f docker/Dockerfile-cu124-uv-dev -t qwen2.5-vl:ubuntu22.04-cu124-torch2.4.0_uv_dev .

docker run --gpus all -it ghcr.io/vlmhyperbenchteam/qwen2.5-vl:ubuntu22.04-cu124-torch2.4.0_uv_v0.1.0 python - <<'PY'
import torch, flash_attn
print("torch:", torch.__version__)
print("flash_attn:", flash_attn.__version__)
PY
```

Если выводится версия `flash_attn 2.6.1` — всё собрано и подхватилось из
wheel, а не компилировалось повторно.