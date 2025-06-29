# Выбор backend-а PyTorch (CUDA)

В проекте предусмотрены два extras:
| Extra | CUDA | Индекс |
|-------|------|--------|
| `cu124` | 12.4 + cuDNN | https://download.pytorch.org/whl/cu124 |
| `cu128` | 12.8 + cuDNN | https://download.pytorch.org/whl/cu128 |

## 1. Локальная установка

```bash
uv sync --extra cu124   # CUDA 12.4
uv sync --extra cu128   # CUDA 12.8
```

Проверка без изменения lock-файла:

```bash
uv sync --check
```

## 2. Генерация lock под конкретный backend

```bash
uv lock --extra cu124   # пересчитать под 12.4
uv lock --extra cu128   # пересчитать под 12.8
```

## 3. Prod-режим

```bash
uv sync --project prod --extra cu124 --frozen
```

## 4. Docker

Dockerfile-uv принимает build-arg `TORCH_BACKEND`:

```bash
docker build -f docker/Dockerfile-uv \
  --target dev \
  --build-arg TORCH_BACKEND=cu128 \
  -t project:dev-cu128 .
```

## 5. Быстрое переключение через переменную окружения

```bash
export UV_TORCH_BACKEND=cu128
uv sync --locked
```
