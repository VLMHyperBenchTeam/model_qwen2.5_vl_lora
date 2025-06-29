#!/usr/bin/env python3
"""
build_docker.py — утилита для сборки Docker-образов по описанию в JSON-конфиге.

Пример запуска:
    ./build_docker.py docker/build_config.json
    ./build_docker.py docker/build_config.json --dry-run    # только вывести команды

Структура конфигурационного файла (пример):
{
  "defaults": {
    "dockerfile": "docker/Dockerfile-uv",
    "context": "."
  },
  "builds": [
    {
      "target": "prod",
      "tag": "myproject:prod-cu124",
      "args": {
        "CUDA_VERSION": "12.4.1",
        "CUDA_VARIANT": "cudnn",
        "UBUNTU_VERSION": "22.04",
        "CMAKE_VERSION": "3.26.1"
      },
      "platforms": "linux/amd64",
      "push": false,
      "load": true
    }
  ]
}

Поле `builds` может быть либо списком описаний, либо одним объектом.

Поддерживаемые ключи одного build-описания:
  target        — имя таргета в Dockerfile (dev | prod | …)
  tag           — итоговый тег образа
  args          — словарь произвольных `--build-arg`
  dockerfile    — путь к Dockerfile (переопределяет defaults)
  context       — контекст сборки
  platforms     — `--platform` (multi-arch)
  push / load   — публикация или загрузка образа
  cache_from / cache_to — настройки кеша
  base_image    — полный образ для директивы `FROM`
  uv_version    — версия uv
  cmake_version — версия CMake
  mode          — `dev` / `prod` (если `target` не указан)
  toml_path / lock_path — пути к `pyproject.toml` и lock-файлу

Для работы требуется установленный docker buildx.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


def _build_command(spec: Dict, defaults: Dict) -> List[str]:
    """Сформировать команду `docker buildx build` для одного описания."""
    dockerfile = spec.get("dockerfile") or defaults.get("dockerfile", "docker/Dockerfile-uv")
    context = spec.get("context") or defaults.get("context", ".")

    cmd: List[str] = [
        "docker",
        "buildx",
        "build",
        "-f",
        dockerfile,
    ]

    if platforms := spec.get("platforms"):
        cmd += ["--platform", platforms]

    if spec.get("push"):
        cmd.append("--push")
    elif spec.get("load"):
        cmd.append("--load")

    if spec.get("no_cache"):
        cmd.append("--no-cache")

    if cache_from := spec.get("cache_from"):
        cmd += ["--cache-from", cache_from]
    if cache_to := spec.get("cache_to"):
        cmd += ["--cache-to", cache_to]

    # Определяем таргет: явный приоритет, затем mode
    target = spec.get("target")
    if not target and (mode := spec.get("mode")):
        target = mode  # предполагаем, что mode совпадает с названием таргета (dev/prod)
    if target:
        cmd += ["--target", target]

    # Собираем build-args: объединяем spec.args + дополнительные псевдонимы
    build_args = dict(spec.get("args", {}))

    # Псевдонимы верхнего уровня → build-args
    alias_map = {
        "base_image": "BASE_IMAGE",
        "uv_version": "UV_VERSION",
        "cmake_version": "CMAKE_VERSION",
        "toml_path": "TOML_PATH",
        "lock_path": "LOCK_PATH",
        "torch_index": "TORCH_INDEX",
    }
    for alias, arg_name in alias_map.items():
        if alias in spec:
            build_args[arg_name] = spec[alias]

    # Проверяем обязательные аргументы
    required = ["BASE_IMAGE", "TOML_PATH", "LOCK_PATH", "TORCH_INDEX"]
    missing = [k for k in required if k not in build_args]
    if missing:
        raise SystemExit(f"В конфигурации отсутствуют обязательные поля: {', '.join(missing).lower()}")

    # --build-arg ...
    for k, v in build_args.items():
        cmd += ["--build-arg", f"{k}={v}"]

    if tag := spec.get("tag"):
        cmd += ["-t", tag]

    cmd.append(context)
    return cmd


def _run(cmd: Sequence[str], dry_run: bool = False) -> None:
    formatted = " ".join(shlex.quote(c) for c in cmd)
    print(formatted)
    if not dry_run:
        subprocess.check_call(cmd)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Сборка Docker-образов на основе JSON-конфига")
    parser.add_argument("config", type=Path, help="Путь к JSON-конфигу")
    parser.add_argument("--dry-run", action="store_true", help="Только вывести команды, не исполнять")
    args = parser.parse_args(argv)

    try:
        data = json.loads(args.config.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"Не удалось прочитать конфиг {args.config}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Если файл описывает список, оборачиваем в единый вид
    if isinstance(data, list):
        builds = data
        defaults: Dict = {}
    elif isinstance(data, dict):
        builds = data.get("builds") or []
        defaults = data.get("defaults", {})
    else:
        print("Конфиг должен быть объектом или списком.", file=sys.stderr)
        sys.exit(1)

    if not builds:
        print("В конфиге нет описаний build'ов.", file=sys.stderr)
        sys.exit(1)

    for spec in builds:
        cmd = _build_command(spec, defaults)
        _run(cmd, dry_run=args.dry_run)


if __name__ == "__main__":  # pragma: no cover
    main()