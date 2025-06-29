# Changelog релиза

## Дата: $(date)

## Изменения в пакетах:

### bench_utils

- Статус: ✅ Завершен
- [x] Анализ изменений - рефакторинг импортов prompt handling
- [x] Обновление версии - 0.1.2.dev2 -> 0.1.2.dev3
- [x] Создание коммита - refactor: migrate prompt handling to dedicated prompt-handler package
- [x] Создание тега - v0.1.2.dev3

### model_interface

- Статус: ✅ Завершен
- [x] Анализ изменений - удаление load_prompt функции (BREAKING CHANGE)
- [x] Обновление версии - 0.1.2.dev2 -> 0.1.2.dev3
- [x] Создание коммита - refactor!: remove load_prompt function and migrate to prompt-handler
- [x] Создание тега - v0.1.2.dev3

### model_qwen2.5-vl

- Статус: 🔄 Пропущен
- [x] Анализ изменений - нет изменений в коде
- [-] Обновление версии - не требуется
- [-] Создание коммита - не требуется
- [-] Создание тега - не требуется

### print_utils

- Статус: ✅ Завершен
- [x] Анализ изменений - обновление метаданных пакета и формата версии
- [x] Обновление версии - 0.0.1dev0 -> 0.0.1dev1
- [x] Создание коммита - chore: update package metadata and version format
- [x] Создание тега - v0.0.1dev1

### prompt_handler

- Статус: 🔄 Пропущен
- [x] Анализ изменений - нет изменений в коде
- [-] Обновление версии - не требуется
- [-] Создание коммита - не требуется
- [-] Создание тега - не требуется

## Изменения в основном проекте:

- Статус: ✅ Завершен
- [x] Анализ изменений - рефакторинг архитектуры импортов prompt-handler
- [x] Обновление версии - 0.0.2 -> 0.0.3
- [x] Создание коммита - refactor: migrate to new prompt-handler architecture
- [x] Создание тега - v0.0.3

## Итоги релиза:

### Выполненные релизы:

- **bench_utils**: v0.1.2.dev3 - рефакторинг импортов prompt handling ✅
- **model_interface**: v0.1.2.dev3 - удаление load_prompt функции (BREAKING CHANGE) ✅
- **model_qwen2.5-vl**: без изменений (пропущен)
- **print_utils**: v0.0.1dev1 - обновление метаданных пакета ✅
- **prompt_handler**: без изменений (пропущен)
- **основной проект**: v0.0.3 - рефакторинг архитектуры prompt-handler ✅

### Полные commit messages:

#### bench_utils (v0.1.2.dev3):

```
refactor: migrate prompt handling to dedicated prompt-handler package

- Migrate load_prompt import from model_interface to prompt_handler
- Add prepare_prompt import from prompt_handler
- Add prompt-handler dependency and workspace source
- Update version to 0.1.2.dev3
```

#### model_interface (v0.1.2.dev3):

```
refactor!: remove load_prompt function and migrate to prompt-handler

- Remove load_prompt function from model_factory.py
- Remove load_prompt import and export from __init__.py
- Add type ignore comment for model import
- Update version to 0.1.2.dev3

BREAKING CHANGE: load_prompt function removed, use prompt-handler package instead
```

#### print_utils (v0.0.1dev1):

```
chore: update package metadata and version format

- Standardize version format from 0.0.1-dev0 to 0.0.1dev1
- Simplify package description
- Reorder metadata fields for consistency
- Add setuptools packages configuration
- Update version to 0.0.1dev1
```

#### основной проект (v0.0.3):

```
refactor: migrate to new prompt-handler architecture

- Update imports in check scripts to use bench_utils.model_utils
- Replace ModelFactory.initialize_model with initialize_model
- Add prepare_prompt usage for better prompt handling
- Add prompt-handler dependency to workspace
- Update release process documentation
- Update version to 0.0.3
```

## 🎉 РЕЛИЗ ЗАВЕРШЕН УСПЕШНО!

### 🚀 Push статус:

- **model_interface**: коммиты и теги запушены в origin/refactoring ✅
- **model_qwen2.5-vl**: коммиты и теги запушены в origin/refactoring ✅
- **основной проект**: коммиты и теги запушены в github/uv_workspaces ✅

## 🏆 ПРОЦЕСС РЕЛИЗА ПОЛНОСТЬЮ ЗАВЕРШЕН!

Все пакеты и основной проект успешно:

- ✅ Проанализированы на предмет изменений
- ✅ Версии обновлены согласно семантическому версионированию
- ✅ Коммиты созданы по conventional commit стандарту
- ✅ Теги проставлены с соответствующими версиями
- ✅ Изменения запушены в соответствующие репозитории

**Итоговые версии релиза:**

- model_interface: v0.1.2.dev2
- model_qwen2.5-vl: v0.1.2.dev3
- основной проект: v0.0.2
