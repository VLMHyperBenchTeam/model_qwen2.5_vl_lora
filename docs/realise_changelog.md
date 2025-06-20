# Changelog для релиза

## Дата релиза: $(date)

### Изменения в пакетах:

#### bench_utils
- [x] Анализ изменений - добавлена зависимость model-interface
- [x] Обновление версии - 0.1.2.dev1 -> 0.1.2.dev2
- [x] Создание коммита - chore: обновление зависимостей
- [x] Создание тега - v0.1.2.dev2

**Commit message:**
```
chore: bump version to 0.1.2.dev2 and add model-interface dependency

- Update version from 0.1.2.dev1 to 0.1.2.dev2
- Add model-interface as dependency
- Add workspace configuration for model-interface
- Add hatch metadata configuration for direct references
```

#### model_interface  
- [x] Анализ изменений - нет изменений, пропущен
- [-] Обновление версии - не требуется
- [-] Создание коммита - не требуется
- [-] Создание тега - не требуется

#### model_qwen2.5-vl
- [x] Анализ изменений - добавлен flash_attn fallback
- [x] Обновление версии - 0.1.2.dev1 -> 0.1.2.dev2
- [x] Создание коммита - feat: новая функциональность
- [x] Создание тега - v0.1.2.dev2

**Commit message:**
```
feat: add flash_attn fallback mechanism and update dependencies

- Add flash_attn availability check with fallback to eager attention
- Improve model robustness when flash_attn is not available  
- Update version to 0.1.2.dev2
- Add torch, torchvision, transformers, accelerate dependencies
- Update model-interface dependency to workspace reference
- Update uv.lock with new dependency resolution
```

### Изменения в основном проекте:
- [x] Анализ изменений - обновлен test_model.py, добавлены файлы
- [x] Обновление версии в pyproject.toml - 0.0.0 -> 0.0.1
- [x] Создание коммита - feat: workspace structure и улучшения тестирования
- [x] Создание тега - v0.0.1

**Commit message:**
```
feat: add workspace structure and improve testing

- Add UV workspace configuration with packages structure
- Add release process templates and changelog
- Improve test_model.py with better error handling and configuration
- Add test_config.py for centralized configuration
- Update dependencies in uv.lock
- Version bump to 0.0.1
```

## Статус выполнения:
- [x] Подготовка
- [x] Релиз пакетов
- [x] Релиз основного проекта
- [x] Завершение

## 🎉 РЕЛИЗ ЗАВЕРШЕН УСПЕШНО! 

### Итоговый результат:
- **bench_utils**: v0.1.2.dev2 - обновление зависимостей ✅ **PUSHED**
- **model_interface**: без изменений (пропущен)
- **model_qwen2.5-vl**: v0.1.2.dev2 - новая функциональность flash_attn fallback ✅ **PUSHED**
- **основной проект**: v0.0.1 - workspace структура и улучшения ✅ **PUSHED**

Все коммиты созданы согласно conventional commits стандарту.
Все теги созданы с соответствующими версиями.
Все изменения успешно запушены в соответствующие репозитории.

### 🚀 Push статус:
- **bench_utils**: коммиты и теги запушены в origin/refactoring ✅
- **model_qwen2.5-vl**: коммиты и теги запушены в origin/refactoring ✅  
- **основной проект**: коммиты и теги запушены в origin/uv_workspaces ✅

## 🏆 ПРОЦЕСС РЕЛИЗА ПОЛНОСТЬЮ ЗАВЕРШЕН!
Все пакеты и основной проект успешно обновлены, закоммичены, отегированы и запушены. 