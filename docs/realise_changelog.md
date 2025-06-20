# Changelog релиза

## Дата: $(date)

## Изменения в пакетах:

### bench_utils
- Статус: 🔄 Пропущен
- [x] Анализ изменений - нет изменений в коде
- [-] Обновление версии - не требуется
- [-] Создание коммита - не требуется
- [-] Создание тега - не требуется

### model_interface  
- Статус: ✅ Завершен
- [x] Анализ изменений - рефакторинг API параметров
- [x] Обновление версии - 0.1.2.dev1 -> 0.1.2.dev2
- [x] Создание коммита - refactor: improve API parameters naming
- [x] Создание тега - v0.1.2.dev2

### model_qwen2.5-vl
- Статус: ✅ Завершен
- [x] Анализ изменений - рефакторинг API и документации
- [x] Обновление версии - 0.1.2.dev2 -> 0.1.2.dev3
- [x] Создание коммита - refactor: comprehensive API and documentation improvements
- [x] Создание тега - v0.1.2.dev3

### print_utils
- Статус: 🔄 Пропущен
- [x] Анализ изменений - нет изменений в коде
- [-] Обновление версии - не требуется
- [-] Создание коммита - не требуется
- [-] Создание тега - не требуется

## Изменения в основном проекте:
- Статус: ✅ Завершен
- [x] Анализ изменений - улучшения workspace структуры и тестирования
- [x] Обновление версии - 0.0.1 -> 0.0.2
- [x] Создание коммита - feat: improve workspace structure and testing capabilities
- [x] Создание тега - v0.0.2

## Итоги релиза:

### Выполненные релизы:
- **bench_utils**: без изменений (пропущен)
- **model_interface**: v0.1.2.dev2 - рефакторинг API параметров ✅
- **model_qwen2.5-vl**: v0.1.2.dev3 - рефакторинг API и документации ✅  
- **print_utils**: без изменений (пропущен)
- **основной проект**: v0.0.2 - улучшения workspace структуры ✅

### Полные commit messages:

#### model_interface (v0.1.2.dev2):
```
refactor: improve API parameters naming

- Rename 'question' parameter to 'prompt' in predict_on_image and predict_on_images methods
- Fix documentation typo in method descriptions
- Update version to 0.1.2.dev2

BREAKING CHANGE: Parameter 'question' renamed to 'prompt' in ModelInterface methods
```

#### model_qwen2.5-vl (v0.1.2.dev3):
```
refactor: comprehensive API and documentation improvements

- Add comprehensive Google-style docstrings for all methods
- Add type annotations throughout the codebase
- Refactor architecture with new helper methods (_generate_answer, get_messages)
- Improve library version compatibility handling
- Update all run scripts to use new API
- Update version to 0.1.2.dev3

BREAKING CHANGE: Parameter 'question' renamed to 'prompt' in all predict methods
```

#### основной проект (v0.0.2):
```
feat: improve workspace structure and testing capabilities

- Add print-utils package to workspace dependencies
- Enhance test_model.py with comprehensive testing improvements
- Update check and structured output scripts
- Add config_test_model.json configuration
- Update documentation and release templates
- Update version to 0.0.2
```

## Следующие шаги:
Нужно выполнить push commit и push tag для каждого измененного пакета и проекта. 