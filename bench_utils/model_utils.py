import logging
from pathlib import Path
from typing import Any, Dict

from model_interface.model_factory import ModelFactory

# Настройка логгера
logger = logging.getLogger(__name__)

def initialize_model(model_config: Dict[str, Any]) -> Any:
    """Инициализирует и возвращает модель согласно конфигурации.

    Args:
        model_config: Словарь конфигурации модели.

    Returns:
        Инициализированный объект модели.

    Raises:
        KeyError: При отсутствии обязательных ключей в конфигурации.
        OSError: При ошибках создания директории кэша.
        ValueError: При ошибке регистрации или создания модели.
    """
    # Проверка обязательных ключей
    required_keys = {
        "model_family", 
        "cache_dir", 
        "package", 
        "module", 
        "model_class",
        "model_name",
        "device_map"
    }
    if missing := required_keys - set(model_config):
        raise KeyError(f"Отсутствуют обязательные ключи: {missing}")

    # Создание директории для кэша
    cache_dir = Path(model_config["cache_dir"])
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Ошибка создания директории кэша {cache_dir}: {str(e)}")
        raise

    # Формирование пути к классу модели
    package = model_config["package"]
    module = model_config["module"]
    model_class = model_config["model_class"]
    model_class_path = f"{package}.{module}:{model_class}"

    # Регистрация модели в фабрике
    model_family = model_config["model_family"]
    ModelFactory.register_model(model_family, model_class_path)

    # Подготовка параметров модели
    model_params = {
        "model_name": model_config["model_name"],
        "system_prompt": model_config.get("system_prompt", ""),
        "cache_dir": str(cache_dir),  # Используем созданный Path объект
        "device_map": model_config["device_map"],
    }

    logger.info(f"Инициализация модели: {model_config['model_name']}")
    
    # Создание экземпляра модели
    try:
        return ModelFactory.get_model(model_family, model_params)
    except Exception as e:
        logger.error(f"Ошибка создания модели {model_family}: {str(e)}")
        raise ValueError(f"Ошибка инициализации модели") from e

def load_prompt(prompt_path: Path) -> str:
    """Загружает промпт из файла с проверкой существования.

    Args:
        prompt_path: Путь к файлу с промптом.

    Returns:
        Текст промпта.

    Raises:
        FileNotFoundError: Если файл не существует.
    """
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Файл промпта не найден: {prompt_path}")
    
    return prompt_path.read_text(encoding="utf-8")