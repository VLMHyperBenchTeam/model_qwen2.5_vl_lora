from pathlib import Path
from typing import Any, Dict

from model_interface.model_factory import ModelFactory

def initialize_model(config: Dict[str, Any]) -> Any:
    """Инициализирует и возвращает модель согласно конфигурации.

    Args:
        config (Dict[str, Any]): Словарь конфигурации, содержащий секции
            'model' с параметрами для инициализации модели.

    Returns:
        Any: Инициализированный объект модели.
    """
    model_config = config["model"]

    model_family = model_config["model_family"]
    cache_dir = Path(model_config["cache_dir"])
    cache_dir.mkdir(exist_ok=True)

    # Формируем путь к классу модели из конфигурации
    package = model_config["package"]
    module = model_config["module"]
    model_class = model_config["model_class"]
    model_class_path = f"{package}.{module}:{model_class}"

    # Регистрация модели в фабрике
    ModelFactory.register_model(model_family, model_class_path)

    model_params = {
        "model_name": model_config["model_name"],
        "system_prompt": model_config.get("system_prompt", ""),
        "cache_dir": model_config["cache_dir"],
        "device_map": model_config["device_map"],
    }
    print(f"Инициализация модели: {model_config['model_name']}")
    return ModelFactory.get_model(model_family, model_params)

def load_prompt(prompt_path: Path) -> str:
    """Загружает промпт из файла.

    Args:
        prompt_path (Path): Путь к файлу с промптом.

    Returns:
        str: Текст промпта.
    """
    return prompt_path.read_text(encoding="utf-8")
