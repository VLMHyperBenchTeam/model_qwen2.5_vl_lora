import json
from pathlib import Path
from typing import Any, Dict, List, Optional

def load_config(config_path: str) -> Dict[str, Any]:
    """Загружает конфигурацию из JSON файла.

    Args:
        config_path (str): Путь к файлу конфигурации JSON.

    Returns:
        Dict[str, Any]: Словарь с параметрами конфигурации.

    Raises:
        FileNotFoundError: Если файл конфигурации не найден по указанному пути.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Файл конфигурации {config_path} не найден")
    with config_file.open("r") as f:
        return json.load(f)

def get_run_id(model_name: str) -> str:
    """Генерирует идентификатор запуска на основе имени модели.

    Args:
        model_name (str): Имя модели из конфигурации.

    Returns:
        str: Идентификатор запуска.
    """
    return Path(model_name).stem

def save_results_to_csv(
    results: Dict[str, float], 
    filename: str, 
    subset_name: str = None
) -> None:
    """Сохраняет результаты в CSV файл.

    Args:
        results (Dict[str, float]): Словарь с результатами.
        filename (str): Имя файла для сохранения.
        subset_name (str, optional): Имя подмножества данных.
    """
    import pandas as pd
    
    results_df = pd.DataFrame([results])
    if subset_name:
        print(f"\n📊 Метрики для сабсета {subset_name}:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    results_df.to_csv(filename, index=False)

def get_document_type_from_config(config: Dict[str, Any], dataset_path: Path) -> str:
    """Определяет тип документа из конфигурации на основе пути к датасету.

    Args:
        config (Dict[str, Any]): Словарь конфигурации.
        dataset_path (Path): Путь к датасету.

    Returns:
        str: Русское название типа документа.
    """
    document_classes = config.get("document_classes", {})
    dataset_name = dataset_path.name

    for doc_type, doc_name in document_classes.items():
        if doc_type in dataset_name:
            return doc_name

    return dataset_name.replace("_", " ").title()
