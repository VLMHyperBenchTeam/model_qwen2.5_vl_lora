import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from model_interface.model_factory import ModelFactory
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from document_classes import document_classes

# Создаем обратное отображение для быстрого поиска ключа класса.
CLASS_NAMES_TO_KEYS = {v: k for k, v in document_classes.items()}


def load_config(config_path: str = "config_classification.json") -> Dict[str, Any]:
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


def initialize_model(config: Dict[str, Any]) -> Any:
    """Инициализирует и возвращает модель согласно конфигурации.

    Args:
        config (Dict[str, Any]): Словарь конфигурации, содержащий параметры
            для инициализации модели, такие как 'model_family', 'model_name',
            'cache_dir', 'device_map', 'package', 'module', 'model_class'.

    Returns:
        Any: Инициализированный объект модели.
    """
    model_family = config["model_family"]
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(exist_ok=True)

    # Формируем путь к классу модели из конфигурации
    package = config["package"]
    module = config["module"]
    model_class = config["model_class"]
    model_class_path = f"{package}.{module}:{model_class}"

    # Регистрация модели в фабрике
    ModelFactory.register_model(model_family, model_class_path)

    model_params = {
        "model_name": config["model_name"],
        "system_prompt": config.get("system_prompt", ""),
        "cache_dir": config["cache_dir"],
        "device_map": config["device_map"],
    }
    print(f"Инициализация модели: {config['model_name']}")
    return ModelFactory.get_model(model_family, model_params)


def get_image_paths(
    dataset_path: Path,
    class_names: List[str],
    subset_name: str,
    sample_size: Optional[int] = None,
) -> List[Path]:
    """Собирает пути к изображениям для указанного подмножества данных.

    Функция обходит директории классов и подмножеств, собирая пути к файлам.
    Она корректно обрабатывает случаи, когда изображения находятся как
    непосредственно в папке сабсета, так и в дополнительных подпапках.

    Args:
        dataset_path (Path): Корневой путь к датасету.
        class_names (List[str]): Список имен классов для обработки.
        subset_name (str): Имя подмножества (например, 'clean', 'blur').
        sample_size (Optional[int]): Количество файлов для выборки из каждой
            директории класса. Если None, обрабатываются все файлы.

    Returns:
        List[Path]: Список объектов Path, ведущих к выбранным изображениям.
    """
    selected_files = []
    print(f"\n📂 Обработка сабсета: {subset_name}")
    for class_name in class_names:
        class_dir = dataset_path / class_name / "images" / subset_name
        if not class_dir.exists():
            continue

        paths = list(class_dir.iterdir())
        if sample_size is not None:
            paths = paths[:sample_size]

        for path in paths:
            if path.is_file():
                selected_files.append(path)
            else:
                selected_files.extend(p for p in path.iterdir() if p.is_file())

    print(f"Найдено файлов для обработки: {len(selected_files)}")
    return selected_files


def get_prediction(model: Any, image_path: Path, prompt: str) -> str:
    """Получает предсказание модели для одного изображения.

    Args:
        model (Any): Инициализированный объект модели для классификации.
        image_path (Path): Путь к файлу изображения.
        prompt (str): Промпт, который будет подан модели вместе с изображением.

    Returns:
        str: Предсказанный ключ класса (например, 'invoice') или 'None'
             в случае ошибки или некорректного ответа модели.
    """
    try:
        # Передаем путь к изображению напрямую в модель
        result = model.predict_on_image(image=str(image_path), question=prompt)
        prediction = result.strip().strip('"')

        if prediction.isdigit():
            class_index = int(prediction)
            if 0 <= class_index < len(document_classes):
                pred_class_name = list(document_classes.values())[class_index]
                return CLASS_NAMES_TO_KEYS.get(pred_class_name, "None")
        return "None"

    except Exception as e:
        print(f"Ошибка при классификации файла {image_path.name}: {e}")
        return "None"


def calculate_and_save_metrics(
    y_true: List[str], y_pred: List[str], subset_name: str, run_id: str
) -> Dict[str, float]:
    """Вычисляет и сохраняет метрики, возвращает словарь с основными метриками.

    Создает отчет о классификации и CSV-файл с метриками для каждого класса,
    а также CSV-файл с общими метриками (accuracy, f1, precision, recall).

    Args:
        y_true (List[str]): Список истинных меток классов.
        y_pred (List[str]): Список предсказанных меток классов.
        subset_name (str): Имя обрабатываемого подмножества.
        run_id (str): Уникальный идентификатор запуска для именования файлов.

    Returns:
        Dict[str, float]: Словарь с вычисленными средневзвешенными метриками
                          или пустой словарь, если данных для расчета нет.
    """
    if not y_true:
        print("Нет данных для оценки метрик.")
        return {}

    all_classes = list(document_classes.keys())
    if "None" in set(y_pred):
        all_classes.append("None")

    print(f"\n📊 Отчет по классификации для сабсета {subset_name}:")
    report = classification_report(
        y_true, y_pred, labels=all_classes, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{run_id}_{subset_name}_per_class_metrics.csv")
    print(report_df)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    results_df = pd.DataFrame([metrics])
    results_df.to_csv(f"{run_id}_{subset_name}_classification_results.csv", index=False)

    return metrics


def run_evaluation(config: Dict[str, Any]) -> None:
    """Основной цикл оценки модели.

    Оркестрирует весь процесс: от загрузки конфигурации и инициализации
    модели до итерации по подмножествам, сбора предсказаний и расчета
    итоговых средних метрик.

    Args:
        config (Dict[str, Any]): Словарь с полной конфигурацией для запуска.
    """
    dataset_path = Path(config["dataset_path"])
    prompt_path = Path(config["prompt_path"])
    sample_size = config.get("sample_size")

    model = initialize_model(config)

    prompt_template = prompt_path.read_text(encoding="utf-8")
    classes_str = ", ".join(
        f"{idx}: {name}" for idx, name in enumerate(document_classes.values())
    )
    prompt = prompt_template.format(classes=classes_str)

    run_id = Path(config["model_name"]).stem
    all_metrics = []

    for subset in config["subsets"]:
        image_paths = get_image_paths(
            dataset_path, list(document_classes.keys()), subset, sample_size
        )

        if not image_paths:
            continue

        y_true, y_pred = [], []
        # Использование `path.parts` делает код более надежным и независимым от ОС.
        # Структура пути: ./dataset/{class_name}/images/{subset_name}/...
        # Поэтому `class_name` это 4-й элемент с конца (индекс -4).
        for path in tqdm(image_paths, desc=f"Обработка {subset}"):
            y_true.append(path.parts[-4])
            y_pred.append(get_prediction(model, path, prompt))

        subset_metrics = calculate_and_save_metrics(y_true, y_pred, subset, run_id)
        if subset_metrics:
            all_metrics.append(subset_metrics)

    if all_metrics:
        final_df = pd.DataFrame(all_metrics)
        avg_metrics = final_df.mean()
        print("\n📊 Средние метрики по всем сабсетам:")
        print(f"  Средняя точность (Accuracy): {avg_metrics['accuracy']:.4f}")
        print(f"  Средний F1-score: {avg_metrics['f1']:.4f}")
        print(f"  Средняя точность (Precision): {avg_metrics['precision']:.4f}")
        print(f"  Средний отзыв (Recall): {avg_metrics['recall']:.4f}")

        final_df.to_csv(f"{run_id}_final_classification_results.csv", index=False)


def main() -> None:
    """Главная функция для запуска процесса классификации.

    Загружает конфигурацию и запускает основной цикл оценки.
    """
    try:
        config = load_config()
        run_evaluation(config)
    except (FileNotFoundError, KeyError) as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
