from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from bench_utils.metrics import calculate_classification_metrics
from model_interface.model_factory import ModelFactory, load_prompt
from bench_utils.utils import get_run_id, load_config, save_results_to_csv


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


def get_prediction(
    model: Any, image_path: Path, prompt: str, document_classes: Dict[str, str]
) -> str:
    """Получает предсказание модели для одного изображения.

    Args:
        model (Any): Инициализированный объект модели для классификации.
        image_path (Path): Путь к файлу изображения.
        prompt (str): Промпт, который будет подан модели вместе с изображением.
        document_classes (Dict[str, str]): Словарь классов документов.

    Returns:
        str: Предсказанный ключ класса (например, 'invoice') или 'None'
             в случае ошибки или некорректного ответа модели.
    """
    try:
        # Передаем путь к изображению напрямую в модель
        result = model.predict_on_image(image=str(image_path), prompt=prompt)
        prediction = result.strip().strip('"')

        if prediction.isdigit():
            class_index = int(prediction)
            if 0 <= class_index < len(document_classes):
                pred_class_name = list(document_classes.values())[class_index]
                # Создаем обратное отображение для быстрого поиска ключа класса
                class_names_to_keys = {v: k for k, v in document_classes.items()}
                return class_names_to_keys.get(pred_class_name, "None")
        return "None"

    except Exception as e:
        print(f"Ошибка при классификации файла {image_path.name}: {e}")
        return "None"


def calculate_and_save_metrics(
    y_true: List[str],
    y_pred: List[str],
    subset_name: str,
    run_id: str,
    document_classes: Dict[str, str],
) -> Dict[str, float]:
    """Вычисляет и сохраняет метрики, возвращает словарь с основными метриками.

    Args:
        y_true (List[str]): Список истинных меток классов.
        y_pred (List[str]): Список предсказанных меток классов.
        subset_name (str): Имя обрабатываемого подмножества.
        run_id (str): Уникальный идентификатор запуска для именования файлов.
        document_classes (Dict[str, str]): Словарь классов документов.

    Returns:
        Dict[str, float]: Словарь с вычисленными метриками или пустой словарь.
    """
    metrics = calculate_classification_metrics(y_true, y_pred, document_classes)
    if metrics:
        save_results_to_csv(
            metrics, f"{run_id}_{subset_name}_classification_results.csv", subset_name
        )
    return metrics


def run_evaluation(config: Dict[str, Any]) -> None:
    """Основной цикл оценки модели.

    Оркестрирует весь процесс: от загрузки конфигурации и инициализации
    модели до итерации по подмножествам, сбора предсказаний и расчета
    итоговых средних метрик.

    Args:
        config (Dict[str, Any]): Словарь с полной конфигурацией для запуска,
                                содержащий секции 'task', 'model' и 'document_classes'.
    """
    task_config = config["task"]
    model_config = config["model"]
    document_classes = config["document_classes"]

    dataset_path = Path(task_config["dataset_path"])
    prompt_path = Path(task_config["prompt_path"])
    sample_size = task_config.get("sample_size")

    model = ModelFactory.initialize_model(model_config)

    prompt_template = load_prompt(prompt_path)
    classes_str = ", ".join(
        f"{idx}: {name}" for idx, name in enumerate(document_classes.values())
    )
    prompt = prompt_template.format(classes=classes_str)

    run_id = get_run_id(model_config["model_name"])
    all_metrics = []

    for subset in task_config["subsets"]:
        image_paths = get_image_paths(
            dataset_path, list(document_classes.keys()), subset, sample_size
        )

        if not image_paths:
            continue

        y_true, y_pred = [], []
        for path in tqdm(image_paths, desc=f"Обработка {subset}"):
            y_true.append(path.parts[-4])
            y_pred.append(get_prediction(model, path, prompt, document_classes))

        subset_metrics = calculate_and_save_metrics(
            y_true, y_pred, subset, run_id, document_classes
        )
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
        config = load_config("config_classification.json")
        run_evaluation(config)
    except (FileNotFoundError, KeyError) as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
