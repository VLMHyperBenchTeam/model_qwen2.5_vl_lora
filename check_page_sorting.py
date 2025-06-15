import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from model_interface.model_factory import ModelFactory
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm


def load_config(config_path: str = "config_page_sorting.json") -> Dict[str, Any]:
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
            'cache_dir', и 'device_map'.

    Returns:
        Any: Инициализированный объект модели.
    """
    model_family = config["model_family"]
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(exist_ok=True)

    ModelFactory.register_model(model_family, "model_qwen2_5_vl.models:Qwen2_5_VLModel")

    model_params = {
        "model_name": config["model_name"],
        "system_prompt": "",
        "cache_dir": str(cache_dir),
        "device_map": config["device_map"],
    }
    print(f"Инициализация модели: {config['model_name']}")
    return ModelFactory.get_model(model_family, model_params)


def get_image_paths_for_document(
    dataset_path: Path, document_id: str, subset_name: str
) -> List[Path]:
    """Получает пути к изображениям страниц для конкретного документа.

    Args:
        dataset_path (Path): Корневой путь к датасету.
        document_id (str): Идентификатор документа.
        subset_name (str): Имя подмножества (например, 'clean', 'blur').

    Returns:
        List[Path]: Список путей к изображениям страниц документа в порядке номеров.
    """
    document_dir = dataset_path / "images" / subset_name / document_id
    if not document_dir.exists():
        return []

    # Сортируем файлы по номерам страниц (0.jpg, 1.jpg, 2.jpg, 3.jpg)
    image_files = []
    for i in range(10):  # Проверяем до 10 страниц
        image_path = document_dir / f"{i}.jpg"
        if image_path.exists():
            image_files.append(image_path)
        else:
            break

    return image_files


def get_document_ids(dataset_path: Path, subset_name: str) -> List[str]:
    """Получает список ID документов в указанном подмножестве.

    Args:
        dataset_path (Path): Корневой путь к датасету.
        subset_name (str): Имя подмножества.

    Returns:
        List[str]: Список ID документов.
    """
    subset_dir = dataset_path / "images" / subset_name
    if not subset_dir.exists():
        return []

    return [d.name for d in subset_dir.iterdir() if d.is_dir()]


def load_ground_truth(dataset_path: Path, document_id: str) -> List[int]:
    """Загружает правильный порядок страниц из JSON файла.

    Args:
        dataset_path (Path): Корневой путь к датасету.
        document_id (str): Идентификатор документа.

    Returns:
        List[int]: Правильный порядок страниц (индексы от 1).
    """
    json_file = dataset_path / "jsons" / f"{document_id}.json"
    if not json_file.exists():
        return []

    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Преобразуем индексы из 0-based в 1-based
    true_order = data["fields"]["interest_free_loan_agreement"]
    return [i + 1 for i in true_order]


def extract_json_from_model_output(model_output: str) -> Optional[Dict[str, Any]]:
    """Извлекает и парсит JSON из вывода модели, обрабатывая различные форматы.

    Функция умеет обрабатывать:
    - JSON обернутый в markdown блоки (```json ... ```)
    - Чистый JSON без оберток
    - JSON с дополнительным текстом вокруг

    Args:
        model_output (str): Сырой вывод модели.

    Returns:
        Optional[Dict[str, Any]]: Распарсенный JSON или None в случае ошибки.
    """
    if not isinstance(model_output, str):
        print(f"Ожидается строка, получен: {type(model_output)}")
        return None

    cleaned_output = model_output.strip()

    # Попытка 1: Поиск JSON в markdown блоке
    json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", cleaned_output, re.DOTALL)
    if json_match:
        json_content = json_match.group(1).strip()
    else:
        # Попытка 2: Поиск JSON блока по фигурным скобкам
        json_match = re.search(r"\{.*\}", cleaned_output, re.DOTALL)
        if json_match:
            json_content = json_match.group(0).strip()
        else:
            # Попытка 3: Используем весь вывод как есть
            json_content = cleaned_output

    # Пытаемся распарсить JSON
    try:
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"Ошибка парсинга JSON: {e}")
        print(f"Содержимое для парсинга: {repr(json_content[:200])}...")
        return None


def extract_ordered_pages_from_json(parsed_json: Dict[str, Any]) -> List[int]:
    """Извлекает список упорядоченных страниц из распарсенного JSON.

    Args:
        parsed_json (Dict[str, Any]): Распарсенный JSON ответ модели.

    Returns:
        List[int]: Список номеров страниц в правильном порядке или пустой список.
    """
    if not isinstance(parsed_json, dict):
        print(f"Ожидается словарь, получен: {type(parsed_json)}")
        return []

    # Проверяем наличие ключа ordered_pages
    if "ordered_pages" in parsed_json:
        pages = parsed_json["ordered_pages"]
        if isinstance(pages, list) and all(isinstance(p, int) for p in pages):
            return pages
        else:
            print(
                f"Значение 'ordered_pages' должно быть списком целых чисел, получено: {pages}"
            )
    else:
        print(
            f"Ключ 'ordered_pages' не найден в JSON. Доступные ключи: {list(parsed_json.keys())}"
        )

    return []


def parse_model_output_fallback(model_output: str) -> List[int]:
    """Резервный способ извлечения чисел из вывода модели.

    Ищет в тексте массивы чисел вида [1, 2, 3, 4] или последовательности чисел.

    Args:
        model_output (str): Сырой вывод модели.

    Returns:
        List[int]: Найденные числа или пустой список.
    """
    # Поиск массива чисел в квадратных скобках
    array_match = re.search(r"\[[\d\s,]+\]", model_output)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass

    # Поиск отдельных чисел
    numbers = re.findall(r"\b[1-9]\b", model_output)
    if numbers:
        try:
            return [int(n) for n in numbers[:4]]  # Берем первые 4 числа
        except ValueError:
            pass

    return []


def process_model_response(model_response: str) -> List[int]:
    """Обрабатывает ответ модели и извлекает порядок страниц.

    Args:
        model_response (str): Сырой ответ модели.

    Returns:
        List[int]: Предсказанный порядок страниц или пустой список в случае ошибки.
    """
    if not isinstance(model_response, str):
        print(f"Модель вернула неожиданный тип: {type(model_response)}")
        return []

    # Основной способ: парсинг JSON
    parsed_json = extract_json_from_model_output(model_response)
    if parsed_json:
        ordered_pages = extract_ordered_pages_from_json(parsed_json)
        if ordered_pages:
            return ordered_pages

    # Резервный способ: поиск чисел в тексте
    print("Основной парсинг не удался, пробуем резервный способ...")
    fallback_result = parse_model_output_fallback(model_response)
    if fallback_result:
        print(f"Резервный способ дал результат: {fallback_result}")
        return fallback_result

    print("Не удалось извлечь порядок страниц из ответа модели")
    print(f"Сырой ответ: {repr(model_response[:300])}...")
    return []


def get_prediction(model: Any, image_paths: List[Path], prompt: str) -> List[int]:
    """Получает предсказание модели для упорядочивания страниц документа.

    Args:
        model (Any): Инициализированный объект модели.
        image_paths (List[Path]): Список путей к изображениям страниц.
        prompt (str): Промпт для модели.

    Returns:
        List[int]: Предсказанный порядок страниц или пустой список в случае ошибки.
    """
    try:
        # Преобразуем пути в строки для передачи в модель
        image_paths_str = [str(path) for path in image_paths]

        # Получаем ответ от модели
        model_response = model.predict_on_images(
            images=image_paths_str, question=prompt
        )

        # Обрабатываем ответ модели
        return process_model_response(model_response)

    except Exception as e:
        print(f"Ошибка при предсказании для документа: {e}")
        return []


def evaluate_ordering(
    true_order: List[int], predicted_order: List[int]
) -> Dict[str, float]:
    """Вычисляет метрики качества упорядочивания страниц.

    Args:
        true_order (List[int]): Правильный порядок страниц.
        predicted_order (List[int]): Предсказанный порядок страниц.

    Returns:
        Dict[str, float]: Словарь с метриками (kendall_tau, accuracy, spearman_rho).
    """
    if not true_order or not predicted_order or len(true_order) != len(predicted_order):
        return {"kendall_tau": 0.0, "accuracy": 0.0, "spearman_rho": 0.0}

    if set(true_order) != set(predicted_order):
        print(f"Предупреждение: наборы страниц не совпадают")
        print(f"Правильный: {true_order}")
        print(f"Предсказанный: {predicted_order}")
        return {"kendall_tau": 0.0, "accuracy": 0.0, "spearman_rho": 0.0}

    # Словарь: страница → её позиция в правильном порядке
    true_positions = {page: i for i, page in enumerate(true_order)}

    # Сопоставим позиции predicted_order с эталонными
    true_ranks = [true_positions[page] for page in predicted_order]
    pred_ranks = list(range(len(predicted_order)))

    # Метрики
    kendall, _ = kendalltau(pred_ranks, true_ranks)
    accuracy = sum(t == p for t, p in zip(true_order, predicted_order)) / len(
        true_order
    )
    rho, _ = spearmanr(true_order, predicted_order)

    return {
        "kendall_tau": round(kendall, 4),
        "accuracy": round(accuracy, 4),
        "spearman_rho": round(rho, 4),
    }


def save_prediction(output_dir: Path, document_id: str, prediction: List[int]) -> None:
    """Сохраняет предсказание в JSON файл.

    Args:
        output_dir (Path): Директория для сохранения результатов.
        document_id (str): Идентификатор документа.
        prediction (List[int]): Предсказанный порядок страниц.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{document_id}.json"

    result = {"ordered_pages": prediction}
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def calculate_and_save_metrics(
    all_metrics: Dict[str, List[float]], subset_name: str, run_id: str
) -> Dict[str, float]:
    """Вычисляет средние метрики и сохраняет результаты.

    Args:
        all_metrics (Dict[str, List[float]]): Словарь со списками метрик.
        subset_name (str): Имя обрабатываемого подмножества.
        run_id (str): Уникальный идентификатор запуска.

    Returns:
        Dict[str, float]: Словарь со средними метриками.
    """
    if not all_metrics or not any(all_metrics.values()):
        print("Нет данных для вычисления метрик.")
        return {}

    # Вычисляем средние значения
    mean_metrics = {
        key: round(sum(values) / len(values), 4) if values else 0.0
        for key, values in all_metrics.items()
    }

    print(f"\n📊 Метрики для сабсета {subset_name}:")
    for key, value in mean_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Сохраняем результаты
    results_df = pd.DataFrame([mean_metrics])
    results_df.to_csv(f"{run_id}_{subset_name}_page_sorting_results.csv", index=False)

    return mean_metrics


def run_evaluation(config: Dict[str, Any]) -> None:
    """Основной цикл оценки модели для задачи упорядочивания страниц.

    Args:
        config (Dict[str, Any]): Словарь с полной конфигурацией для запуска.
    """
    dataset_path = Path(config["dataset_path"])
    prompt_path = Path(config["prompt_path"])
    output_base_dir = Path(config.get("output_dir", "output"))

    model = initialize_model(config)

    prompt = prompt_path.read_text(encoding="utf-8")
    run_id = Path(config["model_name"]).stem
    all_subset_metrics = []

    for subset in config["subsets"]:
        print(f"\n📂 Обработка сабсета: {subset}")

        # Получаем список документов в подмножестве
        document_ids = get_document_ids(dataset_path, subset)
        if not document_ids:
            print(f"Нет документов в сабсете {subset}")
            continue

        # Применяем ограничение по количеству документов если указано
        sample_size = config.get("sample_size")
        if sample_size:
            document_ids = document_ids[:sample_size]

        print(f"Найдено документов для обработки: {len(document_ids)}")

        # Создаем директорию для результатов
        output_dir = output_base_dir / dataset_path.name / subset

        all_metrics = {
            "kendall_tau": [],
            "accuracy": [],
            "spearman_rho": [],
        }

        # Обрабатываем каждый документ
        for doc_id in tqdm(document_ids, desc=f"Обработка {subset}"):
            # Получаем пути к изображениям страниц
            image_paths = get_image_paths_for_document(dataset_path, doc_id, subset)
            if len(image_paths) != 4:
                print(
                    f"Документ {doc_id}: ожидается 4 страницы, найдено {len(image_paths)}"
                )
                continue

            # Загружаем правильный порядок
            true_order = load_ground_truth(dataset_path, doc_id)
            if not true_order:
                print(f"Не удалось загрузить правильный порядок для документа {doc_id}")
                continue

            # Получаем предсказание
            predicted_order = get_prediction(model, image_paths, prompt)
            if not predicted_order:
                print(f"Не удалось получить предсказание для документа {doc_id}")
                continue

            # Сохраняем предсказание
            save_prediction(output_dir, doc_id, predicted_order)

            # Вычисляем метрики
            metrics = evaluate_ordering(true_order, predicted_order)
            for key, value in metrics.items():
                all_metrics[key].append(value)

            print(f"Документ {doc_id}: {metrics}")

        # Вычисляем и сохраняем средние метрики для подмножества
        subset_metrics = calculate_and_save_metrics(all_metrics, subset, run_id)
        if subset_metrics:
            all_subset_metrics.append(subset_metrics)

    # Вычисляем общие метрики по всем подмножествам
    if all_subset_metrics:
        final_df = pd.DataFrame(all_subset_metrics)
        overall_metrics = final_df.mean()

        print("\n📊 Средние метрики по всем сабсетам:")
        print(f"  Средняя точность (Accuracy): {overall_metrics['accuracy']:.4f}")
        print(f"  Средний Kendall Tau: {overall_metrics['kendall_tau']:.4f}")
        print(f"  Средний Spearman Rho: {overall_metrics['spearman_rho']:.4f}")

        final_df.to_csv(f"{run_id}_final_page_sorting_results.csv", index=False)


def main() -> None:
    """Главная функция для запуска процесса упорядочивания страниц.

    Загружает конфигурацию и запускает основной цикл оценки.
    """
    try:
        config = load_config()
        run_evaluation(config)
    except (FileNotFoundError, KeyError) as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
