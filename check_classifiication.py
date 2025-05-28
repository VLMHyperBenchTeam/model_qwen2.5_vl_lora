import base64
import json
import mimetypes
import os
from pathlib import Path

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


def image_to_base64(image_path):
    """Конвертирует изображение в base64 строку с MIME типом"""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_string}"


def read_prompt_from_file(filepath: Path) -> str:
    """Читает промпт из файла"""
    with open(filepath, "r") as file:
        return file.read()


def run_classification(base64_image: str, prompt: str, model) -> str:
    """Выполняет классификацию с использованием локальной модели"""
    try:
        result = model.predict_on_image(image=base64_image, question=prompt)
        print(result.strip().strip('"'))
        return result.strip().strip('"')
    except Exception as e:
        print(f"Ошибка при классификации: {e}")
        return "None"


def classify_images(config):
    """Основная функция классификации с использованием локальной модели"""
    dataset_path = Path(config["dataset_path"])
    prompt_path = Path(config["prompt_path"])
    model_name = config.get("model_name", "Qwen2.5-VL-3B-Instruct")
    subsets = config.get(
        "subsets", ["blur", "noise", "clean", "bright", "gray", "rotated", "spatter"]
    )

    if isinstance(subsets, str):
        subsets = [s.strip() for s in subsets.split(",")]

    # Инициализация модели
    cache_directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_cache"
    )
    model_family = "Qwen2.5-VL"
    ModelFactory.register_model(model_family, "model_qwen2_5_vl.models:Qwen2_5_VLModel")

    model_init_params = {
        "model_name": model_name,
        "system_prompt": "",
        "cache_dir": cache_directory,
        "device_map": "cuda:0",
    }

    model = ModelFactory.get_model(model_family, model_init_params)

    # Чтение промпта
    prompt = read_prompt_from_file(prompt_path)
    run_id = model_name.split("/")[-1]
    y_true = []
    y_pred = []
    all_classes = list(document_classes.keys())
    classes_str = ", ".join(
        f"{idx}: {name}" for idx, name in enumerate(document_classes.values())
    )
    prompt = prompt.format(classes=classes_str)

    # Списки для хранения метрик
    all_dfs = []
    all_field_metrics = []
    all_acc = []
    all_f1 = []
    all_precision = []
    all_recall = []

    # Обработка по сабсетам
    for subset_name in subsets:
        print(f"\n📂 Обработка сабсета: {subset_name}")
        selected_files = []

        # Сбор файлов для обработки
        for cls_d in all_classes:
            class_dir = dataset_path / cls_d / "images" / subset_name
            if not class_dir.exists():
                continue

            pathes = list(class_dir.iterdir())[:3]
            for d in pathes:
                if d.is_file():
                    selected_files.append(d)
                else:
                    selected_files.extend(list(d.iterdir()))

        print(f"Найдено файлов: {len(selected_files)}")

        # Обработка файлов
        for image_path in tqdm(selected_files, desc=f"Processing {subset_name}"):
            try:
                base64_image = image_to_base64(str(image_path))
                pred = run_classification(base64_image, prompt, model)

                # Обработка результата
                if pred.isdigit():
                    pred_class = list(document_classes.values())[int(pred)]
                    key = next(
                        (k for k, v in document_classes.items() if v == pred_class),
                        "None",
                    )
                    y_pred.append(key)
                else:
                    y_pred.append("None")

                y_true.append(str(image_path).split("/")[1])

            except Exception as err:
                print(f"Ошибка обработки {image_path}: {err}")
                continue

        # Вычисление метрик
        if not y_true:
            print("Нет данных для оценки метрик")
            continue

        all_classes_for_target = list(document_classes.keys())
        if len(set(y_true).union(set(y_pred))) > len(all_classes_for_target):
            all_classes_for_target.append("None")

        print(f"\n📊 Classification Report for subset {subset_name}:")
        report = classification_report(
            y_true, y_pred, target_names=all_classes_for_target, output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()

        # Сохранение отчета
        report_df.to_csv(f"{run_id}_{subset_name}_per_class_metrics.csv")
        print(report_df)

        # Вычисление метрик
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")

        # Сохранение результатов
        results_df = pd.DataFrame(
            [
                {
                    "prompt": prompt,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": acc,
                }
            ]
        )

        results_df.to_csv(
            f"{run_id}_{subset_name}_classification_results.csv", index=False
        )
        all_dfs.append(results_df)
        all_field_metrics.append(report_df)

        # Добавление метрик в списки
        all_acc.append(acc)
        all_f1.append(f1)
        all_precision.append(precision)
        all_recall.append(recall)

        # Очистка для следующего сабсета
        y_true.clear()
        y_pred.clear()

    # Средние значения метрик по всем сабсетам
    if all_acc:
        avg_acc = sum(all_acc) / len(all_acc)
        avg_f1 = sum(all_f1) / len(all_f1)
        avg_precision = sum(all_precision) / len(all_precision)
        avg_recall = sum(all_recall) / len(all_recall)

        print("\n📊 Средние метрики по всем сабсетам:")
        print(f"Средняя точность (Accuracy): {avg_acc:.4f}")
        print(f"Средний F1: {avg_f1:.4f}")
        print(f"Средняя точность (Precision): {avg_precision:.4f}")
        print(f"Средний отзыв (Recall): {avg_recall:.4f}")

        # Сохранение финальных результатов
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_field_metrics = pd.concat(all_field_metrics, ignore_index=True)
        final_df.to_csv(f"{run_id}_final_classification_results.csv", index=False)
        final_field_metrics.to_csv(f"{run_id}_final_per_class_metrics.csv", index=False)


def main():
    """Основная функция с чтением конфигурации из JSON"""
    config_path = "config_classification.json"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации {config_path} не найден")

    with open(config_path, "r") as f:
        config = json.load(f)

    classify_images(config)


if __name__ == "__main__":
    main()
