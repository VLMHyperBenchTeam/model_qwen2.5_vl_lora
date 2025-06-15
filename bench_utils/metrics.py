from typing import List, Dict
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from scipy.stats import kendalltau, spearmanr

__all__ = ['calculate_classification_metrics', 'calculate_ordering_metrics', 'kendalltau', 'spearmanr']

def calculate_classification_metrics(
    y_true: List[str], y_pred: List[str], document_classes: Dict[str, str]
) -> Dict[str, float]:
    """Вычисляет метрики классификации и возвращает словарь с основными метриками.

    Args:
        y_true (List[str]): Список истинных меток классов.
        y_pred (List[str]): Список предсказанных меток классов.
        document_classes (Dict[str, str]): Словарь классов документов.

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

    report = classification_report(
        y_true, y_pred, labels=all_classes, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    return metrics

def calculate_ordering_metrics(
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

    true_positions = {page: i for i, page in enumerate(true_order)}
    true_ranks = [true_positions[page] for page in predicted_order]
    pred_ranks = list(range(len(predicted_order)))

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
