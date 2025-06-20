"""
Скрипт для тестирования модели Qwen2.5-VL.
Демонстрирует базовую функциональность модели на примере анализа документа.
"""

import subprocess
from pathlib import Path
from typing import Optional

from model_interface.model_factory import ModelFactory

try:
    from test_config import MODEL_CONFIG, TEST_CONFIG
except ImportError:
    # Fallback конфигурация, если test_config.py не найден
    MODEL_CONFIG = {
        "model_name": "Qwen2.5-VL-3B-Instruct",
        "device_map": "cuda:0",
        "cache_dir_name": "model_cache",
    }
    TEST_CONFIG = {
        "image_path": "dataset/passport/images/clean/0.jpg",
        "question": "Опиши документ.",
        "show_gpu_info": True,
    }


def test_model(
    image_path: Optional[str] = None,
    question: Optional[str] = None,
    model_name: Optional[str] = None,
    device_map: Optional[str] = None,
    show_gpu_info: Optional[bool] = None
) -> Optional[str]:
    """
    Тестирует модель Qwen2.5-VL на заданном изображении.
    
    Args:
        image_path: Путь к изображению для анализа (по умолчанию из конфигурации)
        question: Вопрос к модели (по умолчанию из конфигурации)
        model_name: Имя модели для загрузки (по умолчанию из конфигурации)
        device_map: Устройство для выполнения модели (по умолчанию из конфигурации)
        show_gpu_info: Показывать ли информацию о GPU (по умолчанию из конфигурации)
        
    Returns:
        Ответ модели или None в случае ошибки
    """
    # Используем значения из конфигурации, если параметры не переданы
    image_path = image_path or TEST_CONFIG["image_path"]
    question = question or TEST_CONFIG["question"]
    model_name = model_name or MODEL_CONFIG["model_name"]
    device_map = device_map or MODEL_CONFIG["device_map"]
    show_gpu_info = show_gpu_info if show_gpu_info is not None else TEST_CONFIG["show_gpu_info"]
    
    # Настройка путей
    script_dir = Path(__file__).parent
    cache_dir = script_dir / MODEL_CONFIG["cache_dir_name"]
    
    # Проверяем существование изображения
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"❌ ОШИБКА: Изображение не найдено: {image_path}")
        return None
    
    try:
        # Инициализация модели
        print(f"🔄 Инициализация модели {model_name}...")
        model = ModelFactory.initialize_qwen_model(
            model_name=model_name,
            cache_dir=str(cache_dir),
            device_map=device_map,
            system_prompt=""
        )
        
        # Тестирование модели
        print(f"📷 Обрабатываем изображение: {image_path}")
        print(f"❓ Вопрос: {question}")
        print("⏳ Генерируем ответ...")
        
        model_answer = model.predict_on_image(image=image_path, question=question)
        
        print(f"\n✅ Ответ модели:")
        print("-" * 50)
        print(model_answer)
        print("-" * 50)
        
        if show_gpu_info:
            show_gpu_status()
            
        return model_answer
        
    except Exception as e:
        print(f"❌ ОШИБКА при работе с моделью: {e}")
        return None


def show_gpu_status() -> None:
    """Показывает информацию о состоянии GPU."""
    print("\n" + "="*60)
    print("🖥️  ИНФОРМАЦИЯ О GPU")
    print("="*60)
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi не найден. Возможно, NVIDIA драйверы не установлены.")


def main():
    """Основная функция для тестирования модели."""
    print("🚀 Запуск тестирования модели Qwen2.5-VL")
    print("="*60)
    
    result = test_model()
    
    if result:
        print("\n✅ Тестирование завершено успешно!")
    else:
        print("\n❌ Тестирование завершено с ошибкой!")


if __name__ == "__main__":
    main()