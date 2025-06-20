# 📝 Отчёт по задаче классификации


## Параметры задачи

* **Датасет:** `./dataset`
* **Subsets:** clean
* **Sample size:** 3

## Параметры модели

* **Модель:** `Qwen2.5-VL-3B-Instruct`
* **model_family:** Qwen2.5-VL
* **device_map:** cuda:0
* **cache_dir:** ./model_cache
* **package:** model_qwen2_5_vl
* **module:** models
* **model_class:** Qwen2_5_VLModel
* **system_prompt:** 

## Список классов документов

| Ключ | Название |
|------|----------|
| invoice | Счет-фактура |
| tin_new | ИНН_нового образца |
| tin_old | ИНН старого образца |
| passport | Паспорт |
| snils | СНИЛС |
| interest_free_loan_agreement | Договор беспроцентного займа |

## Итоговые метрики

| Accuracy | F1-score | Precision | Recall |
|----------|---------|-----------|--------|
| 0.7778 | 0.7222 | 0.7037 | 0.7778 |

## Метрики по сабсетам

| Сабсет | Accuracy | F1-score | Precision | Recall |
|--------|----------|---------|-----------|--------|
| clean  0.7778 | 0.7222 | 0.7037 | 0.7778 |

## Метрики по документам — общий датасет

|                              |   precision |   recall |   f1-score |   support |
|------------------------------|-------------|----------|------------|-----------|
| invoice                      |      1      |   1      |     1      |         3 |
| tin_new                      |      0      |   0      |     0      |         3 |
| tin_old                      |      0.3333 |   1      |     0.5    |         3 |
| passport                     |      1      |   1      |     1      |         3 |
| snils                        |      0      |   0      |     0      |         3 |
| interest_free_loan_agreement |      1      |   1      |     1      |        12 |
| macro avg                    |      0.5556 |   0.6667 |     0.5833 |        27 |
| weighted avg                 |      0.7037 |   0.7778 |     0.7222 |        27 |

## Метрики по документам — clean

|                              |   precision |   recall |   f1-score |   support |
|------------------------------|-------------|----------|------------|-----------|
| invoice                      |      1      |   1      |     1      |         3 |
| tin_new                      |      0      |   0      |     0      |         3 |
| tin_old                      |      0.3333 |   1      |     0.5    |         3 |
| passport                     |      1      |   1      |     1      |         3 |
| snils                        |      0      |   0      |     0      |         3 |
| interest_free_loan_agreement |      1      |   1      |     1      |        12 |
| macro avg                    |      0.5556 |   0.6667 |     0.5833 |        27 |
| weighted avg                 |      0.7037 |   0.7778 |     0.7222 |        27 |

## Confusion Matrix — clean

|                              |   invoice |   tin_new |   tin_old |   passport |   snils |   interest_free_loan_agreement |
|------------------------------|-----------|-----------|-----------|------------|---------|--------------------------------|
| invoice                      |         3 |         0 |         0 |          0 |       0 |                              0 |
| tin_new                      |         0 |         0 |         3 |          0 |       0 |                              0 |
| tin_old                      |         0 |         0 |         3 |          0 |       0 |                              0 |
| passport                     |         0 |         0 |         0 |          3 |       0 |                              0 |
| snils                        |         0 |         0 |         3 |          0 |       0 |                              0 |
| interest_free_loan_agreement |         0 |         0 |         0 |          0 |       0 |                             12 |