# USG-LLM

Prediction of menopausal status based on ultrasound reports and clinical indicators.

This project leverages Large Language Models (LLMs) to automatically extract morphological features from transvaginal ultrasound reports. Combined with anthropometric and hormonal indicators, we build multiple machine learning models to predict menopausal status in women aged 45–55.

---

## Project Structure

```
.
├── core/                  # Core modules
│   ├── build_dataset.py      # Dataset builder and feature selector
│   ├── feature_extraction.py # LLM-based feature extraction from ultrasound reports
│   ├── llm_compare.py        # Comparison across multiple LLMs
│   └── metrics_saver.py      # Evaluation metrics saver
├── main/                  # Model training scripts
│   ├── xgboost_train.py
│   ├── lightgbm_train.py
│   ├── rf_train.py
│   ├── catboost_train.py
│   ├── ann_train.py
│   ├── svm_train.py
│   ├── knn_train.py
│   ├── nb_train.py
│   └── run_all_models.py     # Batch runner for all models
├── paints/                # Visualization scripts
│   ├── eda.py
│   ├── model_roc.py
│   ├── paint_shap.py
│   └── llm_compare.py
├── dataset/               # Raw data and extraction results
├── exps/                  # Experiment outputs (ROC data, etc.)
└── tables/                # Result tables (performance metrics, distributions, etc.)
```

---

## Feature Groups

| Category | Features |
|----------|----------|
| **Anthropometric** | Age, Height, Weight, BMI |
| **Morphological (LLM-extracted)** | OA (Ovarian Atrophy), EA (Endometrial Atrophy), UA (Uterine Atrophy) |
| **Hormonal** | AMH, FSH, LH, P, E2, Prolactin, Testosterone |

---

## Requirements

- Python ≥ 3.12
- Dependency management: [uv](https://github.com/astral-sh/uv)

### Install Dependencies

```bash
uv sync
```

---

## Quick Start

### 1. Extract Morphological Features from Ultrasound Reports

```bash
python core/feature_extraction.py
```

This script calls the Tongyi Qwen model to extract the OA / EA / UA features from the ultrasound report text in `dataset/data1.xlsx` and `dataset/data2.xlsx`. The results are saved as `dataset/data1_extraction.xlsx` and `dataset/data2_extraction.xlsx`.

### 2. Train and Evaluate All Models

```bash
python main/run_all_models.py
```

Sequentially runs 8 models: XGBoost, LightGBM, Random Forest, CatBoost, ANN, SVM, KNN, and Naive Bayes. Results are aggregated into `tables/performance.csv`.

### 3. Generate Visualizations

```bash
python paints/model_roc.py      # Plot ROC curves
python paints/paint_shap.py    # SHAP feature importance
python paints/eda.py            # Exploratory data analysis
```

---

## Supported Models

- XGBoost
- LightGBM
- Random Forest
- CatBoost
- Artificial Neural Network (ANN)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes (NB)

---

## Dataset Description

- `data1.xlsx` / `data2.xlsx`: Raw clinical data (including ultrasound report text)
- `data1_extraction.xlsx` / `data2_extraction.xlsx`: Full feature data after LLM extraction
- Age range of study subjects: 45–55 years
- Target variable: `Menopause` (0 = non-menopausal, 1 = menopausal)

---

## License

This project is for academic research purposes only.
