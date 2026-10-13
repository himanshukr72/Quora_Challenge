# Quora Question Pair Duplicate Detection

A Natural Language Processing (NLP) project that predicts whether two Quora questions express the same underlying question. The project explores exploratory data analysis, linguistic feature engineering, traditional machine learning, and transformer-based text-pair classification.

## Project Overview

Quora contains many questions that express similar or identical intents using different wording. Detecting duplicate questions can help reduce redundant content and improve question discovery.

This project develops and evaluates models that classify question pairs as:

- **Duplicate (`1`)** — the questions express the same underlying intent.
- **Not duplicate (`0`)** — the questions do not express the same underlying intent.

The project progresses from traditional NLP features to fine-tuned transformer models.

## Dataset

The project uses the Quora Question Pairs dataset.

| Property | Value |
|---|---|
| Original rows | 404,351 |
| Rows after removing missing question text | 404,348 |
| Original columns | 6 |
| Target | `is_duplicate` |
| Duplicate-pair proportion | Approximately 36.93% |

Original columns include `id`, `qid1`, `qid2`, `question1`, `question2`, and `is_duplicate`.

The dataset is not included in this repository by default. Refer to the dataset provider's terms and instructions to obtain it.

## Project Workflow

### 1. Exploratory Data Analysis

The EDA notebook examines:

- Missing values and repeated question IDs
- Class distribution and duplicate-pair proportion
- Question character lengths and word counts
- Common-word counts and word-sharing ratios
- Distributions and relationships between engineered features

### 2. Text Preprocessing

The preprocessing notebook applies operations such as:

- Lowercasing and whitespace trimming
- Normalizing selected currency and special symbols
- Expanding common English contractions
- Handling selected large-number patterns
- Removing `[math]` markers

### 3. Feature Engineering

The project constructs features from multiple aspects of question similarity.

**Basic features**
- Question character lengths
- Question word counts
- Common-word count
- Total unique-word count
- Word-sharing ratio

**Token features**
- Common non-stopword ratios
- Common stopword ratios
- Common-token ratios
- First-word and last-word equality indicators

**Length features**
- Absolute length difference
- Mean question length
- Longest common substring ratio

**Fuzzy similarity features**
- Fuzzy question ratio
- Partial ratio
- Token-sort ratio
- Token-set ratio

The notebooks also experiment with CountVectorizer representations of both questions and use t-SNE to visualize engineered-feature patterns.

### 4. Traditional Machine Learning

Random Forest and XGBoost are explored using engineered and bag-of-words features.

The recorded exploratory runs report:
- Random Forest accuracy: 82.48%
- XGBoost accuracy: 78.66%

These figures are exploratory results, not a final controlled comparison against the transformer models.

### 5. Transformer-Based Classification

Two pretrained transformer models are fine-tuned as binary sequence classifiers:

- `roberta-base`
- `microsoft/deberta-v3-base`

The training pipeline includes:
- Stratified 80% training, 10% validation, and 10% test splits
- Seed-based reproducibility settings
- AdamW optimization and a linear learning-rate schedule
- Mixed-precision computation
- Random question-order swapping during training
- Averaging predictions from both question orders at inference
- Validation-based checkpoint selection and decision-threshold optimization

The final test split is separate from the validation split used for model selection and threshold tuning.

## Results

The following results are reported by the final evaluation cells in the transformer notebooks.

| Metric | RoBERTa | DeBERTa-v3-base |
|---|---:|---:|
| Test accuracy | 90.81% | **91.75%** |
| Duplicate precision | 84.59% | **86.44%** |
| Duplicate recall | 91.84% | **92.10%** |
| Duplicate F1-score | 88.06% | **89.18%** |
| Test log loss | 0.2357 | **0.2283** |

DeBERTa-v3-base achieves the strongest recorded held-out test results in these experiments. It improves duplicate-class F1-score by 1.12 percentage points compared with the recorded RoBERTa run.

The comparison is based on the outputs currently saved in the notebooks. A fully controlled comparison should verify that the same exact split and evaluation setup were used in both runs.

## Repository Structure

```text
quora-challenge/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_basic_data_analysis.ipynb
│   ├── 02_question_preprocessing.ipynb
│   ├── 03_roberta_baseline.ipynb
│   └── 04_deberta_v3_experiment.ipynb
├── models/
│   └── README.md
├── reports/
│   ├── model_comparison.md
│   └── figures/
└── src/
    └── README.md
```

This is a recommended organization for the repository; it does not imply that all these files have already been created.

## Running the Project

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies listed in `requirements.txt`.
4. Obtain the dataset and place `questions.csv` in the configured data location.
5. Run the EDA and preprocessing notebooks.
6. Run the transformer notebooks in a GPU-enabled environment.

The transformer experiments use Hugging Face Transformers and PyTorch. Training on the full dataset requires substantial computation and may take hours.

## Limitations and Future Improvements

- Re-run traditional ML experiments with a verified target column and consistent train/validation/test splits.
- Remove CSV index columns and other non-predictive identifiers from model features.
- Investigate question-level leakage across random row splits by using a split strategy that accounts for shared question IDs or connected question groups.
- Compare models using identical splits and report precision, recall, F1-score, accuracy, and log loss.
- Save the selected model, tokenizer, and decision threshold together for reproducible inference.
- Add a reusable inference script and automated tests for preprocessing and prediction.

## Technologies

Python, pandas, NumPy, scikit-learn, NLTK, fuzzy string matching, Matplotlib, Seaborn, PyTorch, Hugging Face Transformers, RoBERTa, and DeBERTa-v3-base.****
