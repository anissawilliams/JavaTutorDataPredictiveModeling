# JavaTutorDataPredictiveModeling

**Predictive modeling of student quiz performance in an AI-powered Java tutoring system.**

---

## Overview

This repository contains all code, data, and documentation needed to reproduce the predictive modeling results from the Java Tutor study. It includes raw and engineered datasets, Jupyter notebooks covering the full analysis pipeline, and a self-contained machine learning script.

**`models/pipeline_final.py` is the final machine learning pipeline.** It supports two feature modes (16-feature full mode and 5-feature behavioral-only mode), trains regression and classification models, runs SHAP interpretability analysis, and saves all outputs automatically.

---

## How to Reproduce the Results

### Option A: Running the Jupyter Notebooks

Open and run the notebooks **in order** from the `analysis/` directory:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_data_feature_analysis.ipynb` | Loads raw data, engineers response-time and pacing features, and explores distributions |
| 2 | `02_hypothesis_testing.ipynb` | Runs statistical tests comparing quiz performance, engagement, and pacing across conditions |
| 3 | `03_baseline_modeling.ipynb` | Trains baseline regression models (Linear Regression, Random Forest, XGBoost) to predict quiz percentage |
| 4 | `04_predictive_models.ipynb` | Builds a full classification pipeline (PCA + Logistic Regression, Random Forest, GBM, SVM) |
| 5 | `05_shap_analysis.ipynb` | SHAP feature importance stub — full SHAP analysis is in `pipeline_final.py` |

To run a notebook: open it in Jupyter, then click **Cell → Run All**, or press `Shift+Enter` to step through cells one at a time.

---

### Option B: Running the Final ML Pipeline

The most complete pipeline is `models/pipeline_final.py`.

#### 16-Feature Mode (default)

```bash
cd models
mkdir -p ../out/16feat
python pipeline_final.py
```

This will load and preprocess session data, train 6 regression models and 4 classification models, generate all plots and SHAP analyses, and save predictions to:

```
out/16feat/ml_predictions_16feat_full.csv
```

#### 5-Feature Behavioral-Only Mode

Open `pipeline_final.py` and change line 40:

```python
FEATURE_MODE = "5"   # Change from "16" to "5"
```

Then run:

```bash
mkdir -p ../out/5feat
python pipeline_final.py
```

Outputs are saved to `out/5feat/`.

---

## Setup and Installation

### Prerequisites

- Python 3.8 or higher — [python.org](https://www.python.org/downloads/)
- pip (included with Python)
- Git — [git-scm.com](https://git-scm.com/downloads)
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/anissawilliams/JavaTutorDataPredictiveModeling.git
   cd JavaTutorDataPredictiveModeling
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   pip install jupyter seaborn shap scipy statsmodels
   ```

4. **Launch Jupyter:**

   ```bash
   jupyter notebook
   ```

   Navigate to the `analysis/` folder and open the notebooks in order.

---

## Repository Structure

```
JavaTutorDataPredictiveModeling/
│
├── README.md                                         ← This file
├── requirements.txt                                  ← Python dependencies
│
├── data/                                             ← All datasets
│   ├── users.csv
│   ├── sessions.csv
│   ├── messages.csv
│   ├── quiz_questions.csv
│   ├── quiz_questions_updated.csv
│   ├── survey_responses.csv
│   ├── sessions_with_engagement_features.csv
│   ├── sessions_with_engagement_features_updated.csv ← Primary modeling file
│   └── java-session-data.json
│
├── analysis/                                         ← Jupyter notebooks (run in order)
│   ├── 01_data_feature_analysis.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   ├── 03_baseline_modeling.ipynb
│   ├── 04_predictive_models.ipynb
│   ├── 05_shap_analysis.ipynb
│   └── features.py                                   ← Computes response-time features from messages
│
├── models/                                           ← ML scripts
│   ├── baseline_models.py                            ← Reusable preprocessing and evaluation functions
│   ├── model_pipeline.py                             ← Pipeline v4 (5 features, min_response_time)
│   ├── model_pipeline_avg_resp_time.py               ← Pipeline v4 variant (5 features, avg_response_time)
│   └── pipeline_final.py                             ← Final pipeline — 5-feature and 16-feature modes
│
├── figures/                                          ← Generated plots and visualizations
│   ├── h1_quiz_performance_by_condition.png
│   ├── h2_*.png
│   ├── h3_*.png
│   ├── performance_by_difficulty.png
│   ├── figure3_efficiency_*.png
│   └── quantitative_behavior_graphic.png

```

---

## Project Overview

An AI-powered Java tutor was used to teach students two topics — **ArrayLists** and **Recursion** — under three tutoring conditions:

| Condition | Description |
|-----------|-------------|
| **1 — Character Scaffolded** | AI tutor with a persona guides students through a structured 7-step instructional sequence |
| **2 — Non-Character Scaffolded** | Same 7-step scaffolded structure, without a character persona |
| **3 — Direct Chat** | Open-ended conversation with no structured scaffolding |

Each student completed two tutoring sessions (one per topic), followed by a 5-question quiz and a survey. The study investigates whether scaffolded tutoring improves quiz performance, whether session engagement behaviors predict quiz scores, and which features are most predictive of student success.

---

## Dataset Description

All data files are in the `data/` directory:

| File | Rows | Description |
|------|------|-------------|
| `users.csv` | 68 | Student IDs, assigned conditions, and emails |
| `sessions.csv` | 136 | One row per student per topic — duration, message counts, quiz scores, difficulty metrics |
| `messages.csv` | 12,371 | Every chat message (user and assistant) with timestamps and content length |
| `quiz_questions.csv` | 585 | Individual quiz responses with difficulty levels (1–3) and correctness |
| `quiz_questions_updated.csv` | 585 | Updated version of quiz question responses |
| `survey_responses.csv` | 124 | Post-session survey responses (Likert scales and open text) |
| `sessions_with_engagement_features.csv` | 136 | Sessions enriched with engineered pacing and engagement features |
| `sessions_with_engagement_features_updated.csv` | 136 | **Primary modeling file** — final enriched session data used by the ML pipeline |
| `java-session-data.json` | — | Raw session data export from the tutoring platform |

**Key columns in `sessions_with_engagement_features_updated.csv`:**

- `condition` — tutoring condition (1, 2, or 3)
- `session_type` — topic (`arraylist` or `recursion`)
- `duration_seconds` — total session length
- `total_messages`, `user_messages`, `assistant_messages` — message counts
- `avg_response_time`, `median_response_time`, `std_response_time`, `min_response_time`, `max_response_time` — pacing features
- `rapid_response_count`, `rapid_response_pct` — proportion of responses under 10 seconds
- `avg_difficulty_correct`, `avg_difficulty_incorrect` — average difficulty of correctly and incorrectly answered questions
- `quiz_percentage` — **target variable** (0–100)

---

## Analysis Pipeline

The project follows four phases:

### Phase 1: Feature Engineering

Response-time features are computed from raw `messages.csv` timestamps using `analysis/features.py`. These features capture how quickly and consistently students respond to the AI tutor, including average, median, and standard deviation of response times, and a count of rapid responses (under 10 seconds).

### Phase 2: Hypothesis Testing

Three hypotheses are tested in `02_hypothesis_testing.ipynb`:

- **H1:** Scaffolded conditions produce higher quiz scores than direct chat
- **H2:** Scaffolded conditions show higher engagement (more messages, longer sessions)
- **H3:** Scaffolded conditions produce more consistent pacing behavior

### Phase 3: Predictive Modeling

Models are trained to predict `quiz_percentage` using two approaches:

- **Regression** — predicts the continuous quiz score (0–100)
- **Classification** — predicts whether a student scores High (≥70%) or Low (<70%)

Two feature sets are compared:

- **5-feature set** (behavioral only): condition, session type, duration, total messages, average response time
- **16-feature set** (full): adds all pacing metrics, message breakdowns, and difficulty features

### Phase 4: Interpretability

SHAP (SHapley Additive exPlanations) analysis identifies which features drive predictions, run via the `run_shap()` function in `pipeline_final.py`.

---

## Models and Results

`pipeline_final.py` trains and evaluates the following models using 5-fold cross-validation and an 80/20 train/test split (`random_state=44`). PCA is applied before training (up to 5 components).

**Regression models** (predicting `quiz_percentage`):

| Model | Notes |
|-------|-------|
| Linear Regression | Baseline |
| Ridge Regression | α = 1.0 |
| Lasso Regression | α = 1.0 |
| Random Forest Regressor | — |
| Gradient Boosting Regressor | — |
| Support Vector Regression | RBF kernel |

**Classification models** (predicting High ≥70% vs. Low <70%):

| Model | Notes |
|-------|-------|
| Logistic Regression | — |
| Random Forest Classifier | — |
| Gradient Boosting Classifier | — |
| SVM | RBF kernel — best overall performer |

**Best model: SVM (RBF kernel).** The SVM classifier achieved the highest classification accuracy across both feature modes. Key predictive features include session duration, average response time, and tutoring condition. Scaffolded conditions consistently show stronger predictive signal than direct chat.

All outputs — confusion matrices, scatter plots, residual distributions, model comparison charts, feature importance rankings, and SHAP summary plots — are saved to the `out/` directory at runtime.

---

## Code Attribution and Changes

### External Libraries

- **scikit-learn** — preprocessing, model training, cross-validation, and evaluation
- **SHAP** — feature interpretability and dependence plots
- **XGBoost** — gradient boosting baseline
- **seaborn / matplotlib** — all visualizations
- The classification pipeline structure in `04_predictive_models.ipynb` was informed by a Titanic dataset PCA + Logistic Regression tutorial (noted in the notebook docstring); no code was copied directly

### Original Contributions

- **`features.py`** — custom feature engineering code computing session-level response-time statistics from raw message timestamps
- **`02_hypothesis_testing.ipynb`** — all statistical analyses (ANOVA, t-tests, correlation, regression) written from scratch for this dataset
- **`pipeline_final.py`** — fully original; includes dual regression/classification training, toggleable 5-feature vs. 16-feature mode, a unified forest/sage/gold visualization theme, SHAP interpretability, and automated plot export
- **`baseline_models.py`** — reusable scikit-learn Pipeline wrappers written from scratch
- `model_pipeline.py` and `model_pipeline_avg_resp_time.py` represent earlier iterations exploring different feature combinations before converging on the final pipeline
- All visualizations and the custom color palette were developed for this project's poster presentation

---

## Transparency Note

Some statistical analyses in this repository — including response-time distributions, bootstrap confidence intervals, and within-session trend estimation — also appear in a separate Scientific Computing course assignment. Both projects analyze the same Java Tutor dataset. The analyses were developed for and are presented in both contexts with full acknowledgment of the overlap.

---

## License

This project was created for academic coursework at the College of Charleston. All datasets contain anonymized student interaction data from an AI Java tutoring study and are used strictly for educational research purposes.
