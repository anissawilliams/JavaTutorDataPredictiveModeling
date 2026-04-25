# JavaTutorDataPredictiveModeling

**Predictive Modeling of Student Quiz Performance in an AI-Powered Java Tutor**

This project analyzes data from an AI Java tutoring system to understand how different tutoring conditions (character-scaffolded, non-character-scaffolded, and direct chat) affect student learning outcomes. It uses statistical hypothesis testing and machine learning to predict student quiz performance based on session engagement features.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Repository Structure](#repository-structure)
4. [Setup & Installation](#setup--installation)
5. [How to Reproduce Results](#how-to-reproduce-results)
6. [Analysis Pipeline](#analysis-pipeline)
7. [Models & Results](#models--results)
8. [Code Attribution & Changes](#code-attribution--changes)

---

## Project Overview

An AI-powered Java tutor was used to teach students two topics — **ArrayLists** and **Recursion** — under three different tutoring conditions:

| Condition | Description |
|-----------|-------------|
| **1 – Character Scaffolded** | AI tutor uses a character-based scaffolding approach |
| **2 – Non-Character Scaffolded** | AI tutor uses scaffolding without a character persona |
| **3 – Direct Chat** | Students interact with the AI in a standard chat format |

Each student completed two tutoring sessions (one per topic), followed by a 5-question quiz and a survey. This project investigates:

- Whether scaffolded tutoring improves quiz performance over direct chat
- Whether engagement behaviors (message pacing, response times, session duration) predict quiz scores
- Which features are most important for predicting student success

---

## Dataset Description

All data files are in the `data/` directory. Below is a summary of each file:

| File | Rows | Description |
|------|------|-------------|
| `users.csv` | 68 | Student IDs, assigned conditions, and emails |
| `sessions.csv` | 136 | One row per student per topic — includes duration, message counts, quiz scores, and difficulty metrics |
| `messages.csv` | 12,371 | Every individual chat message (user and assistant), with timestamps and content length |
| `quiz_questions.csv` | 585 | Individual quiz question responses with difficulty levels (1–3) and correctness |
| `quiz_questions_updated.csv` | 585 | Updated version of quiz questions |
| `survey_responses.csv` | 124 | Post-session survey responses (Likert scales and open text) |
| `sessions_with_engagement_features.csv` | 136 | Sessions enriched with engineered pacing/engagement features |
| `sessions_with_engagement_features_updated.csv` | 136 | Final version of enriched session data used by the ML pipeline |
| `java-session-data.json` | — | Raw session data export from the tutoring platform |

**Key columns in `sessions_with_engagement_features_updated.csv`** (used by the ML models):

- `condition` — tutoring condition (1, 2, or 3)
- `session_type` — topic (arraylist or recursion)
- `duration_seconds` — total session length
- `total_messages`, `user_messages`, `assistant_messages` — message counts
- `avg_response_time`, `median_response_time`, `std_response_time`, `min_response_time`, `max_response_time` — pacing features
- `rapid_response_count`, `rapid_response_pct` — how often the student responded in under 10 seconds
- `avg_difficulty_correct`, `avg_difficulty_incorrect` — average difficulty of questions answered correctly/incorrectly
- `quiz_percentage` — **target variable** (0–100)

---

## Repository Structure

```
JavaTutorDataPredictiveModeling/
│
├── README.md                  ← This file
├── requirements.txt           ← Python dependencies
│
├── data/                      ← All datasets
│   ├── users.csv
│   ├── sessions.csv
│   ├── messages.csv
│   ├── quiz_questions.csv
│   ├── quiz_questions_updated.csv
│   ├── survey_responses.csv
│   ├── sessions_with_engagement_features.csv
│   ├── sessions_with_engagement_features_updated.csv
│   └── java-session-data.json
│
├── analysis/                  ← Jupyter notebooks (run in order)
│   ├── 01_data_feature_analysis.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   ├── 03_baseline_modeling.ipynb
│   ├── 04_predictive_models.ipynb
│   ├── 05_shap_analysis..ipynb
│   └── features.py            ← Helper: computes response-time features from messages
│
├── models/                    ← Python scripts for ML pipelines
│   ├── baseline_models.py     ← Reusable functions for preprocessing, splitting, and evaluating models
│   ├── model_pipeline.py      ← ML pipeline v4 (5 features, uses min_response_time)
│   ├── model_pipeline_avg_resp_time.py  ← ML pipeline v4 variant (5 features, uses avg_response_time)
│   └── pipeline_final.py      ← Final pipeline — supports 5-feature and 16-feature modes
│
├── figures/                   ← Generated plots and visualizations
│   ├── h1_quiz_performance_by_condition.png
│   ├── h2_*.png               ← Hypothesis 2 engagement plots
│   ├── h3_*.png               ← Hypothesis 3 pacing plots
│   ├── performance_by_difficulty.png
│   ├── figure3_efficiency_*.png
│   └── quantitative_behavior_graphic.png
│
└── planning/                  ← Project planning notes
    ├── ml_plan                ← Machine learning phase plan
    ├── sc_plan                ← Statistical comparison plan with hypotheses
    ├── data_dictionary        ← (empty) Data dictionary placeholder
    └── grad_poster_checklist  ← (empty) Poster checklist placeholder
```

---

## Setup & Installation

### Prerequisites

- **Python 3.8 or higher** — [Download Python](https://www.python.org/downloads/)
- **pip** — comes with Python (used to install packages)
- **Jupyter Notebook or JupyterLab** — to run the `.ipynb` files
- **Git** — to clone this repository ([Download Git](https://git-scm.com/downloads))

### Step-by-Step Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/anissawilliams/JavaTutorDataPredictiveModeling.git
   cd JavaTutorDataPredictiveModeling
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional packages needed by the notebooks and scripts:**
   ```bash
   pip install jupyter seaborn shap scipy statsmodels
   ```

5. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   This will open a browser window. Navigate to the `analysis/` folder and open the notebooks in order.

---

## How to Reproduce Results

### Running the Analysis Notebooks

Open and run the notebooks **in order** from the `analysis/` directory:

| Notebook | What It Does |
|----------|--------------|
| `01_data_feature_analysis.ipynb` | Loads raw data, engineers response-time and pacing features, and explores distributions |
| `02_hypothesis_testing.ipynb` | Runs statistical tests comparing quiz performance, engagement, and pacing across conditions |
| `03_baseline_modeling.ipynb` | Trains baseline regression models (Linear Regression, Random Forest, XGBoost) to predict quiz percentage |
| `04_predictive_models.ipynb` | Builds a full classification pipeline (PCA + Logistic Regression, Random Forest, GBM, SVM) |
| `05_shap_analysis..ipynb` | SHAP feature importance analysis (note: this notebook is a stub — full SHAP analysis is in `pipeline_final.py`) |

**To run a notebook:** Open it in Jupyter, then click **Cell → Run All** (or press `Shift+Enter` to run cells one at a time).

### Running the Final ML Pipeline (Script)

The most complete and polished pipeline is `models/pipeline_final.py`. To run it:

```bash
cd models
mkdir -p ../out/16feat    # Create the output directory for 16-feature mode
python pipeline_final.py
```

This script will:
1. Load and preprocess the session data
2. Train 6 regression models (Linear, Ridge, Lasso, Random Forest, Gradient Boosting, SVR)
3. Train 4 classification models (Logistic Regression, Random Forest, GBM, SVM)
4. Generate confusion matrices, scatter plots, residual plots, model comparisons, and feature importance charts
5. Run SHAP interpretability analysis
6. Save predictions to `out/16feat/ml_predictions_16feat_full.csv`

To switch to the **5-feature behavioral-only mode**, open `pipeline_final.py` and change line 40:
```python
FEATURE_MODE = "5"   # Change from "16" to "5"
```
Then create the output directory and re-run:
```bash
mkdir -p ../out/5feat
python pipeline_final.py
```

---

## Analysis Pipeline

The project follows four phases:

### Phase 1: Feature Engineering
Response-time features are computed from the raw `messages.csv` data using `analysis/features.py`. These features capture how quickly and consistently students respond to the AI tutor, including average/median/std response times and a count of "rapid responses" (under 10 seconds).

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
- **5-feature set** (behavioral only): condition, session type, duration, total messages, avg response time
- **16-feature set** (full): adds all pacing metrics, message breakdowns, and difficulty features

### Phase 4: Interpretability
SHAP (SHapley Additive exPlanations) analysis reveals which features contribute most to predictions, run via the `run_shap()` function in `pipeline_final.py`.

---

## Models & Results

The final pipeline (`pipeline_final.py`) trains and evaluates these models:

**Regression Models** (predicting quiz_percentage):
- Linear Regression
- Ridge Regression (α=1.0)
- Lasso Regression (α=1.0)
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression (RBF kernel)

**Classification Models** (predicting High ≥70% vs Low <70%):
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- SVM (RBF kernel)

All models use 5-fold cross-validation and an 80/20 train/test split with `random_state=44` for reproducibility. PCA is applied before model training (up to 5 components).

Generated plots are saved to the `out/` directory (created at runtime) and include confusion matrices, regression scatter plots, residual distributions, model comparison bar charts, feature importance rankings, and SHAP summary/dependence plots.

---

## Code Attribution & Changes

### Original / External Code

- **Scikit-learn, SHAP, XGBoost, PyTorch** — open-source libraries used for modeling and interpretability. No external code templates were copied wholesale.
- The classification pipeline structure in `04_predictive_models.ipynb` was inspired by a **Titanic dataset PCA + Logistic Regression tutorial** (as noted in the notebook docstring).
- `baseline_models.py` provides reusable sklearn Pipeline wrappers written from scratch for this project.

### Updates and Original Contributions

- **Feature engineering** (`features.py`): Custom code to compute session-level response-time features from raw message timestamps.
- **Hypothesis testing** (`02_hypothesis_testing.ipynb`): All statistical analyses (ANOVA, t-tests, correlation, regression) were written for this dataset.
- **Final ML pipeline** (`pipeline_final.py`): Fully original — includes dual regression/classification training, a toggleable 5-feature vs. 16-feature mode, a unified forest/sage/gold visualization theme, SHAP interpretability, and automated plot generation.
- **Iterative model development**: `model_pipeline.py` and `model_pipeline_avg_resp_time.py` represent earlier iterations that explored different feature combinations before converging on the final pipeline.
- All visualizations and the custom color palette were developed specifically for this project's poster presentation.

---

## License

This project was created for academic coursework at the College of Charleston. The datasets contain anonymized student interaction data from an AI Java tutoring study.