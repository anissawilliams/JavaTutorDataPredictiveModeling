#!/usr/bin/env python3

"""
AI Java Tutor — Final ML Pipeline (Refactored)
Unified forest/sage/gold palette
Single toggle for 5‑feature vs 16‑feature
Modular plotting + training functions
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR

import shap
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. FEATURE MODE TOGGLE
# ============================================================
FEATURE_MODE = "16"   # <-- CHANGE HERE ("5" or "16")

# ============================================================
# 2. COLOR PALETTE (FOREST / SAGE / GOLD)
# ============================================================
NAVY_BG     = '#0F172A'
FOREST_DARK = '#2B5233'
FOREST_MED  = '#3A6B4A'
GOLD        = '#BC9000'
SAGE_LIGHT  = '#8FAF97'
WHITE       = '#FFFFFF'
GRAY6       = '#475569'

plt.rcParams.update({
    'figure.facecolor': NAVY_BG,
    'axes.facecolor': FOREST_DARK,
    'axes.edgecolor': GRAY6,
    'axes.labelcolor': SAGE_LIGHT,
    'xtick.color': SAGE_LIGHT,
    'ytick.color': SAGE_LIGHT,
    'text.color': WHITE,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.color': FOREST_MED,
    'grid.alpha': 0.5,
})

# ============================================================
# 3. FEATURE LISTS
# ============================================================
FIVE_FEATURES = [
    'condition', 'session_type_enc', 'duration_seconds',
    'total_messages', 'avg_response_time'
]

SIXTEEN_FEATURES = [
    'condition', 'session_type_enc', 'duration_seconds', 'total_messages',
    'user_messages', 'assistant_messages', 'avg_response_time',
    'median_response_time', 'std_response_time', 'min_response_time',
    'max_response_time', 'rapid_response_count', 'rapid_response_pct',
    'has_both_enc', 'avg_difficulty_correct', 'avg_difficulty_incorrect'
]

features = FIVE_FEATURES if FEATURE_MODE == "5" else SIXTEEN_FEATURES

FEATURE_SET_LABEL = (
    "5‑Feature Behavioral‑Only (No Leakage)"
    if FEATURE_MODE == "5"
    else "16‑Feature Full Set (Includes Difficulty)"
)

FEATURE_SET_SHORT = (
    "5feat_behavioral"
    if FEATURE_MODE == "5"
    else "16feat_full"
)

OUT = f"../out/{FEATURE_MODE}feat/"
DATA = "../data/"

# ============================================================
# 4. DATA LOADING + PREPARATION
# ============================================================

def load_and_prepare_data(features):
    """
    Loads the dataset, encodes categorical fields, fills missing values,
    scales features, applies PCA, and returns all prepared matrices.
    """

    df = pd.read_csv(DATA + "sessions_with_engagement_features_updated.csv")

    # Encode session type
    df["session_type_enc"] = LabelEncoder().fit_transform(df["session_type"])

    # Filter to has_both == True
    df = df[df["has_both"] == True].copy()
    df["has_both_enc"] = df["has_both"].astype(int)

    # Fill missing values for selected features
    df[features] = df[features].fillna(df[features].median())

    # Targets
    y_cont = df["quiz_percentage"]
    y_bin = (df["quiz_percentage"] >= 70).astype(int)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # PCA (max 5 components or number of features)
    pca = PCA(n_components=min(len(features), 5), random_state=44)
    X_pca = pca.fit_transform(X_scaled)

    # Diagnostics
    print("\n" + "═" * 70)
    print(f"  FEATURE SET: {FEATURE_SET_LABEL}")
    print(f"  Features ({len(features)}): {features}")
    print(f"  Dataset: {len(df)} sessions (has_both=True filtered)")
    print(f"  Quiz % — mean={y_cont.mean():.1f}, std={y_cont.std():.1f}, median={y_cont.median():.0f}")
    print(f"  Binary split: Low(<70)={sum(y_bin==0)}, High(≥70)={sum(y_bin==1)}")
    print("═" * 70)

    return df, X_scaled, X_pca, y_cont, y_bin

# ============================================================
# 5. PLOTTING FUNCTIONS (Unified Forest/Sage/Gold Theme)
# ============================================================

from matplotlib.colors import LinearSegmentedColormap

# Custom colormap for confusion matrices
CMAP = LinearSegmentedColormap.from_list(
    "forest_cm",
    [FOREST_DARK, FOREST_MED, GOLD]
)


# ------------------------------------------------------------
# A. CONFUSION MATRIX
# ------------------------------------------------------------
def plot_confusion_matrix(model, X_train, X_test, y_train, y_test, name, out_prefix):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor(FOREST_DARK)

    sns.heatmap(
        cm, annot=False, cmap=CMAP,
        xticklabels=['Low (<70%)', 'High (≥70%)'],
        yticklabels=['Low (<70%)', 'High (≥70%)'],
        ax=ax, linewidths=0,
        vmin=0, vmax=cm.max() + 3,
        cbar=True
    )

    # Text overlay logic
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            if val == 0:
                color = SAGE_LIGHT
            elif val >= cm.max() * 0.55:
                color = NAVY_BG
            else:
                color = WHITE
            ax.text(j + 0.5, i + 0.5, str(val),
                    ha='center', va='center',
                    fontsize=22, fontweight='bold', color=color)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    ax.set_xlabel("Predicted Label", fontsize=12, color=SAGE_LIGHT)
    ax.set_ylabel("Actual Label", fontsize=12, color=SAGE_LIGHT)
    ax.set_title(f"Confusion Matrix — {name}",
                 fontsize=13, fontweight='bold', color=GOLD)

    ax.text(
        0.5, -0.22,
        f"Acc: {acc:.3f}  |  F1: {f1:.3f}  |  AUC: {auc:.3f}",
        transform=ax.transAxes, ha='center',
        fontsize=10, color=SAGE_LIGHT, fontfamily='monospace'
    )

    safe = name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.tight_layout()
    plt.savefig(f"{OUT}/{out_prefix}_confusion_{safe}.png",
                dpi=150, bbox_inches='tight', facecolor=NAVY_BG)
    plt.close()


# ------------------------------------------------------------
# B. REGRESSION SCATTER
# ------------------------------------------------------------
def plot_regression_scatter(y_true, y_pred, model_name, out_prefix):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        y_true, y_pred,
        c=FOREST_MED, alpha=0.6, s=50,
        edgecolors=WHITE, linewidth=0.5
    )

    ax.plot([0, 100], [0, 100], '--',
            color=GOLD, linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Actual Quiz %", fontsize=13)
    ax.set_ylabel("Predicted Quiz %", fontsize=13)
    ax.set_title(
        f"Actual vs Predicted — {model_name}\n({FEATURE_SET_LABEL})",
        fontsize=14, fontweight='bold', color=GOLD
    )

    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig(f"{OUT}/{out_prefix}_reg_scatter.png",
                dpi=150, bbox_inches='tight', facecolor=NAVY_BG)
    plt.close()


# ------------------------------------------------------------
# C. RESIDUAL PLOTS
# ------------------------------------------------------------
def plot_residuals(y_true, y_pred, out_prefix):
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(residuals, bins=20,
             color=FOREST_MED, alpha=0.75,
             edgecolor=FOREST_DARK, linewidth=0.5)
    ax1.axvline(x=0, color=GOLD, linewidth=2, linestyle='--')
    ax1.set_title(f"Residual Distribution\n({FEATURE_SET_LABEL})",
                  fontsize=13, fontweight='bold', color=GOLD)

    # Scatter
    ax2.scatter(
        y_pred, residuals,
        c=FOREST_MED, alpha=0.5, s=40,
        edgecolors=WHITE, linewidth=0.3
    )
    ax2.axhline(y=0, color=GOLD, linewidth=1.5, linestyle='--')
    ax2.set_title(f"Residuals vs Predicted\n({FEATURE_SET_LABEL})",
                  fontsize=13, fontweight='bold', color=GOLD)

    plt.tight_layout()
    plt.savefig(f"{OUT}/{out_prefix}_residuals.png",
                dpi=150, bbox_inches='tight', facecolor=NAVY_BG)
    plt.close()


# ------------------------------------------------------------
# D. MODEL COMPARISON BARS
# ------------------------------------------------------------
def plot_model_comparison(reg_results, cls_results, out_prefix):
    # Regression R²
    reg_names = list(reg_results.keys())
    reg_short = [n.split()[0] for n in reg_names]
    r2_vals = [reg_results[n]['cv_r2_mean'] for n in reg_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(reg_names))
    bars = ax1.bar(
        x, r2_vals, 0.6,
        color=[FOREST_MED if v > 0 else GOLD for v in r2_vals],
        alpha=0.85
    )
    for bar, v in zip(bars, r2_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            max(v + 0.01, 0.01),
            f"{v:.3f}",
            ha='center', va='bottom',
            fontsize=10, color=WHITE
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(reg_short, fontsize=10)
    ax1.set_ylabel("CV R²", fontsize=12)
    ax1.set_title(
        f"Regression Models (CV R²)\n{FEATURE_SET_LABEL}",
        fontsize=13, fontweight='bold', color=GOLD
    )
    ax1.axhline(y=0, color=SAGE_LIGHT, linewidth=1)

    # Classification
    cls_names = list(cls_results.keys())
    cls_short = ['LogReg', 'RF', 'GBM', 'SVM']
    cv_accs = [cls_results[n]['cv_mean'] for n in cls_names]
    aucs = [cls_results[n]['auc'] for n in cls_names]

    x = np.arange(len(cls_names))
    w = 0.3
    b1 = ax2.bar(x - w/2, cv_accs, w,
                 color=FOREST_MED, alpha=0.85, label="CV Accuracy")
    b2 = ax2.bar(x + w/2, aucs, w,
                 color=GOLD, alpha=0.85, label="Test AUC")

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.2f}",
                ha='center', va='bottom',
                fontsize=10, color=WHITE
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels(cls_short, fontsize=10)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.set_title(
        f"Classification Models\n{FEATURE_SET_LABEL}",
        fontsize=13, fontweight='bold', color=GOLD
    )
    ax2.legend(facecolor=FOREST_DARK, edgecolor=SAGE_LIGHT, labelcolor=WHITE)

    plt.tight_layout()
    plt.savefig(f"{OUT}/{out_prefix}_model_comparison.png",
                dpi=150, bbox_inches='tight', facecolor=NAVY_BG)
    plt.close()


# ------------------------------------------------------------
# E. FEATURE IMPORTANCE
# ------------------------------------------------------------
def plot_feature_importance(importances, feature_names, out_prefix):
    imp = pd.Series(importances, index=feature_names).sort_values()

    # Determine threshold for "top" features (top 2)
    top_k = 2
    top_features = imp.nlargest(top_k).index

    colors = [
        GOLD if feat in top_features else FOREST_MED
        for feat in imp.index
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(NAVY_BG)
    ax.set_facecolor(NAVY_BG)

    ax.barh(range(len(imp)), imp.values, color=colors, height=0.6)

    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(
        [c.replace("_", " ").title() for c in imp.index],
        fontsize=11, color=SAGE_LIGHT
    )

    ax.set_xlabel("Feature Importance (Gini)", fontsize=12, color=SAGE_LIGHT)
    ax.set_title(
        f"Feature Importance — Regression Target\n{FEATURE_SET_LABEL}",
        fontsize=13, fontweight='bold', color=GOLD
    )

    for spine in ax.spines.values():
        spine.set_color(SAGE_LIGHT)

    plt.tight_layout()
    plt.savefig(
        f"{OUT}/{out_prefix}_feature_importance.png",
        dpi=150, bbox_inches='tight', facecolor=NAVY_BG
    )
    plt.close()

# ============================================================
# 6. MODEL TRAINING FUNCTIONS
# ============================================================

# ------------------------------------------------------------
# A. REGRESSION TRAINING
# ------------------------------------------------------------
def train_regression_models(X_pca, y_cont):
    """
    Trains all regression models, computes metrics, and returns a dict
    with CV results, predictions, and model objects.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_cont, test_size=0.2, random_state=44
    )

    cv = KFold(5, shuffle=True, random_state=44)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (α=1.0)": Ridge(alpha=1.0, random_state=44),
        "Lasso (α=1.0)": Lasso(alpha=1.0, random_state=44),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=44, max_depth=5
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=44, max_depth=3
        ),
        "SVR (RBF)": SVR(kernel="rbf"),
    }

    results = {}

    print("\n" + "=" * 70)
    print(f"PART 1: REGRESSION — {FEATURE_SET_LABEL}")
    print("Predict quiz_percentage (continuous)")
    print("=" * 70)
    print(f"\n{'Model':22s} {'Train R²':>9s} {'Test R²':>8s} "
          f"{'MAE':>9s} {'RMSE':>10s} {'CV R² (5-fold)':>16s}")
    print("─" * 78)

    for name, model in models.items():
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        mae = mean_absolute_error(y_test, test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        cv_r2 = cross_val_score(
            model.__class__(**model.get_params()),
            X_pca, y_cont, cv=cv, scoring="r2"
        )
        cv_mae = -cross_val_score(
            model.__class__(**model.get_params()),
            X_pca, y_cont, cv=cv, scoring="neg_mean_absolute_error"
        )

        results[name] = {
            "model": model,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "mae": mae,
            "rmse": rmse,
            "cv_r2_mean": cv_r2.mean(),
            "cv_r2_std": cv_r2.std(),
            "cv_mae_mean": cv_mae.mean(),
            "cv_mae_std": cv_mae.std(),
            "test_pred": test_pred,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        print(
            f"{name:22s} {train_r2:9.3f} {test_r2:8.3f} "
            f"{mae:9.2f} {rmse:10.2f} "
            f"{cv_r2.mean():7.3f}±{cv_r2.std():.3f}"
        )

    return results


# ------------------------------------------------------------
# B. CLASSIFICATION TRAINING
# ------------------------------------------------------------
def train_classification_models(X_pca, y_bin):
    """
    Trains all classification models, computes metrics, and returns a dict
    with CV results, predictions, and model objects.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_bin, test_size=0.2, random_state=44, stratify=y_bin
    )

    cv = StratifiedKFold(5, shuffle=True, random_state=44)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=44),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=44, max_depth=5
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=44, max_depth=3
        ),
        "SVM (RBF)": SVC(kernel="rbf", random_state=44, probability=True),
    }

    results = {}

    print("\n" + "=" * 70)
    print(f"PART 2: CLASSIFICATION — {FEATURE_SET_LABEL}")
    print("Predict quiz ≥70% (binary High/Low)")
    print("=" * 70)
    print(f"\n{'Model':22s} {'Train':>7s} {'Test':>7s} "
          f"{'F1':>7s} {'AUC':>7s} {'CV Acc (5-fold)':>16s}")
    print("─" * 70)

    for name, model in models.items():
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        f1 = f1_score(y_test, test_pred, average="weighted")
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        cv_acc = cross_val_score(
            model.__class__(**model.get_params()),
            X_pca, y_bin, cv=cv, scoring="accuracy"
        )

        results[name] = {
            "model": model,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "test_f1": f1,
            "auc": auc,
            "cv_mean": cv_acc.mean(),
            "cv_std": cv_acc.std(),
            "test_pred": test_pred,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        print(
            f"{name:22s} {train_acc:7.3f} {test_acc:7.3f} "
            f"{f1:7.3f} {auc:7.3f} "
            f"{cv_acc.mean():7.3f}±{cv_acc.std():.3f}"
        )

    return results

# ============================================================
# 7. PIPELINE RUNNER
# ============================================================

def run_pipeline():
    print("\n" + "="*70)
    print(f"RUNNING PIPELINE — {FEATURE_SET_LABEL}")
    print("="*70)

    # ----------------------------------------
    # Load + prepare data
    # ----------------------------------------
    df, X_scaled, X_pca, y_cont, y_bin = load_and_prepare_data(features)

    # ----------------------------------------
    # Train regression models
    # ----------------------------------------
    reg_results = train_regression_models(X_pca, y_cont)

    # Best regression model by CV R²
    best_reg_name = max(reg_results, key=lambda k: reg_results[k]["cv_r2_mean"])
    best_reg = reg_results[best_reg_name]["model"]
    y_pred_reg = best_reg.predict(X_pca)

    # ----------------------------------------
    # Train classification models
    # ----------------------------------------
    cls_results = train_classification_models(X_pca, y_bin)

    # Best classification model by CV accuracy
    best_cls_name = max(cls_results, key=lambda k: cls_results[k]["cv_mean"])
    best_cls = cls_results[best_cls_name]["model"]

    # ----------------------------------------
    # Generate plots
    # ----------------------------------------
    print("\n" + "="*70)
    print(f"GENERATING VISUALIZATIONS — {FEATURE_SET_LABEL}")
    print("="*70)

    # A. Confusion matrices for all classifiers
    for name, result in cls_results.items():
        model = result["model"]
        X_train = result["X_train"]
        X_test = result["X_test"]
        y_train = result["y_train"]
        y_test = result["y_test"]

        plot_confusion_matrix(
            model, X_train, X_test, y_train, y_test,
            name=name,
            out_prefix=FEATURE_SET_SHORT
        )

    # B. Regression scatter
    plot_regression_scatter(
        y_true=y_cont,
        y_pred=y_pred_reg,
        model_name=best_reg_name,
        out_prefix=FEATURE_SET_SHORT
    )

    # C. Residuals
    plot_residuals(
        y_true=y_cont,
        y_pred=y_pred_reg,
        out_prefix=FEATURE_SET_SHORT
    )

    # D. Model comparison (reg + cls)
    plot_model_comparison(
        reg_results=reg_results,
        cls_results=cls_results,
        out_prefix=FEATURE_SET_SHORT
    )

    # E. Feature importance (Random Forest Regressor)
    rf_imp = RandomForestRegressor(
        n_estimators=200, random_state=44, max_depth=5
    )
    rf_imp.fit(X_scaled, y_cont)

    plot_feature_importance(
        importances=rf_imp.feature_importances_,
        feature_names=features,
        out_prefix=FEATURE_SET_SHORT
    )

    # ----------------------------------------
    # Save predictions
    # ----------------------------------------
    preds = pd.DataFrame({
        "user_id": df["user_id"],
        "session_type": df["session_type"],
        "condition": df["condition"].astype(int),
        "quiz_percentage_actual": df["quiz_percentage"],
        "predicted_quiz_pct": y_pred_reg.round(2),
        "predicted_label": best_cls.predict(X_pca),
        "predicted_prob_high": best_cls.predict_proba(X_pca)[:, 1].round(4),
        "feature_set": FEATURE_SET_LABEL
    })

    preds.to_csv(f"{OUT}/ml_predictions_{FEATURE_SET_SHORT}.csv", index=False)
    print(f"\n  ✓ Predictions saved: ml_predictions_{FEATURE_SET_SHORT}.csv")

    return df, reg_results, cls_results, best_reg_name, best_cls_name
# ============================================================
# 8. SHAP INTERPRETABILITY (UPDATED PALETTE)
# ============================================================

from matplotlib.colors import LinearSegmentedColormap

def shap_summary_plot(shap_values, X_scaled, features, out_prefix):
    shap_cmap = LinearSegmentedColormap.from_list(
        "shap_forest",
        [FOREST_MED, GOLD]
    )

    plt.figure()
    fig = plt.gcf()
    fig.patch.set_facecolor(NAVY_BG)

    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=features,
        cmap=shap_cmap,
        show=False
    )

    # Force text to white
    for text in plt.gca().texts:
        text.set_color("white")
    plt.xticks(color="white")
    plt.yticks(color="white")

    plt.tight_layout()
    plt.savefig(
        f"{OUT}/shap_summary_{out_prefix}.png",
        dpi=200, bbox_inches='tight', facecolor=NAVY_BG
    )
    plt.close()



def shap_bar_plot(shap_values, X_scaled, features, out_prefix):
    mean_abs = np.abs(shap_values).mean(axis=0)
    series = pd.Series(mean_abs, index=features).sort_values()

    top_k = 2
    top_features = series.nlargest(top_k).index

    colors = [
        GOLD if feat in top_features else FOREST_MED
        for feat in series.index
    ]

    plt.figure()
    fig = plt.gcf()
    fig.patch.set_facecolor(NAVY_BG)

    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=features,
        plot_type='bar',
        color=colors,
        show=False
    )

    # Force text to white
    for text in plt.gca().texts:
        text.set_color("white")
    plt.xticks(color="white")
    plt.yticks(color="white")

    plt.tight_layout()
    plt.savefig(
        f"{OUT}/shap_bar_{out_prefix}.png",
        dpi=200, bbox_inches='tight', facecolor=NAVY_BG
    )
    plt.close()



def shap_dependence_plot(shap_values, X_scaled, features, feature_name, out_prefix):
    shap_cmap = LinearSegmentedColormap.from_list(
        "shap_forest",
        [FOREST_MED, GOLD]
    )

    plt.figure()
    fig = plt.gcf()
    fig.patch.set_facecolor(NAVY_BG)

    shap.dependence_plot(
        feature_name,
        shap_values,
        X_scaled,
        feature_names=features,
        cmap=shap_cmap,
        show=False
    )

    # Force text to white
    for text in plt.gca().texts:
        text.set_color("white")
    plt.xticks(color="white")
    plt.yticks(color="white")

    plt.tight_layout()
    plt.savefig(
        f"{OUT}/shap_dependence_{feature_name}_{out_prefix}.png",
        dpi=200, bbox_inches='tight', facecolor=NAVY_BG
    )
    plt.close()



def run_shap(X_scaled, y_cont, features, out_prefix):
    """
    Runs SHAP on a RandomForestRegressor trained on scaled features.
    Saves summary, bar, and dependence plots using forest/sage/gold theme.
    """

    print("\n" + "="*70)
    print(f"SHAP INTERPRETABILITY — {FEATURE_SET_LABEL}")
    print("="*70)

    # Train RF for SHAP
    rf = RandomForestRegressor(
        n_estimators=200, random_state=44, max_depth=5
    )
    rf.fit(X_scaled, y_cont)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)

    # Print top features
    shap_importance = np.abs(shap_values).mean(axis=0)
    print("\nTop SHAP Feature Contributions (mean |SHAP|):")
    for feat, val in sorted(zip(features, shap_importance), key=lambda x: -x[1]):
        print(f"  {feat:30s}  {val:.4f}")

    # Summary plot
    shap_summary_plot(shap_values, X_scaled, features, out_prefix)

    # Bar plot
    shap_bar_plot(shap_values, X_scaled, features, out_prefix)

    # Dependence plot (only if feature exists)
    if "avg_response_time" in features:
        shap_dependence_plot(
            shap_values, X_scaled, features,
            feature_name="avg_response_time",
            out_prefix=out_prefix
        )

# ============================================================
# 9. SUMMARY + MAIN ENTRY POINT
# ============================================================

def print_summary(reg_results, cls_results, best_reg_name, best_cls_name):
    print("\n" + "═"*70)
    print(f"SUMMARY — {FEATURE_SET_LABEL}")
    print("═"*70)

    # Best regression
    br = reg_results[best_reg_name]
    print(f"\n  Best regression:      {best_reg_name}")
    print(f"    CV R²:  {br['cv_r2_mean']:.3f} ± {br['cv_r2_std']:.3f}")
    print(f"    CV MAE: {br['cv_mae_mean']:.1f} ± {br['cv_mae_std']:.1f} percentage points")

    # Best classification
    bc = cls_results[best_cls_name]
    print(f"\n  Best classification:  {best_cls_name}")
    print(f"    CV Acc: {bc['cv_mean']:.3f} ± {bc['cv_std']:.3f}")
    print(f"    AUC:    {bc['auc']:.3f}")

    print(f"\n  Feature set:  {FEATURE_SET_LABEL}")
    print(f"  Features ({len(features)}): {features}")
    print("═"*70)


# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    df, reg_results, cls_results, best_reg_name, best_cls_name = run_pipeline()

    # Run SHAP
    # (We reload scaled data because SHAP uses X_scaled, not PCA)
    df2, X_scaled, X_pca2, y_cont2, y_bin2 = load_and_prepare_data(features)
    run_shap(X_scaled, y_cont2, features, FEATURE_SET_SHORT)

    # Print summary
    print_summary(reg_results, cls_results, best_reg_name, best_cls_name)
