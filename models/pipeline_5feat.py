#!/usr/bin/env python3
"""
AI Java Tutor — Final ML Pipeline (v4)
=======================================
Two modeling approaches on the same feature set:
  1. Regression:     predict quiz_percentage (0–100, continuous)
  2. Classification: predict quiz_percentage >= 70 (binary)

Feature sets:
  5-feature  (behavioral only): condition, session_type, duration_seconds,
                                 total_messages, avg_response_time
  16-feature (full set):        all engineered features including
                                 difficulty-weighted quiz features (leakage risk)

Filtered: has_both == True (N=104)
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, \
    GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             mean_squared_error, mean_absolute_error, r2_score)
import warnings

warnings.filterwarnings('ignore')

NAVY = '#0B1D3A';
DEEP = '#132B50';
ACCENT = '#F59E0B';
ACCENT_L = '#FCD34D'
TEAL = '#14B8A6';
TEAL_L = '#5EEAD4';
CORAL = '#F87171';
WHITE = '#FAFBFC'
GRAY4 = '#94A3B8';
GRAY6 = '#475569'

plt.rcParams.update({
    'figure.facecolor': NAVY, 'axes.facecolor': DEEP, 'axes.edgecolor': GRAY6,
    'axes.labelcolor': GRAY4, 'xtick.color': GRAY4, 'ytick.color': GRAY4,
    'text.color': WHITE, 'font.family': 'sans-serif', 'font.size': 11,
    'axes.grid': True, 'grid.color': '#1E3A5F', 'grid.alpha': 0.5,
})

OUT = '../out/5feat/'
DATA = '../data/'

# ═══════════════════════════════════════════════════════════════
# FEATURE SET SELECTION
# Toggle between 5-feature (behavioral only) and 16-feature (full)
# ═══════════════════════════════════════════════════════════════
features = ['condition', 'session_type_enc', 'duration_seconds', 'total_messages', 'avg_response_time']
# features = ['condition', 'session_type_enc', 'duration_seconds', 'total_messages', 'avg_response_time',
#             'user_messages', 'assistant_messages', 'avg_difficulty_correct', 'avg_difficulty_incorrect',
#             'median_response_time', 'std_response_time', 'min_response_time', 'max_response_time',
#             'rapid_response_count', 'rapid_response_pct', 'messages_per_minute']

# Auto-label based on feature count
FEATURE_SET_LABEL = "5-Feature Behavioral-Only (No Leakage)" if len(features) == 5 else f"{len(features)}-Feature Full Set (Includes Difficulty — Leakage Risk)"
FEATURE_SET_SHORT = "5feat_behavioral" if len(features) == 5 else f"{len(features)}feat_full"

# ═══════════════════════════════════════════════════════════════
# LOAD & PREPARE
# ═══════════════════════════════════════════════════════════════
df = pd.read_csv(DATA + 'sessions_with_engagement_features_updated.csv')
df['session_type_enc'] = LabelEncoder().fit_transform(df['session_type'])
df['avg_response_time'] = df['avg_response_time'].fillna(df['avg_response_time'].median())
df = df[df['has_both'] == True].copy()

X = df[features]
y_cont = df['quiz_percentage']  # continuous target
y_bin = (df['quiz_percentage'] >= 70).astype(int)  # binary target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=min(len(features), 5), random_state=44)
X_pca = pca.fit_transform(X_scaled)

print(f"\n{'═'*70}")
print(f"  FEATURE SET: {FEATURE_SET_LABEL}")
print(f"  Features ({len(features)}): {features}")
print(f"  Dataset: {len(df)} sessions (has_both=True filtered)")
print(f"  Quiz % — mean={y_cont.mean():.1f}, std={y_cont.std():.1f}, median={y_cont.median():.0f}")
print(f"  Binary split: Low(<70)={sum(y_bin==0)}, High(≥70)={sum(y_bin==1)}")
print(f"{'═'*70}")

# ═══════════════════════════════════════════════════════════════
# 1. REGRESSION — predict quiz_percentage
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"PART 1: REGRESSION — {FEATURE_SET_LABEL}")
print(f"        predict quiz_percentage (continuous)")
print(f"{'='*70}")

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_pca, y_cont, test_size=0.2, random_state=44)

cv_reg = KFold(5, shuffle=True, random_state=44)

reg_models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1.0)': Ridge(alpha=1.0, random_state=44),
    'Lasso (α=1.0)': Lasso(alpha=1.0, random_state=44),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=44, max_depth=5),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=44, max_depth=3),
    'SVR (RBF)': SVR(kernel='rbf'),
}

print(f"\n{'Model':22s} {'Train R²':>9s} {'Test R²':>8s} {'Test MAE':>9s} {'Test RMSE':>10s} {'CV R² (5-fold)':>16s}")
print("─" * 78)

reg_results = {}
for name, model in reg_models.items():
    model.fit(X_train_r, y_train_r)

    train_pred = model.predict(X_train_r)
    test_pred = model.predict(X_test_r)

    train_r2 = r2_score(y_train_r, train_pred)
    test_r2 = r2_score(y_test_r, test_pred)
    test_mae = mean_absolute_error(y_test_r, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_r, test_pred))

    cv_r2 = cross_val_score(model.__class__(**model.get_params()), X_pca, y_cont, cv=cv_reg, scoring='r2')
    cv_mae = -cross_val_score(model.__class__(**model.get_params()), X_pca, y_cont, cv=cv_reg,
                              scoring='neg_mean_absolute_error')

    reg_results[name] = {
        'train_r2': train_r2, 'test_r2': test_r2,
        'test_mae': test_mae, 'test_rmse': test_rmse,
        'cv_r2_mean': cv_r2.mean(), 'cv_r2_std': cv_r2.std(),
        'cv_mae_mean': cv_mae.mean(), 'cv_mae_std': cv_mae.std(),
        'test_pred': test_pred,
    }

    print(
        f"{name:22s} {train_r2:9.3f} {test_r2:8.3f} {test_mae:9.2f} {test_rmse:10.2f} {cv_r2.mean():7.3f}±{cv_r2.std():.3f}")

print(f"\n  ▲ Results above from: {FEATURE_SET_LABEL}")

# ═══════════════════════════════════════════════════════════════
# 2. CLASSIFICATION — predict High/Low
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"PART 2: CLASSIFICATION — {FEATURE_SET_LABEL}")
print(f"        predict quiz ≥70% (binary High/Low)")
print(f"{'='*70}")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_pca, y_bin, test_size=0.2, random_state=44, stratify=y_bin)

cv_cls = StratifiedKFold(5, shuffle=True, random_state=44)

cls_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=44),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=44, max_depth=5),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=44, max_depth=3),
    'SVM (RBF)': SVC(kernel='rbf', random_state=44, probability=True),
}

print(f"\n{'Model':22s} {'Train':>7s} {'Test':>7s} {'F1':>7s} {'AUC':>7s} {'CV Acc (5-fold)':>16s}")
print("─" * 70)

cls_results = {}
for name, model in cls_models.items():
    model.fit(X_train_c, y_train_c)
    train_acc = accuracy_score(y_train_c, model.predict(X_train_c))
    test_pred = model.predict(X_test_c)
    test_acc = accuracy_score(y_test_c, test_pred)
    test_f1 = f1_score(y_test_c, test_pred, average='weighted')
    test_proba = model.predict_proba(X_test_c)[:, 1]
    auc = roc_auc_score(y_test_c, test_proba)
    cv_acc = cross_val_score(model.__class__(**model.get_params()), X_pca, y_bin, cv=cv_cls, scoring='accuracy')

    cls_results[name] = {
        'train_acc': train_acc, 'test_acc': test_acc,
        'test_f1': test_f1, 'auc': auc,
        'cv_mean': cv_acc.mean(), 'cv_std': cv_acc.std(),
    }
    print(
        f"{name:22s} {train_acc:7.3f} {test_acc:7.3f} {test_f1:7.3f} {auc:7.3f} {cv_acc.mean():7.3f}±{cv_acc.std():.3f}")

print(f"\n  ▲ Results above from: {FEATURE_SET_LABEL}")

# --- CONFUSION MATRICES ---
from sklearn.metrics import confusion_matrix

conf_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=44),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=44, max_depth=5),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=44, max_depth=3),
    'SVM (RBF)': SVC(kernel='rbf', random_state=44, probability=True),
}

for name, model in conf_models.items():
    model.fit(X_train_c, y_train_c)
    y_pred_cm = model.predict(X_test_c)
    cm = confusion_matrix(y_test_c, y_pred_cm)

    fig, ax = plt.subplots(figsize=(5, 4.5))

    sns.heatmap(cm, annot=False, cmap='magma',
                xticklabels=['Low (<70%)', 'High (≥70%)'],
                yticklabels=['Low (<70%)', 'High (≥70%)'],
                ax=ax, linewidths=0,
                vmin=0, vmax=cm.max() + 3,
                cbar=True)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            if val == 0:
                color = GRAY4
            elif val >= cm.max() * 0.55:
                color = '#0B1D3A'
            else:
                color = 'white'
            ax.text(j + 0.5, i + 0.5, str(val),
                    ha='center', va='center',
                    fontsize=22, fontweight='bold', color=color)

    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)

    acc = accuracy_score(y_test_c, y_pred_cm)
    f1 = f1_score(y_test_c, y_pred_cm, average='weighted')
    auc = roc_auc_score(y_test_c, model.predict_proba(X_test_c)[:, 1])

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('Actual Label', fontsize=12)
    ax.set_title(f'Confusion Matrix — {name}', fontsize=13,
                 fontweight='bold', color=ACCENT)
    ax.text(0.5, -0.22,
            f'Acc: {acc:.3f}  |  F1: {f1:.3f}  |  AUC: {auc:.3f}',
            transform=ax.transAxes, ha='center', fontsize=10,
            color=GRAY4, fontfamily='monospace')

    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.tight_layout()
    plt.savefig(f'{OUT}/confusion_{safe_name}_{FEATURE_SET_SHORT}.png',
                dpi=150, bbox_inches='tight', facecolor=NAVY)
    plt.close()
    print(f"  ✓ confusion_{safe_name}_{FEATURE_SET_SHORT}.png")

# ═══════════════════════════════════════════════════════════════
# 3. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"GENERATING VISUALIZATIONS — {FEATURE_SET_LABEL}")
print(f"{'='*70}")

# --- A. Actual vs Predicted scatter (best regression model) ---
best_reg_name = max(reg_results, key=lambda k: reg_results[k]['cv_r2_mean'])
best_reg = reg_models[best_reg_name].__class__(**reg_models[best_reg_name].get_params())
best_reg.fit(X_train_r, y_train_r)
pred_all = best_reg.predict(X_pca)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_cont, pred_all, c=TEAL, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
ax.plot([0, 100], [0, 100], '--', color=CORAL, linewidth=1.5, alpha=0.7, label='Perfect prediction')
ax.set_xlabel('Actual Quiz %', fontsize=13)
ax.set_ylabel('Predicted Quiz %', fontsize=13)
ax.set_title(f'Actual vs Predicted — {best_reg_name}\n({FEATURE_SET_LABEL})', fontsize=14, fontweight='bold', color=ACCENT)
ax.set_xlim(-5, 105)
ax.set_ylim(-5, 105)
ax.legend(facecolor=DEEP, edgecolor=GRAY6, labelcolor=WHITE, fontsize=11)

cv_r2 = reg_results[best_reg_name]['cv_r2_mean']
cv_mae = reg_results[best_reg_name]['cv_mae_mean']
ax.text(5, 92, f'CV R² = {cv_r2:.3f}\nCV MAE = {cv_mae:.1f}pp\n{FEATURE_SET_SHORT}', fontsize=11,
        color=ACCENT_L, fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=DEEP, edgecolor=GRAY6, alpha=0.9))

plt.tight_layout()
plt.savefig(f'{OUT}/regression_scatter_{FEATURE_SET_SHORT}.png', dpi=150, bbox_inches='tight', facecolor=NAVY)
plt.close()
print(f"  ✓ regression_scatter_{FEATURE_SET_SHORT}.png")

# --- B. Residual distribution ---
residuals = y_cont.values - pred_all

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(residuals, bins=20, color=TEAL, alpha=0.75, edgecolor=TEAL_L, linewidth=0.5)
ax1.axvline(x=0, color=ACCENT, linewidth=2, linestyle='--')
ax1.set_xlabel('Residual (Actual − Predicted)', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title(f'Residual Distribution\n({FEATURE_SET_LABEL})', fontsize=13, fontweight='bold', color=ACCENT)
ax1.text(0.95, 0.95, f'Mean: {residuals.mean():.1f}\nStd: {residuals.std():.1f}',
         transform=ax1.transAxes, ha='right', va='top', fontsize=11, color=ACCENT_L,
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=DEEP, edgecolor=GRAY6))

ax2.scatter(pred_all, residuals, c=TEAL, alpha=0.5, s=40, edgecolors='white', linewidth=0.3)
ax2.axhline(y=0, color=ACCENT, linewidth=1.5, linestyle='--')
ax2.set_xlabel('Predicted Quiz %', fontsize=12)
ax2.set_ylabel('Residual', fontsize=12)
ax2.set_title(f'Residuals vs Predicted\n({FEATURE_SET_LABEL})', fontsize=13, fontweight='bold', color=ACCENT)

plt.tight_layout()
plt.savefig(f'{OUT}/regression_residuals_{FEATURE_SET_SHORT}.png', dpi=150, bbox_inches='tight', facecolor=NAVY)
plt.close()
print(f"  ✓ regression_residuals_{FEATURE_SET_SHORT}.png")

# --- C. Dual model comparison ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

reg_names = list(reg_results.keys())
reg_short = ['Linear', 'Ridge', 'Lasso', 'RF', 'GBM', 'SVR']
r2_vals = [reg_results[n]['cv_r2_mean'] for n in reg_names]

x = np.arange(len(reg_names))
bars = ax1.bar(x, r2_vals, 0.6, color=[TEAL if v > 0 else CORAL for v in r2_vals], alpha=0.85)
for bar, v in zip(bars, r2_vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, max(v + 0.01, 0.01), f'{v:.3f}',
             ha='center', va='bottom', fontsize=10, color=WHITE)
ax1.set_xticks(x)
ax1.set_xticklabels(reg_short, fontsize=10)
ax1.set_ylabel('CV R²', fontsize=12)
ax1.set_title(f'Regression Models (CV R²)\n{FEATURE_SET_LABEL}', fontsize=13, fontweight='bold', color=ACCENT)
ax1.axhline(y=0, color=GRAY6, linewidth=1)

cls_names = list(cls_results.keys())
cls_short = ['LogReg', 'RF', 'GBM', 'SVM']
cv_accs = [cls_results[n]['cv_mean'] for n in cls_names]
aucs = [cls_results[n]['auc'] for n in cls_names]

x = np.arange(len(cls_names))
w = 0.3
b1 = ax2.bar(x - w / 2, cv_accs, w, color=TEAL, alpha=0.85, label='CV Accuracy')
b2 = ax2.bar(x + w / 2, aucs, w, color=ACCENT, alpha=0.85, label='Test AUC')
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.2f}',
                 ha='center', va='bottom', fontsize=10, color=WHITE)
ax2.set_xticks(x)
ax2.set_xticklabels(cls_short, fontsize=10)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_ylim(0, 1.1)
ax2.set_title(f'Classification Models\n{FEATURE_SET_LABEL}', fontsize=13, fontweight='bold', color=ACCENT)
ax2.legend(facecolor=DEEP, edgecolor=GRAY6, labelcolor=WHITE, fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUT}/dual_model_comparison_{FEATURE_SET_SHORT}.png', dpi=150, bbox_inches='tight', facecolor=NAVY)
plt.close()
print(f"  ✓ dual_model_comparison_{FEATURE_SET_SHORT}.png")

# --- C2. Regression chart — separate ---
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(reg_names))
bars = ax.bar(x, r2_vals, 0.6, color=[TEAL if v > 0 else CORAL for v in r2_vals], alpha=0.85)
for bar, v in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, max(v + 0.005, 0.005), f'{v:.3f}',
            ha='center', va='bottom', fontsize=11, color=WHITE)
ax.set_xticks(x)
ax.set_xticklabels(reg_short, fontsize=11)
ax.set_ylabel('CV R²', fontsize=12)
ax.set_title(f'Regression Models (CV R²)\n{FEATURE_SET_LABEL}', fontsize=13, fontweight='bold', color=ACCENT)
ax.axhline(y=0, color=GRAY6, linewidth=1)
plt.tight_layout()
plt.savefig(f'{OUT}/regression_comparison_{FEATURE_SET_SHORT}.png', dpi=150, bbox_inches='tight', facecolor=NAVY)
plt.close()
print(f"  ✓ regression_comparison_{FEATURE_SET_SHORT}.png")

# --- C3. Classification chart — separate ---
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(cls_names))
w = 0.3
b1 = ax.bar(x - w / 2, cv_accs, w, color=TEAL, alpha=0.85, label='CV Accuracy')
b2 = ax.bar(x + w / 2, aucs, w, color=ACCENT, alpha=0.85, label='Test AUC')
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=11, color=WHITE)
ax.set_xticks(x)
ax.set_xticklabels(cls_short, fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_title(f'Classification Models — CV Accuracy vs AUC\n{FEATURE_SET_LABEL}', fontsize=13, fontweight='bold', color=ACCENT)
ax.legend(facecolor=DEEP, edgecolor=GRAY6, labelcolor=WHITE, fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT}/classification_comparison_{FEATURE_SET_SHORT}.png', dpi=150, bbox_inches='tight', facecolor=NAVY)
plt.close()
print(f"  ✓ classification_comparison_{FEATURE_SET_SHORT}.png")

# --- D. Feature importance ---
rf_imp = RandomForestRegressor(n_estimators=200, random_state=44, max_depth=5)
rf_imp.fit(X_scaled, y_cont)
imp = pd.Series(rf_imp.feature_importances_, index=features).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = [ACCENT if v >= imp.quantile(0.6) else TEAL for v in imp]
ax.barh(range(len(imp)), imp.values, color=colors, height=0.6)
ax.set_yticks(range(len(imp)))
ax.set_yticklabels([c.replace('_', ' ').title() for c in imp.index], fontsize=11)
ax.set_xlabel('Feature Importance (Gini)', fontsize=12)
ax.set_title(f'Feature Importance — Regression Target\n{FEATURE_SET_LABEL}', fontsize=13, fontweight='bold', color=ACCENT)
plt.tight_layout()
plt.savefig(f'{OUT}/feature_importance_{FEATURE_SET_SHORT}.png', dpi=150, bbox_inches='tight', facecolor=NAVY)
plt.close()
print(f"  ✓ feature_importance_{FEATURE_SET_SHORT}.png")

# ═══════════════════════════════════════════════════════════════
# 4. PREDICTIONS
# ═══════════════════════════════════════════════════════════════
best_cls = SVC(kernel='rbf', random_state=44, probability=True)
best_cls.fit(X_pca, y_bin)

output = pd.DataFrame({
    'user_id': df['user_id'], 'session_type': df['session_type'],
    'condition': df['condition'].astype(int),
    'quiz_percentage_actual': df['quiz_percentage'],
    'predicted_quiz_pct': pred_all.round(2),
    'predicted_label': best_cls.predict(X_pca),
    'predicted_prob_high': best_cls.predict_proba(X_pca)[:, 1].round(4),
    'feature_set': FEATURE_SET_LABEL
})
output.to_csv(f'{OUT}/ml_predictions_{FEATURE_SET_SHORT}.csv', index=None)
print(f"\n  ✓ Predictions saved: ml_predictions_{FEATURE_SET_SHORT}.csv")

# ═══════════════════════════════════════════════════════════════
# 5. SHAP INTERPRETABILITY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SHAP INTERPRETABILITY — {FEATURE_SET_LABEL}")
print(f"{'='*70}")

import shap

rf_raw = rf_imp
X_train_raw = X_scaled

explainer = shap.TreeExplainer(rf_raw)
shap_values = explainer.shap_values(X_train_raw)

print(f"\nTop SHAP Feature Contributions (mean |SHAP|) — {FEATURE_SET_LABEL}:")
shap_importance = np.abs(shap_values).mean(axis=0)
for feat, val in sorted(zip(features, shap_importance), key=lambda x: -x[1]):
    print(f"  {feat:30s}  {val:.4f}")

plt.figure()
shap.summary_plot(shap_values, X_train_raw, feature_names=features, show=False)
plt.tight_layout()
plt.savefig(f"{OUT}/shap_summary_{FEATURE_SET_SHORT}.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ shap_summary_{FEATURE_SET_SHORT}.png")

plt.figure()
shap.summary_plot(shap_values, X_train_raw, feature_names=features, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig(f"{OUT}/shap_bar_{FEATURE_SET_SHORT}.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ shap_bar_{FEATURE_SET_SHORT}.png")

plt.figure()
shap.dependence_plot("avg_response_time", shap_values, X_train_raw,
                     feature_names=features, show=False)
plt.tight_layout()
plt.savefig(f"{OUT}/shap_dependence_avg_response_time_{FEATURE_SET_SHORT}.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ shap_dependence_avg_response_time_{FEATURE_SET_SHORT}.png")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print(f"SUMMARY — {FEATURE_SET_LABEL}")
print(f"{'═'*70}")

best_reg_name = max(reg_results, key=lambda k: reg_results[k]['cv_r2_mean'])
print(f"\n  Best regression:      {best_reg_name}")
print(f"    CV R²:  {reg_results[best_reg_name]['cv_r2_mean']:.3f} ± {reg_results[best_reg_name]['cv_r2_std']:.3f}")
print(f"    CV MAE: {reg_results[best_reg_name]['cv_mae_mean']:.1f} ± {reg_results[best_reg_name]['cv_mae_std']:.1f} percentage points")

best_cls_name = max(cls_results, key=lambda k: cls_results[k]['cv_mean'])
print(f"\n  Best classification:  {best_cls_name}")
print(f"    CV Acc: {cls_results[best_cls_name]['cv_mean']:.3f} ± {cls_results[best_cls_name]['cv_std']:.3f}")
print(f"    AUC:    {cls_results[best_cls_name]['auc']:.3f}")

print(f"\n  Feature set:  {FEATURE_SET_LABEL}")
print(f"  Features ({len(features)}): {features}")
print(f"  N = {len(df)} sessions (has_both filtered)")
print(f"\n{'═'*70}")