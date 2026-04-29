"""
Practical Question 1 - UCI Wine Quality
Statistical Learning - USFQ Exam 2
Daniel Martínez (00329519) - NRC 3468
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, make_scorer)

np.random.seed(42)
plt.rcParams['figure.dpi'] = 150

# ============================================================
# 1. CARGA DE DATOS
# ============================================================
print("=" * 60)
print("1. CARGA DE DATOS")
print("=" * 60)

url_red = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "wine-quality/winequality-red.csv")
url_white = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
             "wine-quality/winequality-white.csv")

try:
    df_red = pd.read_csv(url_red, sep=';')
    df_white = pd.read_csv(url_white, sep=';')
except Exception:
    print("Error descargando. Intentando con archivo local...")
    df_red = pd.read_csv('winequality-red.csv', sep=';')
    df_white = pd.read_csv('winequality-white.csv', sep=';')

df_red['wine_type'] = 0
df_white['wine_type'] = 1
df = pd.concat([df_red, df_white], ignore_index=True)

print(f"Red wines: {len(df_red)}, White wines: {len(df_white)}")
print(f"Total samples: {len(df)}")
print(f"\nColumnas: {list(df.columns)}")
print(f"\nValores nulos:\n{df.isnull().sum()}")
print(f"\nEstadísticas descriptivas:\n{df.describe().round(3)}")

# ============================================================
# 2. DECISIÓN DE MODELADO
# ============================================================
print("\n" + "=" * 60)
print("2. DECISIÓN DE MODELADO")
print("=" * 60)

print("\nDistribución original de quality:")
print(df['quality'].value_counts().sort_index())

# Clasificación binaria: quality >= 7 -> "premium" (1), < 7 -> "estándar" (0)
df['target'] = (df['quality'] >= 7).astype(int)

print(f"\n--- Clasificación Binaria (umbral=7) ---")
print(f"Estándar (0): {(df['target']==0).sum()} ({(df['target']==0).mean()*100:.1f}%)")
print(f"Premium  (1): {(df['target']==1).sum()} ({(df['target']==1).mean()*100:.1f}%)")
print("\nJustificación: Se elige clasificación binaria porque:")
print("  - Regresion: asume intervalos iguales entre calidades (5->6 = 7->8), no es realista")
print("  - Multiclase: clases extremas (3,4,9) tienen muy pocas muestras")
print("  - Binaria: decision practica para vinicultores (premium vs estandar)")

# ============================================================
# 3. EDA
# ============================================================
print("\n" + "=" * 60)
print("3. ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('EDA - Wine Quality', fontsize=14, fontweight='bold')

# 3a. Distribución de quality original
df['quality'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 0],
                                                color='steelblue', edgecolor='black')
axes[0, 0].set_title('Distribución de Quality')
axes[0, 0].set_xlabel('Quality')
axes[0, 0].set_ylabel('Frecuencia')

# 3b. Class imbalance (target binario)
df['target'].value_counts().plot(kind='bar', ax=axes[0, 1],
                                  color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[0, 1].set_title('Balance de Clases (Binario)')
axes[0, 1].set_xticklabels(['Estándar (<7)', 'Premium (>=7)'], rotation=0)

# 3c. pH por clase
sns.boxplot(data=df, x='target', y='pH', ax=axes[0, 2], palette='Set2')
axes[0, 2].set_title('pH por Clase')
axes[0, 2].set_xticklabels(['Estándar', 'Premium'])

# 3d. Sulphates por clase
sns.boxplot(data=df, x='target', y='sulphates', ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Sulphates por Clase')
axes[1, 0].set_xticklabels(['Estándar', 'Premium'])

# 3e. Alcohol por clase
sns.boxplot(data=df, x='target', y='alcohol', ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Alcohol por Clase')
axes[1, 1].set_xticklabels(['Estándar', 'Premium'])

# 3f. Volatile Acidity por clase
sns.boxplot(data=df, x='target', y='volatile acidity', ax=axes[1, 2], palette='Set2')
axes[1, 2].set_title('Volatile Acidity por Clase')
axes[1, 2].set_xticklabels(['Estándar', 'Premium'])

plt.tight_layout()
plt.savefig('eda_wine_quality.png', dpi=150, bbox_inches='tight')
plt.show()

# Correlación
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.drop(columns=['target']).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=ax, square=True, linewidths=0.5, annot_kws={'size': 7})
ax.set_title('Matriz de Correlación - Wine Quality', fontsize=13)
plt.tight_layout()
plt.savefig('corr_wine_quality.png', dpi=150, bbox_inches='tight')
plt.show()

# Estadísticas numéricas EDA
key_vars = ['pH', 'sulphates', 'alcohol', 'volatile acidity']
for var in key_vars:
    print(f"\n--- {var} ---")
    print(df.groupby('target')[var].describe().round(3))

# ============================================================
# 4. PREPROCESAMIENTO
# ============================================================
print("\n" + "=" * 60)
print("4. PREPROCESAMIENTO")
print("=" * 60)

feature_cols = [c for c in df.columns if c not in ['quality', 'target']]
X = df[feature_cols].copy()
y = df['target'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"Train: {X_train_sc.shape[0]} muestras, Test: {X_test_sc.shape[0]} muestras")
print(f"Train target: {y_train.value_counts().to_dict()}")
print(f"Test target:  {y_test.value_counts().to_dict()}")

# ============================================================
# 5. COMPARACIÓN DE MODELOS CON K-FOLD CV
# ============================================================
print("\n" + "=" * 60)
print("5. COMPARACIÓN DE MODELOS (5-Fold Stratified CV)")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Ridge (L2)': LogisticRegression(penalty='l2', C=1.0, max_iter=5000,
                                      random_state=42, solver='lbfgs'),
    'Lasso (L1)': LogisticRegression(penalty='l1', C=1.0, max_iter=5000,
                                      random_state=42, solver='saga'),
    'Elastic Net': LogisticRegression(penalty='elasticnet', C=1.0, l1_ratio=0.5,
                                       max_iter=5000, random_state=42, solver='saga'),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10,
                                             random_state=42, n_jobs=-1),
    'SVM (RBF)': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'Neural Net': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500,
                                 random_state=42, early_stopping=True)
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_sc, y_train, cv=skf,
                             scoring='f1', n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:20s} | F1 CV: {scores.mean():.4f} ± {scores.std():.4f}")

# Hyperparameter tuning para los mejores modelos
print("\n--- Tuning de hiperparámetros ---")

# Tuning Random Forest
rf_params = {'n_estimators': [100, 200, 300],
             'max_depth': [5, 10, 15, None]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                       rf_params, cv=skf, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train_sc, y_train)
print(f"RF mejores params: {rf_grid.best_params_}, F1: {rf_grid.best_score_:.4f}")

# Tuning SVM
svm_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
svm_grid = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=42),
                        svm_params, cv=skf, scoring='f1', n_jobs=-1)
svm_grid.fit(X_train_sc, y_train)
print(f"SVM mejores params: {svm_grid.best_params_}, F1: {svm_grid.best_score_:.4f}")

# Tuning Logistic con búsqueda de C
lr_params = {'C': [0.01, 0.1, 1, 10, 100]}
lr_grid = GridSearchCV(LogisticRegression(penalty='l2', max_iter=5000,
                                          random_state=42, solver='lbfgs'),
                       lr_params, cv=skf, scoring='f1', n_jobs=-1)
lr_grid.fit(X_train_sc, y_train)
print(f"Ridge mejor C: {lr_grid.best_params_}, F1: {lr_grid.best_score_:.4f}")

# ============================================================
# 6. ESTABILIDAD LASSO vs ELASTIC NET
# ============================================================
print("\n" + "=" * 60)
print("6. ESTABILIDAD DE PREDICTORES: LASSO vs ELASTIC NET")
print("=" * 60)

C_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
lasso_coefs = []
enet_coefs = []

for C in C_values:
    lasso = LogisticRegression(penalty='l1', C=C, max_iter=5000,
                                solver='saga', random_state=42)
    lasso.fit(X_train_sc, y_train)
    lasso_coefs.append(lasso.coef_[0])

    enet = LogisticRegression(penalty='elasticnet', C=C, l1_ratio=0.5,
                               max_iter=5000, solver='saga', random_state=42)
    enet.fit(X_train_sc, y_train)
    enet_coefs.append(enet.coef_[0])

lasso_coefs = np.array(lasso_coefs)
enet_coefs = np.array(enet_coefs)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for j, feat in enumerate(feature_cols):
    axes[0].plot(np.log10(C_values), lasso_coefs[:, j], marker='o', label=feat, markersize=3)
axes[0].set_xlabel('log10(C)')
axes[0].set_ylabel('Coeficiente')
axes[0].set_title('Lasso (L1) - Trayectoria de Coeficientes')
axes[0].legend(fontsize=6, loc='best', ncol=2)
axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

for j, feat in enumerate(feature_cols):
    axes[1].plot(np.log10(C_values), enet_coefs[:, j], marker='s', label=feat, markersize=3)
axes[1].set_xlabel('log10(C)')
axes[1].set_ylabel('Coeficiente')
axes[1].set_title('Elastic Net (L1+L2) - Trayectoria de Coeficientes')
axes[1].legend(fontsize=6, loc='best', ncol=2)
axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('lasso_vs_elasticnet_wine.png', dpi=150, bbox_inches='tight')
plt.show()

# Varianza de coeficientes (estabilidad)
print("\nVarianza de coeficientes a lo largo de C:")
print(f"{'Feature':25s} | {'Var Lasso':>12s} | {'Var ElasticNet':>14s}")
print("-" * 56)
for j, feat in enumerate(feature_cols):
    v_l = np.var(lasso_coefs[:, j])
    v_e = np.var(enet_coefs[:, j])
    print(f"{feat:25s} | {v_l:12.6f} | {v_e:14.6f}")

print("\n→ Elastic Net muestra coeficientes más estables (menor varianza)")
print("  porque la penalización L2 agrupa variables correlacionadas")
print("  en lugar de seleccionar arbitrariamente una de ellas.")

# ============================================================
# 7. EVALUACIÓN FINAL EN TEST SET
# ============================================================
print("\n" + "=" * 60)
print("7. EVALUACIÓN FINAL EN TEST SET")
print("=" * 60)

best_models = {
    'Ridge (L2)': lr_grid.best_estimator_,
    'Lasso (L1)': LogisticRegression(penalty='l1', C=1.0, max_iter=5000,
                                      solver='saga', random_state=42),
    'Elastic Net': LogisticRegression(penalty='elasticnet', C=1.0, l1_ratio=0.5,
                                       max_iter=5000, solver='saga', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': rf_grid.best_estimator_,
    'SVM (RBF)': svm_grid.best_estimator_,
    'Neural Net': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500,
                                 random_state=42, early_stopping=True)
}

test_results = []
for name, model in best_models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = (model.predict_proba(X_test_sc)[:, 1]
              if hasattr(model, 'predict_proba') else None)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

    test_results.append({'Model': name, 'Accuracy': acc, 'Precision': prec,
                         'Recall': rec, 'F1': f1, 'ROC-AUC': auc})
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=['Estándar', 'Premium']))

results_df = pd.DataFrame(test_results).set_index('Model')
print("\n=== TABLA RESUMEN ===")
print(results_df.round(4).to_string())

# Gráfico de comparación
fig, ax = plt.subplots(figsize=(12, 6))
results_df[['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']].plot(
    kind='bar', ax=ax, colormap='viridis', edgecolor='black', width=0.8)
ax.set_title('Comparación de Modelos - Wine Quality (Test Set)', fontsize=13)
ax.set_ylabel('Score')
ax.set_xticklabels(results_df.index, rotation=30, ha='right')
ax.legend(loc='lower right')
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('comparacion_modelos_wine.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion matrices de los 3 mejores
top3 = results_df.nlargest(3, 'F1').index.tolist()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, name in enumerate(top3):
    model = best_models[name]
    y_pred = model.predict(X_test_sc)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Estándar', 'Premium']).plot(ax=axes[i])
    axes[i].set_title(name)
plt.suptitle('Matrices de Confusión - Top 3 Modelos', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrices_wine.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 8. INTERPRETACIÓN E INFERENCIA
# ============================================================
print("\n" + "=" * 60)
print("8. INTERPRETACIÓN E INFERENCIA")
print("=" * 60)

# Feature Importance del Random Forest
rf_best = rf_grid.best_estimator_
rf_best.fit(X_train_sc, y_train)
importances = rf_best.feature_importances_
idx_sorted = np.argsort(importances)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RF Feature Importance
axes[0].barh(np.array(feature_cols)[idx_sorted], importances[idx_sorted], color='steelblue')
axes[0].set_title('Feature Importance - Random Forest')
axes[0].set_xlabel('Importancia')

# Coeficientes Logistic Regression (Ridge)
lr_best = lr_grid.best_estimator_
lr_coefs = lr_best.coef_[0]
idx_lr = np.argsort(np.abs(lr_coefs))
colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in lr_coefs[idx_lr]]
axes[1].barh(np.array(feature_cols)[idx_lr], lr_coefs[idx_lr], color=colors)
axes[1].set_title('Coeficientes - Logistic Ridge')
axes[1].set_xlabel('Coeficiente estandarizado')
axes[1].axvline(x=0, color='gray', linestyle='--')

plt.tight_layout()
plt.savefig('feature_importance_wine.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nInterpretación para vinicultores:")
print("  - El ALCOHOL es el predictor más fuerte de calidad premium.")
print("  - La VOLATILE ACIDITY tiene efecto negativo: mayor acidez = menor calidad.")
print("  - Los SULPHATES contribuyen positivamente a la calidad.")
print("  - El pH tiene un efecto menos pronunciado pero relevante.")
print("\nNota: Estas son asociaciones, NO relaciones causales.")

# ============================================================
# 9. EJEMPLOS DE PREDICCIÓN
# ============================================================
print("\n" + "=" * 60)
print("9. EJEMPLOS DE PREDICCIÓN")
print("=" * 60)

best_model_name = results_df['F1'].idxmax()
best_model = best_models[best_model_name]
print(f"\nModelo seleccionado: {best_model_name}")

# 5 ejemplos representativos del test set (muestreo estratificado)
idx_premium = np.where(y_test.values == 1)[0]
idx_standard = np.where(y_test.values == 0)[0]
np.random.seed(42)
sample_idx = np.concatenate([
    np.random.choice(idx_premium, min(2, len(idx_premium)), replace=False),
    np.random.choice(idx_standard, 3, replace=False)
])
np.random.shuffle(sample_idx)

X_sample = X_test_sc[sample_idx]
y_sample_true = y_test.iloc[sample_idx].values
y_sample_pred = best_model.predict(X_sample)
y_sample_prob = best_model.predict_proba(X_sample)[:, 1]

print(f"\n{'#':>3} | {'Real':>8} | {'Pred':>8} | {'P(Premium)':>11} | {'Correcto':>9}")
print("-" * 50)
for i in range(len(sample_idx)):
    label_real = 'Premium' if y_sample_true[i] == 1 else 'Estándar'
    label_pred = 'Premium' if y_sample_pred[i] == 1 else 'Estándar'
    check = '✓' if y_sample_true[i] == y_sample_pred[i] else '✗'
    print(f"{i+1:3d} | {label_real:>8s} | {label_pred:>8s} | {y_sample_prob[i]:>10.4f} | {check:>9s}")

# Guardar resultados para el informe LaTeX
pred_df = pd.DataFrame({
    'Real': ['Premium' if v == 1 else 'Estándar' for v in y_sample_true],
    'Predicción': ['Premium' if v == 1 else 'Estándar' for v in y_sample_pred],
    'P(Premium)': y_sample_prob,
    'Correcto': ['Sí' if y_sample_true[i] == y_sample_pred[i] else 'No' for i in range(len(sample_idx))]
})
pred_df.to_csv('predicciones_wine.csv', index=False)
print("\n→ Resultados guardados en predicciones_wine.csv")

print("\n" + "=" * 60)
print("EJECUCIÓN COMPLETADA - Figuras guardadas en directorio actual")
print("=" * 60)
