"""
Practical Question 2 - UCI Bank Marketing
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve,
                             precision_recall_curve, PrecisionRecallDisplay)
import urllib.request, zipfile, io, os

np.random.seed(42)
plt.rcParams['figure.dpi'] = 150


# 1. ------------CARGA DE DATOS-------------

print("=" * 60)
print("1. CARGA DE DATOS")
print("=" * 60)

csv_path = 'bank-additional-full.csv'
if not os.path.exists(csv_path):
    try:
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "00222/bank-additional.zip")
        print("Descargando Bank Marketing dataset...")
        resp = urllib.request.urlopen(url)
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        z.extractall('.')
        csv_path = 'bank-additional/bank-additional-full.csv'
    except Exception as e:
        print(f"Error: {e}")
        print("Descarga manual: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing")
        raise

if not os.path.exists(csv_path):
    csv_path = 'bank-additional/bank-additional-full.csv'

df = pd.read_csv(csv_path, sep=';')
print(f"Shape: {df.shape}")
print(f"Columnas: {list(df.columns)}")

# Target encoding
df['target'] = (df['y'] == 'yes').astype(int)


# 2. ------------EDA-------------

print("\n" + "=" * 60)
print("2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("=" * 60)

# 2a. Class imbalance
print(f"\n--- Balance de Clases ---")
print(f"No suscripción: {(df['target']==0).sum()} ({(df['target']==0).mean()*100:.1f}%)")
print(f"Suscripción:    {(df['target']==1).sum()} ({(df['target']==1).mean()*100:.1f}%)")

# 2b. Categorical feature levels
cat_cols = df.select_dtypes(include='object').columns.drop('y')
print(f"\n--- Variables Categóricas ---")
for col in cat_cols:
    n_levels = df[col].nunique()
    unknown_pct = (df[col] == 'unknown').mean() * 100
    print(f"  {col:15s}: {n_levels:3d} niveles | 'unknown': {unknown_pct:.1f}%")

# 2c. Missing-like values ("unknown")
unknown_counts = {}
for col in cat_cols:
    n = (df[col] == 'unknown').sum()
    if n > 0:
        unknown_counts[col] = n
print(f"\n--- Valores 'unknown' (missing-like) ---")
for col, n in sorted(unknown_counts.items(), key=lambda x: -x[1]):
    print(f"  {col:15s}: {n:6d} ({n/len(df)*100:.1f}%)")

# 2d. Campaign/contact variables
print(f"\n--- Variables de Campaña/Contacto ---")
campaign_vars = ['campaign', 'pdays', 'previous', 'poutcome', 'contact', 'duration']
for var in campaign_vars:
    if pd.api.types.is_numeric_dtype(df[var]):
        print(f"  {var}: mean={df[var].mean():.2f}, median={df[var].median():.2f}, "
              f"min={df[var].min()}, max={df[var].max()}")
    else:
        print(f"  {var}: {df[var].value_counts().to_dict()}")

# Figuras EDA
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('EDA - Bank Marketing', fontsize=14, fontweight='bold')

# Class imbalance
df['target'].value_counts().plot(kind='bar', ax=axes[0, 0],
                                  color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[0, 0].set_title('Balance de Clases')
axes[0, 0].set_xticklabels(['No', 'Sí'], rotation=0)

# Age distribution by target
for t in [0, 1]:
    axes[0, 1].hist(df[df['target'] == t]['age'], bins=30, alpha=0.6,
                     label=f"{'No' if t==0 else 'Sí'}", edgecolor='black')
axes[0, 1].set_title('Edad por Clase')
axes[0, 1].legend()

# Duration distribution
for t in [0, 1]:
    axes[0, 2].hist(df[df['target'] == t]['duration'], bins=50, alpha=0.6,
                     label=f"{'No' if t==0 else 'Sí'}", edgecolor='black')
axes[0, 2].set_title('Duración de Llamada (LEAKAGE)')
axes[0, 2].legend()

# Campaign contacts
axes[1, 0].hist(df['campaign'], bins=30, color='steelblue', edgecolor='black')
axes[1, 0].set_title('Nº Contactos en Campaña')

# Job distribution
job_counts = df.groupby('job')['target'].mean().sort_values()
job_counts.plot(kind='barh', ax=axes[1, 1], color='steelblue')
axes[1, 1].set_title('Tasa de Suscripción por Trabajo')
axes[1, 1].set_xlabel('Tasa')

# Previous outcome
pout = df.groupby('poutcome')['target'].mean().sort_values()
pout.plot(kind='bar', ax=axes[1, 2], color='steelblue', edgecolor='black')
axes[1, 2].set_title('Tasa de Suscripción por Resultado Previo')
axes[1, 2].set_xticklabels(pout.index, rotation=30, ha='right')

plt.tight_layout()
plt.savefig('eda_bank_marketing.png', dpi=150, bbox_inches='tight')
plt.show()

# Correlación numérica
num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols = [c for c in num_cols if c != 'target']
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[num_cols + ['target']].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title('Correlación - Variables Numéricas', fontsize=13)
plt.tight_layout()
plt.savefig('corr_bank_marketing.png', dpi=150, bbox_inches='tight')
plt.show()


# 3. ------------PIPELINE DE PREPROCESAMIENTO-------------

print("\n" + "=" * 60)
print("3. PIPELINE DE PREPROCESAMIENTO")
print("=" * 60)

# LEAKAGE CHECK: Remover 'duration' (solo se conoce después de la llamada)
print("\n⚠ LEAKAGE: Removiendo 'duration' - solo conocida post-llamada")
print("  En producción, no se puede usar para predecir antes del contacto.")

features_to_drop = ['y', 'target', 'duration']
feature_cols = [c for c in df.columns if c not in features_to_drop]

# Reemplazar 'unknown' con NaN para imputación
df_clean = df.copy()
for col in cat_cols:
    df_clean[col] = df_clean[col].replace('unknown', np.nan)

X = df_clean[feature_cols]
y = df_clean['target']

# Identificar columnas
num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = X.select_dtypes(include='object').columns.tolist()

print(f"Features numéricas ({len(num_features)}): {num_features}")
print(f"Features categóricas ({len(cat_features)}): {cat_features}")

# Pipeline de preprocesamiento
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Train/Test split ESTRATIFICADO
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Obtener nombres de features procesadas
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_feature_names = cat_encoder.get_feature_names_out(cat_features).tolist()
all_feature_names = num_features + cat_feature_names

print(f"\nTrain: {X_train_proc.shape}, Test: {X_test_proc.shape}")
print(f"Total features procesadas: {X_train_proc.shape[1]}")


# 4. ------------COMPARACIÓN DE MODELOS (STRATIFIED CV)-------------

print("\n" + "=" * 60)
print("4. COMPARACIÓN DE MODELOS (5-Fold Stratified CV)")
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
    'SVM (Linear)': SVC(kernel='linear', C=1.0, probability=True, random_state=42,
                        max_iter=5000),
    'Neural Net': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                 random_state=42, early_stopping=True)
}

scoring_metrics = ['roc_auc', 'f1', 'recall', 'precision']
cv_results = {}

for name, model in models.items():
    results = {}
    for metric in scoring_metrics:
        scores = cross_val_score(model, X_train_proc, y_train, cv=skf,
                                 scoring=metric, n_jobs=-1)
        results[metric] = (scores.mean(), scores.std())
    cv_results[name] = results
    print(f"\n{name}:")
    for m, (mean, std) in results.items():
        print(f"  {m:12s}: {mean:.4f} ± {std:.4f}")

# Tuning mejores modelos
print("\n--- Tuning de hiperparámetros ---")

rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                       rf_params, cv=skf, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train_proc, y_train)
print(f"RF best: {rf_grid.best_params_}, AUC: {rf_grid.best_score_:.4f}")

lr_params = {'C': [0.01, 0.1, 1, 10]}
lr_grid = GridSearchCV(LogisticRegression(penalty='l2', max_iter=5000,
                                          random_state=42),
                       lr_params, cv=skf, scoring='roc_auc', n_jobs=-1)
lr_grid.fit(X_train_proc, y_train)
print(f"Ridge best C: {lr_grid.best_params_}, AUC: {lr_grid.best_score_:.4f}")


# 5. ------------EVALUACIÓN FINAL EN TEST SET-------------

print("\n" + "=" * 60)
print("5. EVALUACIÓN FINAL EN TEST SET")
print("=" * 60)

final_models = {
    'Ridge (L2)': lr_grid.best_estimator_,
    'Lasso (L1)': LogisticRegression(penalty='l1', C=1.0, max_iter=5000,
                                      solver='saga', random_state=42),
    'Elastic Net': LogisticRegression(penalty='elasticnet', C=1.0, l1_ratio=0.5,
                                       max_iter=5000, solver='saga', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': rf_grid.best_estimator_,
    'SVM (Linear)': SVC(kernel='linear', C=1.0, probability=True, random_state=42,
                        max_iter=5000),
    'Neural Net': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                 random_state=42, early_stopping=True)
}

test_results = []
model_preds = {}

for name, model in final_models.items():
    model.fit(X_train_proc, y_train)
    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)

    test_results.append({'Model': name, 'Accuracy': acc, 'Precision': prec,
                         'Recall': rec, 'F1': f1, 'ROC-AUC': auc_roc,
                         'PR-AUC': auc_pr})
    model_preds[name] = (y_pred, y_prob)

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=['No', 'Sí']))

results_df = pd.DataFrame(test_results).set_index('Model')
print("\n=== TABLA RESUMEN ===")
print(results_df.round(4).to_string())

# ROC Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, (_, y_prob) in model_preds.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].set_title('ROC Curves')
axes[0].set_xlabel('FPR')
axes[0].set_ylabel('TPR')
axes[0].legend(fontsize=7)

# PR Curves
for name, (_, y_prob) in model_preds.items():
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    axes[1].plot(rec_arr, prec_arr, label=f'{name} (AP={ap:.3f})')
axes[1].set_title('Precision-Recall Curves')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(fontsize=7)

plt.tight_layout()
plt.savefig('roc_pr_curves_bank.png', dpi=150, bbox_inches='tight')
plt.show()

# Comparación de métricas
fig, ax = plt.subplots(figsize=(12, 6))
results_df[['ROC-AUC', 'PR-AUC', 'Recall', 'Precision', 'F1']].plot(
    kind='bar', ax=ax, colormap='viridis', edgecolor='black', width=0.8)
ax.set_title('Comparación de Modelos - Bank Marketing (Test Set)', fontsize=13)
ax.set_ylabel('Score')
ax.set_xticklabels(results_df.index, rotation=30, ha='right')
ax.legend(loc='lower right')
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('comparacion_modelos_bank.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion matrices con business threshold
print("\n--- Recall a Precision Fija ---")
best_name = results_df['ROC-AUC'].idxmax()
best_prob = model_preds[best_name][1]

prec_arr, rec_arr, thresholds = precision_recall_curve(y_test, best_prob)
# Recall at precision >= 0.50
mask = prec_arr >= 0.50
if mask.any():
    best_recall_at_prec = rec_arr[mask].max()
    idx = np.argmax(rec_arr[mask])
    print(f"Modelo: {best_name}")
    print(f"Recall @ Precision ≥ 0.50: {best_recall_at_prec:.4f}")

# Business threshold confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
thresholds_biz = [0.3, 0.5, 0.7]
for i, thr in enumerate(thresholds_biz):
    y_pred_thr = (best_prob >= thr).astype(int)
    cm = confusion_matrix(y_test, y_pred_thr)
    ConfusionMatrixDisplay(cm, display_labels=['No', 'Sí']).plot(ax=axes[i])
    rec = recall_score(y_test, y_pred_thr)
    prec = precision_score(y_test, y_pred_thr, zero_division=0)
    axes[i].set_title(f'Umbral={thr} | Rec={rec:.2f}, Prec={prec:.2f}')
plt.suptitle(f'Matrices de Confusión a Distintos Umbrales ({best_name})', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_thresholds_bank.png', dpi=150, bbox_inches='tight')
plt.show()


# 6. ------------FEATURE IMPORTANCE / COEFICIENTES-------------

print("\n" + "=" * 60)
print("6. FEATURE IMPORTANCE / COEFICIENTES")
print("=" * 60)

# Random Forest importance
rf_best = rf_grid.best_estimator_
importances = rf_best.feature_importances_
top_n = 15
idx_top = np.argsort(importances)[-top_n:]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].barh(np.array(all_feature_names)[idx_top], importances[idx_top], color='steelblue')
axes[0].set_title('Top 15 Feature Importance - Random Forest')
axes[0].set_xlabel('Importancia')

# Logistic coefs
lr_model = lr_grid.best_estimator_
lr_coefs = lr_model.coef_[0]
idx_lr_top = np.argsort(np.abs(lr_coefs))[-top_n:]
colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in lr_coefs[idx_lr_top]]
axes[1].barh(np.array(all_feature_names)[idx_lr_top], lr_coefs[idx_lr_top], color=colors)
axes[1].set_title('Top 15 Coeficientes - Logistic Ridge')
axes[1].set_xlabel('Coeficiente')
axes[1].axvline(x=0, color='gray', linestyle='--')

plt.tight_layout()
plt.savefig('feature_importance_bank.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nInterpretación (asociacional, NO causal):")
print("  - Variables socioeconómicas (euribor3m, nr.employed) son las más")
print("    influyentes: reflejan contexto macroeconómico, no causalidad directa.")
print("  - poutcome_success: clientes contactados previamente con éxito")
print("    tienen mayor probabilidad de suscribirse.")
print("  - contact_telephone: canal de contacto asociado con menor conversión.")
print("  - month: ciertos meses muestran estacionalidad en suscripciones.")
print("\n  ⚠ NO se puede afirmar que llamar más veces 'causa' suscripción.")


# 7. ------------ESTRATEGIA DE CONTACTO-------------

print("\n" + "=" * 60)
print("7. RECOMENDACIÓN DE ESTRATEGIA DE CONTACTO")
print("=" * 60)

print("""
Bajo capacidad limitada de call center, se recomienda:

1. PRIORIZACIÓN POR SCORE: Ordenar clientes por P(suscripción) descendente
   y contactar solo al top-K según capacidad disponible.

2. UMBRAL ÓPTIMO: Usar umbral de 0.30 para maximizar recall sin
   desperdiciar demasiados recursos en falsos positivos.

3. SEGMENTACIÓN PRIORITARIA:
   - Alta prioridad: Clientes con poutcome=success (contacto previo exitoso)
   - Media prioridad: Clientes jóvenes/jubilados con alto score
   - Baja prioridad: Clientes con muchos contactos previos sin éxito

4. TIMING: Concentrar llamadas en meses con mayor tasa histórica
   de conversión (evitar períodos de baja respuesta).

5. CAPACIDAD: Con ~12% de tasa base, contactando al top-20% por score
   se espera capturar >50% de las suscripciones potenciales.
""")

# Ejemplo: curva de captura (lift)
best_prob_sorted = np.sort(best_prob)[::-1]
y_test_sorted = y_test.values[np.argsort(best_prob)[::-1]]
cumulative_capture = np.cumsum(y_test_sorted) / y_test_sorted.sum()
pct_contacted = np.arange(1, len(y_test_sorted) + 1) / len(y_test_sorted)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(pct_contacted * 100, cumulative_capture * 100, color='steelblue', linewidth=2)
ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Aleatorio')
ax.set_xlabel('% de Clientes Contactados')
ax.set_ylabel('% de Suscripciones Capturadas')
ax.set_title(f'Curva de Captura (Lift) - {best_name}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lift_curve_bank.png', dpi=150, bbox_inches='tight')
plt.show()


# 8. ------------EJEMPLOS DE PREDICCIÓN-------------

print("\n" + "=" * 60)
print("8. EJEMPLOS DE PREDICCIÓN")
print("=" * 60)

best_model_final = final_models[best_name]
# 5 ejemplos representativos (muestreo estratificado de ambas clases)
idx_yes = np.where(y_test.values == 1)[0]
idx_no = np.where(y_test.values == 0)[0]
np.random.seed(42)
sample_idx = np.concatenate([
    np.random.choice(idx_yes, min(2, len(idx_yes)), replace=False),
    np.random.choice(idx_no, 3, replace=False)
])
np.random.shuffle(sample_idx)

X_sample = X_test_proc[sample_idx]
y_true = y_test.iloc[sample_idx].values
y_pred_s = best_model_final.predict(X_sample)
y_prob_s = best_model_final.predict_proba(X_sample)[:, 1]

print(f"\nModelo: {best_name}")
print(f"{'#':>3} | {'Real':>6} | {'Pred':>6} | {'P(Sí)':>8} | {'Correcto':>9}")
print("-" * 45)
for i in range(len(sample_idx)):
    real = 'Sí' if y_true[i] == 1 else 'No'
    pred = 'Sí' if y_pred_s[i] == 1 else 'No'
    check = '✓' if y_true[i] == y_pred_s[i] else '✗'
    print(f"{i+1:3d} | {real:>6s} | {pred:>6s} | {y_prob_s[i]:>7.4f} | {check:>9s}")

# Guardar resultados para el informe LaTeX
pred_df = pd.DataFrame({
    'Real': ['Sí' if v == 1 else 'No' for v in y_true],
    'Predicción': ['Sí' if v == 1 else 'No' for v in y_pred_s],
    'P(Suscripción)': y_prob_s,
    'Correcto': ['Sí' if y_true[i] == y_pred_s[i] else 'No' for i in range(len(sample_idx))]
})
pred_df.to_csv('predicciones_bank.csv', index=False)
print("\n→ Resultados guardados en predicciones_bank.csv")

print("\n" + "=" * 60)
print("EJECUCIÓN COMPLETADA - Figuras guardadas en directorio actual")
print("=" * 60)
