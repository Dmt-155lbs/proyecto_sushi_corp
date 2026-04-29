# Statistical Learning: Wine Quality & Bank Marketing

Este repositorio contiene la resolución de dos problemas prácticos de Machine Learning enfocados en clasificación, análisis exploratorio de datos (EDA) y la toma de decisiones basada en métricas de negocio. El proyecto incluye la implementación en Python (`scikit-learn`) y el informe técnico detallado en LaTeX.

## 🍷 Proyecto 1: Predicción de Calidad del Vino (Wine Quality)

Se combinaron los datasets de vino tinto y blanco del repositorio UCI para clasificar la calidad del vino. Se optó por una **clasificación binaria** (premium vs. estándar) ya que ofrece una decisión práctica y directa para los vinicultores, evitando las asunciones poco realistas de la regresión y la inestabilidad multiclase en los niveles extremos de calidad.

**Puntos Clave:**
* **Comparación de Modelos:** Se evaluaron modelos lineales regularizados (Ridge, Lasso, Elastic Net), basados en árboles (Decision Tree, Random Forest), SVM y Redes Neuronales.
* **Estabilidad de Predictores:** Se demostró visualmente que Elastic Net exhibe trayectorias de coeficientes más suaves y estables que Lasso al manejar variables correlacionadas, distribuyendo mejor los pesos.
* **Resultados:** El modelo Random Forest tuneado presentó el mejor balance entre precisión y recall, logrando los valores más altos de F1-Score y ROC-AUC en el conjunto de prueba.

## 🏦 Proyecto 2: Campaña de Marketing Bancario (Bank Marketing)

El objetivo es predecir si un cliente suscribirá un depósito a plazo durante una campaña de telemarketing. El dataset presenta un severo desbalance de clases (~88.7% "no" y ~11.3% "sí"), lo que requirió un enfoque en métricas más allá del simple accuracy.

**Puntos Clave:**
* **Manejo de Data Leakage:** Se identificó y removió la variable `duration` del pipeline de preprocesamiento. Al conocerse únicamente después de la llamada, su inclusión inflaría artificialmente el rendimiento del modelo, restándole utilidad predictiva real.
* **Estrategia Operativa:** Se evaluaron matrices de confusión a múltiples umbrales de decisión. Se determinó que un umbral de 0.3 optimiza el recall, permitiendo capturar más suscriptores potenciales sin una pérdida excesiva de precisión.
* **Recomendación de Contacto:** Mediante un análisis de curva Lift, se demostró que priorizando por score y contactando solo al top 20% de los clientes con mayor probabilidad, es posible capturar más del 50% de las suscripciones potenciales.

## 📂 Estructura del Repositorio

* `practical_question_1.py`: Pipeline completo para el análisis de Wine Quality.
* `practical_question_2.py`: Pipeline completo para el análisis de Bank Marketing.
* `informe_parcial2.tex`: Código fuente del informe técnico detallado.
* `*.png`: Visualizaciones de EDA, matrices de correlación, importancia de variables y curvas de rendimiento.

## 🛠 Tecnologías Utilizadas

* **Lenguaje:** Python 3.x
* **Librerías:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
* **Documentación:** LaTeX
