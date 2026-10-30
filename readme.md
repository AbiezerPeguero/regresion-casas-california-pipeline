# California Housing — Regresión con Pipelines de Scikit-Learn

> Proyecto puramente académico, hecho para aprender regresión lineal aplicada con Machine Learning — parte de mi práctica guiada por el libro _Hands-On Machine Learning_ (Aurélien Géron) y mi camino de autoestudio hacia Data Engineering / Data Science.

## El problema

El dataset es el clásico **California Housing** (censo de 1990): cada fila describe un grupo de manzanas (block group), no una vivienda individual, con variables como ubicación, antigüedad promedio, cantidad de habitaciones/dormitorios, población, ingreso promedio y cercanía al océano. El objetivo: predecir `median_house_value`, el valor medio de las viviendas de ese grupo.

Es un dataset pequeño y con una limitación real importante: al describir grupos de manzanas y no casas individuales, no captura calidad de construcción, tamaño de lote, vista ni otros factores que normalmente determinan el precio — así que hay un techo de precisión inherente que ningún modelo puede superar sin más información.

## Cómo se abordó

En vez de ir directo a probar modelos a ciegas, el proyecto siguió un ciclo de **diagnóstico guiado**:

1. **Línea base** — se compararon varios modelos de regresión lineal (Lineal, Polinomial, Ridge, Lasso, ElasticNet) con validación cruzada, y se afinaron hiperparámetros con `GridSearchCV`.
2. **Diagnóstico del error** — en vez de conformarse con un RMSE promedio, se usaron las predicciones individuales de `cross_val_predict` para ver _dónde_ fallaba el modelo: por zona geográfica, por rango de precio, por residuales. Esto reveló que el error relativo era peor en las zonas más baratas (no en las caras, como parecería mirando solo dólares), y que el modelo sobre-predecía cerca del techo de precio del dataset.
3. **Hipótesis de mejora, una a la vez** — features de dominio (ratios por hogar), distancia a centros urbanos, ajuste del grado polinomial, y finalmente **segmentar el modelo en dos** (zonas interiores vs. costeras), porque ambas tienen dinámicas de precio distintas que un solo modelo no capturaba bien.
4. **Validación con datos reales** — antes de confiar en una mejora, se confirmó con predicciones reales fuera de muestra (no solo métricas ya resumidas).
5. **Evaluación final** — el modelo congelado se probó **una sola vez** contra un test nunca antes visto, siguiendo la disciplina de no tocar el test hasta que todas las decisiones del modelo ya estuvieran cerradas.

## Resultado

| Etapa                                            | Error (% sobre la mediana del precio) |
| ------------------------------------------------ | ------------------------------------- |
| Línea base                                       | 30.26%                                |
| Después de las 4 hipótesis de mejora             | 28.56%                                |
| Validado con predicciones reales (train)         | 28.60%                                |
| **Evaluación final en test (nunca antes visto)** | **28.68%**                            |

El resultado final está muy cerca del obtenido en la validación con train (diferencia de solo 0.08 puntos), lo que indica que el modelo generaliza bien en vez de estar sobreajustado — y queda razonablemente cerca de un margen de referencia externo (~27.8%, basado en benchmarks publicados de Random Forest sobre este mismo dataset).

El modelo final no es uno solo: son **dos modelos Ridge + Polynomial independientes** (uno para zonas `INLAND`, otro para el resto), cada uno con su propio preprocesamiento, sus propias features y su propio nivel de regularización.

## Estructura del proyecto

regresion_casas_california_pipeline/
├── notebooks/
│ ├── 01_EDA.ipynb # Análisis exploratorio
│ ├── 02_pipeline_entrenamiento.ipynb # Pipeline base con Scikit-Learn (Ronda 1)
│ └── 03_diagnostico_y_mejora.ipynb # Diagnóstico, hipótesis de mejora y evaluación final (Ronda 2)
├── src/
│ └── preprocesamiento.py # Funciones de preprocesamiento reutilizables
├── data/
│ ├── raw/ # Datos crudos y filtrados de outliers estructurales
│ └── processed/ # Test set congelado de la Ronda 2
├── artifacts/ # Modelos finales entrenados (joblib)
├── documentacion.md # Documentación técnica detallada, notebook por notebook
└── README.md

## Herramientas

Python · pandas · NumPy · Scikit-Learn (`Pipeline`, `ColumnTransformer`, `GridSearchCV`, `Ridge`) · Matplotlib · Jupyter

## Notas

Este repositorio documenta el proceso completo, incluyendo los errores y correcciones encontrados en el camino (ver `documentacion.md`) — no solo el resultado final. La intención es mostrar el razonamiento detrás de cada decisión, no solo el código que funcionó.

---

**Autor:** Abiezer Peguero — [GitHub](https://github.com/AbiezerPeguero)
