# Documentación técnica — California Housing (pipeline con Scikit-Learn)

Documentación completa de los tres notebooks de `regresion_casas_california_pipeline`, en el orden en que se trabajaron: exploración de los datos, construcción del pipeline base con Scikit-Learn (Ronda 1), y diagnóstico + mejora iterativa (Ronda 2).

---

## 01_EDA

### Contexto y objetivo

Antes de tocar modelos, el primer paso es conocer los datos: cuántas filas hay, si faltan valores, cómo se distribuye cada columna y si hay algo raro (topes artificiales, sesgo). Este notebook es un EDA (Exploratory Data Analysis) corto y directo sobre el dataset crudo (`casas_california.csv`, 20,640 filas), antes de cualquier limpieza o split.

**Qué se logró en este notebook:**

1. Revisar la forma y los tipos de datos del dataset.
2. Contar valores nulos por columna.
3. Ver un resumen estadístico (`describe()`) para detectar sesgo y outliers.
4. Graficar histogramas de todas las columnas numéricas.
5. Revisar la única columna categórica (`ocean_proximity`).

---

### Parte 1 — Primer vistazo

#### Celda: Cargar el dataset

**Qué le decimos a Python:**

> "Python, lee el archivo CSV crudo y guárdalo en un DataFrame."

**Explicación técnica:**
`pd.read_csv('../data/raw/casas_california.csv')`. Este es el dataset original, sin ningún filtro ni transformación todavía.

---

#### Celda: Forma y tipos de datos

**Qué le decimos a Python:**

> "Python, dime cuántas filas y columnas tiene el dataset, y de qué tipo es cada columna."

**Explicación técnica:**
`df.shape` → **20,640 filas, 10 columnas**. `df.dtypes` muestra que todas las columnas son numéricas (`float64`) excepto `ocean_proximity`, que es texto.

---

#### Celda: Valores nulos por columna

**Qué le decimos a Python:**

> "Python, dime cuántos valores vacíos hay en cada columna."

**Explicación técnica:**
`df.isnull().sum()`. Solo `total_bedrooms` tiene nulos: **207 de 20,640** (confirmado después con el cálculo exacto: **1.003%** de las filas). El resto de columnas está completo.

**Lo aprendido:** este hallazgo justifica directamente la decisión tomada después en `02_pipeline_entrenamiento`: usar `SimpleImputer(strategy="median")` solo donde hace falta, sin asumir que todas las columnas necesitan imputación.

---

#### Celda: Resumen estadístico

**Qué le decimos a Python:**

> "Python, dame la media, mediana, percentiles, mínimo y máximo de cada columna numérica."

**Explicación técnica:**
`df.describe()`. Dos cosas saltan a la vista:

- `median_house_value` tiene un máximo de exactamente **500,001** — sospechoso de ser un tope artificial, no un precio real (se confirma visualmente en el histograma).
- `housing_median_age` tiene un máximo de **52** — otro posible tope artificial.
- Columnas como `total_rooms`, `total_bedrooms`, `population` y `households` tienen una diferencia enorme entre la mediana y el máximo (ej. `population`: mediana 1,166 vs. máximo 35,682) — señal de **sesgo con cola larga** (unos pocos grupos censales muy grandes distorsionan la distribución).

**Lo aprendido:** este es el origen de la decisión de aplicar logaritmo (`pipeline_log`) a esas columnas específicas en `02_pipeline_entrenamiento` — no fue arbitrario, salió de mirar estos números.

---

#### Celda: Histogramas

**Qué le decimos a Python:**

> "Python, grafica el histograma de cada columna numérica para ver su forma de un vistazo."

**Explicación técnica:**
`df.hist(bins=50, figsize=(20, 12))`. Confirma visualmente lo que ya sugería el `describe()`: varias columnas con cola larga hacia la derecha, y dos barras sospechosamente altas en los extremos de `median_house_value` y `housing_median_age` — los topes artificiales en 500,001 y 52.

**Lo aprendido:** estos dos topes artificiales son la base del filtro estructural (`filtrar_outliers_estructurales`) aplicado más adelante tanto en `02_pipeline_entrenamiento` como en `03_diagnostico_y_mejora` — se eliminan esas filas porque no representan precios o edades reales, sino un límite de captura de datos.

---

#### Celda: Columna categórica

**Qué le decimos a Python:**

> "Python, dime qué valores únicos tiene `ocean_proximity` y cuántas veces aparece cada uno."

**Explicación técnica:**
`df['ocean_proximity'].value_counts()`:

| Categoría    | Filas |
| ------------ | ----- |
| `<1H OCEAN`  | 9,136 |
| `INLAND`     | 6,551 |
| `NEAR OCEAN` | 2,658 |
| `NEAR BAY`   | 2,290 |
| `ISLAND`     | 5     |

**Lo aprendido:** `ISLAND` tiene solo 5 filas — una categoría casi inexistente. Esto se vuelve relevante más adelante: en `03_diagnostico_y_mejora`, la segmentación del modelo final se hizo por `INLAND` vs. el resto, agrupando `ISLAND` (y las demás) dentro de "costa" por simplicidad, dado lo poco representada que está.

---

### Conclusión del EDA

El vistazo inicial ya deja tres decisiones claras para el resto del proyecto:

1. **Imputar solo `total_bedrooms`** (es la única columna con nulos).
2. **Aplicar logaritmo a las columnas con cola larga** (`total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`).
3. **Filtrar los topes artificiales** de `median_house_value` (500,001) y `housing_median_age` (52) antes de entrenar cualquier modelo.

Estas tres decisiones se ven reflejadas directamente en el preprocesamiento construido en `02_pipeline_entrenamiento`.

---

## 02_pipeline_entrenamiento

Documentación del notebook `regresion_casas_california_pipeline/notebooks/02_pipeline_entrenamiento.ipynb`.

---

### Contexto y objetivo

En este notebook se construye el preprocesamiento y el modelo con **`Pipeline` y `ColumnTransformer` de Scikit-Learn**. La idea: meter todo el preprocesamiento + el modelo en **un solo objeto**. Con una sola línea (`fit`) se entrena todo, y con otra (`predict`) se aplica todo a datos nuevos.

**Qué logramos en este notebook:**

1. Preparar los datos (split estratificado y filtrado de outliers).
2. Construir el preprocesamiento con Pipelines.
3. Entrenar un modelo base (Regresión Lineal).
4. Comparar otros modelos de regresión lineal (Polinomial, Ridge, Lasso, ElasticNet).
5. Buscar los mejores hiperparámetros con `GridSearchCV`.
6. Evaluar el mejor modelo **una sola vez** en el test set.

**La métrica que usamos:** RMSE (error en dólares) y, para entenderlo mejor, lo convertimos a **% de error sobre la mediana del precio** (RMSE ÷ mediana de `median_house_value`).

---

### Conceptos clave en palabras simples

| Concepto                | Qué es (simple)                                                                                           |
| ----------------------- | --------------------------------------------------------------------------------------------------------- |
| **Pipeline**            | Una "cadena de montaje": los datos pasan por varios pasos en orden (imputar → escalar → modelo).          |
| **make_pipeline**       | Crea un Pipeline sin que tengas que ponerle nombre a cada paso (los nombra solo).                         |
| **Pipeline(\[...\])**   | Lo mismo, pero **tú** le pones nombre a cada paso. Lo necesitas para `GridSearchCV`.                      |
| **ColumnTransformer**   | Le dices "a estas columnas aplícales este pipeline, a estas otras este otro". Luego junta todo.           |
| **FunctionTransformer** | Convierte una función normal (como `np.log`) en un paso que cabe dentro de un Pipeline.                   |
| **cross_val_score**     | Entrena y evalúa el modelo varias veces con partes distintas del train para tener una nota más confiable. |
| **GridSearchCV**        | Prueba muchas combinaciones de hiperparámetros con validación cruzada y se queda con la mejor.            |
| **Data leakage**        | Cuando información del test se "cuela" al entrenamiento. El Pipeline lo evita solo.                       |

> **Regla de oro:** `fit` aprende (solo con train). `transform` / `predict` solo aplica lo aprendido. El Pipeline respeta esto automáticamente, y dentro de la validación cruzada lo hace en **cada** partición.

> **Limitación importante:** un Pipeline **no puede eliminar filas**. Por eso los outliers estructurales (tope de 500001 y 52 años) se filtran **antes** de crear el Pipeline.

---

### Parte 1 — Preparación de los datos

#### Celda: Importar librerías y cargar el CSV

**Qué le decimos a Python:**

> "Python, trae las herramientas que voy a usar (pandas, numpy, Scikit-Learn) y lee el archivo CSV con las casas de California."

**Explicación técnica:**
`pd.read_csv()` lee el archivo desde `../data/raw/casas_california.csv` y lo guarda en un DataFrame. Las importaciones de Scikit-Learn (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, modelos, `GridSearchCV`...) se hacen al inicio para tener todo a mano.

---

#### Celda: Crear `income_cat` con `pd.cut`

**Qué le decimos a Python:**

> "Python, crea una columna temporal que meta los ingresos en 5 categorías (de bajos a altos)."

**Explicación técnica:**
`pd.cut(bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1,2,3,4,5])` convierte `median_income` (continua) en 5 grupos. **No es una variable para el modelo**: solo sirve para que el split sea estratificado.

**Lo aprendido:** `np.inf` permite que la última categoría no tenga límite superior.

---

#### Celda: Split estratificado (train / test)

**Qué le decimos a Python:**

> "Python, separa 80% de las casas para entrenar y 20% para probar, y asegúrate de que ambos grupos tengan la misma proporción de ingresos bajos, medios y altos."

**Explicación técnica:**
`train_test_split(test_size=0.2, stratify=df["income_cat"], random_state=42)`.

- `stratify` mantiene las proporciones de `income_cat` en train y test (así el test representa bien a la población).
- `random_state=42` hace que el split sea **siempre el mismo** (reproducible).

**Lo aprendido:** el split se hace **antes** de limpiar o transformar nada. Así el test queda "virgen".

---

#### Celda: Eliminar `income_cat`

**Qué le decimos a Python:**

> "Python, ya no necesito la columna temporal de categorías de ingreso, bórrala de train y de test."

**Explicación técnica:**
`drop(columns="income_cat")` en ambos conjuntos. Ya cumplió su trabajo (estratificar); dejarla sería meter ruido o una columna que en producción no existiría.

---

#### Celda: Separar X e y (train y test)

**Qué le decimos a Python:**

> "Python, separa lo que el modelo va a _ver_ (las características) de lo que debe _adivinar_ (el precio)."

**Explicación técnica:**

- `X_train` = todas las columnas menos `median_house_value`.
- `y_train` = solo `median_house_value`.
- Igual para `X_test` e `y_test`.

---

#### Celda: Importar `filtrar_outliers_estructurales` y filtrar

**Qué le decimos a Python:**

> "Python, usa mi función del módulo para quitar las casas con precio topado en 500001 o con edad 52 (valores artificiales)."

**Explicación técnica:**
Se agrega la carpeta `src/` al `sys.path` para poder importar la función desde `preprocesamiento.py`. Luego se aplica el filtro a train **y** a test, y se hace `reset_index(drop=True)` en **X y en y** para que sus índices queden alineados (0, 1, 2...).

**Lo aprendido (error real que tuvimos):** si haces `reset_index` solo en X y no en y, los índices no coinciden y al unir/comparar salen filas de más. Siempre resetear ambos.

---

### Parte 2 — Preprocesamiento con Pipelines

#### Celda: `pipeline_log` (columnas sesgadas)

**Qué le decimos a Python:**

> "Python, para las columnas con cola larga: rellena los vacíos con la mediana, aplícales logaritmo y luego escálalas."

**Explicación técnica:**

```python
pipeline_log = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)
```

1. `SimpleImputer(median)`: rellena nulos con la mediana (aprendida en train).
2. `FunctionTransformer(np.log)`: aplica logaritmo para "aplastar" la cola larga y que la distribución se parezca más a una campana.
3. `StandardScaler`: deja la media en 0 y la desviación en 1.

`feature_names_out="one-to-one"` le dice al Pipeline que las columnas de salida se llaman igual que las de entrada (necesario para que el `ColumnTransformer` pueda conservar nombres).

---

#### Celda: `num_pipeline_simple` (columnas no sesgadas)

**Qué le decimos a Python:**

> "Python, para las columnas que ya se ven normales: rellena nulos con la mediana y escálalas. Sin logaritmo."

**Explicación técnica:**
`make_pipeline(SimpleImputer(strategy="median"), StandardScaler())`. Aplicar log a `longitude` (valores negativos) daría error, y a `latitude` / `housing_median_age` no les hace falta.

---

#### Celda: `cat_pipeline` (columna categórica)

**Qué le decimos a Python:**

> "Python, para `ocean_proximity`: rellena vacíos con la categoría más común y conviértela en columnas de 0 y 1."

**Explicación técnica:**

```python
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
)
```

- `most_frequent` (moda) porque una categoría no tiene mediana.
- `handle_unknown="ignore"`: si en test aparece una categoría nueva, no se rompe (pone ceros).
- `sparse_output=False`: devuelve una matriz normal, no "dispersa".

**Lo aprendido (error real):** pusimos `strategy="median"` por error en esta celda. La mediana solo sirve para números; para texto va `most_frequent`.

---

#### Celda: Listas de columnas

**Qué le decimos a Python:**

> "Python, guarda en listas qué columnas van a cada pipeline."

**Explicación técnica:**

```python
columnas_sesgadas    = ["total_rooms","total_bedrooms","population","households","median_income"]
columnas_no_sesgadas = ["longitude","latitude","housing_median_age"]
columna_categorica   = ["ocean_proximity"]
```

Estas listas salen del EDA: miramos los histogramas y decidimos cuáles tenían cola larga.

---

#### Celda: `ColumnTransformer`

**Qué le decimos a Python:**

> "Python, junta los tres pipelines: cada uno trabaja con sus columnas y al final pega todos los resultados en una sola tabla."

**Explicación técnica:**

```python
column_transformer = ColumnTransformer([
    ("num_log",    pipeline_log,         columnas_sesgadas),
    ("num_simple", num_pipeline_simple,  columnas_no_sesgadas),
    ("cat",        cat_pipeline,         columna_categorica),
])
```

Cada tupla es `(nombre, transformador, columnas)`. El resultado es una matriz con todas las columnas ya procesadas: 5 + 3 + 5 (las categorías de `ocean_proximity` en one-hot) = **13 columnas**.

**Lo aprendido:** esto reemplaza todo lo que antes hacíamos a mano con funciones + `joblib`. El `ColumnTransformer` ya "recuerda" lo aprendido en `fit`.

---

### Parte 3 — Modelo 1: Regresión Lineal (línea base)

#### Celda: `pipeline_final`

**Qué le decimos a Python:**

> "Python, arma una cadena completa: primero el preprocesamiento y al final una Regresión Lineal."

**Explicación técnica:**

```python
pipeline_final = Pipeline([
    ("transformer", column_transformer),
    ("modelo", LinearRegression())
])
```

Un Pipeline de dos pasos: el `ColumnTransformer` y el modelo.

---

#### Celda: Validación cruzada del modelo base

**Qué le decimos a Python:**

> "Python, evalúa este modelo 10 veces con partes distintas del train y dime qué error tiene en promedio."

**Explicación técnica:**

```python
scores = cross_val_score(pipeline_final, X_train_filtrado, y_train_filtrado,
                         cv=10, scoring="neg_root_mean_squared_error")
rmse_cv = -scores
```

- `cv=10`: divide el train en 10 partes; entrena con 9 y evalúa con 1, y repite 10 veces.
- Scikit-Learn maximiza los scores, por eso el RMSE sale **negativo**; lo multiplicamos por -1 para leerlo normal.

**Lo aprendido:** el Pipeline completo (con imputer, escalador, etc.) se **reentrena en cada partición**, así no hay leakage ni siquiera dentro de la validación cruzada.

---

#### Celda: Convertir RMSE a % de error

**Qué le decimos a Python:**

> "Python, dime cuánto se equivoca el modelo en porcentaje, comparado con el precio típico de una casa."

**Explicación técnica:**
`mediana_precio = y_train_filtrado.median()` y luego `rmse_promedio / mediana_precio * 100`.
Un RMSE en dólares es difícil de interpretar; en porcentaje sí: "el modelo se equivoca ~36% del precio típico".

**Resultado: Regresión Lineal ≈ 35.86% de error.** Esta es nuestra línea base: cualquier modelo posterior debe superarla.

---

### Parte 4 — Otros modelos de regresión lineal

Para todos usamos el **mismo `column_transformer`**, solo cambiamos el modelo del final. Ese es el poder del Pipeline.

#### Celda: Regresión Polinomial

**Qué le decimos a Python:**

> "Python, antes del modelo, crea columnas nuevas con combinaciones y cuadrados de las que ya hay, para que la línea pueda curvarse."

**Explicación técnica:**

```python
pipeline_polinomal = Pipeline([
    ("transformer", column_transformer),
    ("polinomial", PolynomialFeatures(degree=2, include_bias=False)),
    ("regresor", LinearRegression())
])
```

`PolynomialFeatures(degree=2)` pasa de 13 columnas a muchas más (cuadrados y productos entre pares). `include_bias=False` evita la columna de unos (el modelo ya tiene su propio intercepto).

**Resultado ≈ 30.75%** → mejora clara sobre la línea base. La relación entre las variables y el precio **no es una línea recta**.

---

#### Celda: Ridge

**Qué le decimos a Python:**

> "Python, usa regresión lineal pero castiga los coeficientes muy grandes para que el modelo no se pase de ajustado."

**Explicación técnica:**
`Ridge(alpha=1.0)` agrega una penalización L2 (suma de cuadrados de los coeficientes). Más `alpha` = más castigo = modelo más simple.

**Resultado ≈ 35.86%** → igual que la lineal. Con solo 13 columnas no hay overfitting que corregir, así que la regularización no aporta.

---

#### Celda: Ridge + Polinomial

**Qué le decimos a Python:**

> "Python, combina las dos ideas: crea las columnas polinomiales y usa Ridge para controlar que no se descontrole."

**Explicación técnica:**
Pipeline con `PolynomialFeatures(degree=2)` + `Ridge(alpha=1.0)`. Aquí la regularización sí tiene sentido: al crear muchas columnas nuevas hay riesgo de overfitting.

**Resultado ≈ 30.75%** → igual que la polinomial sola con alpha=1.0.

---

#### Celda: Lasso

**Qué le decimos a Python:**

> "Python, usa regresión lineal con una penalización que puede dejar coeficientes en exactamente cero (descarta columnas inútiles)."

**Explicación técnica:**
`Lasso(alpha=0.1, max_iter=10000)` usa penalización L1 (suma de valores absolutos). Hace selección automática de variables.

**Resultado ≈ 35.86%, pero con `ConvergenceWarning`.**

**Lo aprendido / pendiente:**

- El warning significa que el algoritmo iterativo **no terminó de converger** (el "duality gap" seguía alto). Por eso este resultado **no es confiable**.
- Subir `max_iter` no lo solucionó. La causa no la comprobamos; hipótesis a probar: escala/colinealidad de las columnas (ver pendientes).

---

#### Celda: ElasticNet

**Qué le decimos a Python:**

> "Python, mezcla las dos penalizaciones (Ridge y Lasso) a partes iguales."

**Explicación técnica:**
`ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42)`. `l1_ratio=0.5` = 50% L1 y 50% L2.

**Resultado ≈ 36.41%** → algo peor que la lineal. Con `alpha=0.1` penaliza demasiado y le quita capacidad al modelo.

---

### Parte 5 — Búsqueda de hiperparámetros con GridSearchCV

#### Celda: GridSearchCV sobre Ridge + Polinomial

**Qué le decimos a Python:**

> "Python, prueba todas las combinaciones de grado polinomial y de alpha, evalúalas con validación cruzada y dime cuál es la mejor."

**Explicación técnica:**

```python
param_grid = [{"poly__degree": [1, 2, 3],
               "ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}]
buscando_mejor = GridSearchCV(pipeline_ridge_poly, param_grid,
                              cv=10, scoring="neg_root_mean_squared_error")
buscando_mejor.fit(X_train_filtrado, y_train_filtrado)
```

- Los nombres siguen la forma **`paso__parametro`** (doble guion bajo). `poly` y `ridge` son los nombres que le pusimos a los pasos del Pipeline.
- **Un solo diccionario** = todas las combinaciones (3 × 5 = 15). Si pones diccionarios separados, prueba cada uno por su lado, no combinados.
- Total de entrenamientos: 15 combinaciones × 10 particiones = **150**.

**Cómo leer el resultado:**

- `best_params_`: la mejor combinación.
- `best_score_`: el score de esa combinación, **ya promediado** de las 10 particiones (y negativo; se multiplica por -1).
- `best_estimator_`: el Pipeline con esa combinación, **ya reentrenado con todo el train**. Listo para predecir.

**Resultado ≈ 29.84%** → el mejor de todos los modelos probados.

**Lo aprendido (errores reales):**

- Se usaron nombres de paso distintos a los del Pipeline (error de clave). Los nombres del `param_grid` deben coincidir exactamente.
- Los valores ganadores quedaron **en el borde** del grid, lo que sugiere que el óptimo real podría estar más allá. Pendiente: ampliar el grid.

---

#### Celda: GridSearchCV sobre Lasso

**Qué le decimos a Python:**

> "Python, busca el mejor `alpha` para Lasso."

**Explicación técnica:**
`param_grid = [{"regresor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]}]` con `GridSearchCV(pipeline_lasso, ...)`.

**Resultado ≈ 35.86%**, sin mejorar sobre la lineal. Además sigue el tema de convergencia, así que lo tomamos con reserva.

---

### Parte 6 — Evaluación final en el test set

#### Celda: Extraer el modelo final

**Qué le decimos a Python:**

> "Python, quédate con el mejor modelo que encontró la búsqueda."

**Explicación técnica:**
`modelo_final = buscando_mejor.best_estimator_`. Ya está entrenado con todo el train; **no hace falta volver a hacer `fit`**.

---

#### Celda: Predecir en test y calcular el error

**Qué le decimos a Python:**

> "Python, ahora sí, usa el modelo con las casas que nunca vio y dime cuánto se equivoca."

**Explicación técnica:**

```python
y_pred = modelo_final.predict(X_test_filtrado)
rmse_test = root_mean_squared_error(y_test_filtrado, y_pred)
```

- `predict` aplica automáticamente todo el preprocesamiento aprendido en train (imputar, log, escalar, one-hot, polinomial) y luego predice.
- `root_mean_squared_error(y_true, y_pred)`: el **primer argumento son los valores reales** y el segundo las predicciones.
- Luego se divide entre la mediana del precio para obtener el %.

**Resultado en test: ≈ 29.46% de error** (vs. ≈ 29.84% en validación cruzada).

**Lo aprendido:**

- Los dos números son muy parecidos → **no hay señal de overfitting** y la estimación de validación cruzada fue honesta.
- El test se usa **una sola vez**, al final. Si lo usáramos para ajustar el modelo, dejaría de ser una prueba justa.

---

### Resumen de resultados

| Modelo                                | Error aprox. (% sobre la mediana) | Comentario                                  |
| ------------------------------------- | --------------------------------- | ------------------------------------------- |
| Regresión Lineal (línea base)         | 35.86%                            | Punto de partida                            |
| Polinomial (grado 2)                  | 30.75%                            | Mejora grande: relación no lineal           |
| Ridge                                 | 35.86%                            | Sin cambio: no hay overfitting que corregir |
| Ridge + Polinomial (alpha=1)          | 30.75%                            | Igual que la polinomial sola                |
| Lasso (alpha=0.1)                     | 35.86%                            | ⚠️ ConvergenceWarning, no confiable         |
| ElasticNet (alpha=0.1)                | 36.41%                            | Algo peor: penaliza demasiado               |
| **Ridge + Polinomial (GridSearchCV)** | **29.84%**                        | **Mejor modelo (CV)**                       |
| Lasso (GridSearchCV)                  | 35.86%                            | Sin mejora, con reserva                     |
| **Mejor modelo en test**              | **29.46%**                        | **Resultado final**                         |

**Conclusión:** lo que más ayudó fue **dar flexibilidad al modelo (términos polinomiales)**. La regularización aporta poco por sí sola en este dataset pequeño, pero cuida que lo polinomial no se pase de ajuste.

---

### Limitaciones

- Un ~29% de error sobre el precio típico es **alto** para uso real. Los modelos lineales tienen techo en este problema.
- Lasso no convergió: sus resultados no se pueden usar para sacar conclusiones.
- El grid de Ridge + Polinomial dejó valores ganadores en el borde, así que no sabemos si hay un óptimo mejor.
- Solo evaluamos el error **global**; no sabemos si falla más en ciertas zonas (por ejemplo `ocean_proximity`) o en ciertos rangos de precio.
- El filtrado de outliers estructurales (500001 y 52) significa que el modelo **no sabe predecir** casas topadas.

---

### Pendientes (próximos pasos)

1. **Lasso / convergencia:** investigar la causa (escala, colinealidad, `tol`, `selection="random"`).
2. **Ampliar el grid** de Ridge + Polinomial (más valores de `alpha`, probar otros grados).
3. **Error por segmento:** ver el error por `ocean_proximity` y por rango de precio.
4. **SGDRegressor:** para cuando el dataset sea demasiado grande para los modelos anteriores.
5. **Siguientes modelos del libro** (árboles, Random Forest, etc.), que suelen superar a los lineales aquí.

---

### Buenas prácticas aplicadas

- Split **antes** de cualquier transformación (test intacto).
- Estratificación por ingresos + `random_state=42` (reproducible).
- Preprocesamiento **dentro** del Pipeline: sin leakage, ni siquiera en validación cruzada.
- Filtrar outliers **antes** del Pipeline (los Pipelines no eliminan filas).
- `reset_index(drop=True)` en X **y** y para mantenerlos alineados.
- Comparar modelos con **validación cruzada**, no con el test.
- Test usado **una sola vez** al final.
- Interpretar el error en **porcentaje**, no solo en dólares.

---

## 03_diagnostico_y_mejora

Documentación del notebook `regresion_casas_california_pipeline/notebooks/03_diagnostico_y_mejora.ipynb`.

---

### Contexto y objetivo

Este notebook es la **Ronda 2** del proyecto California Housing. La Ronda 1 (trabajada en `02_pipeline_entrenamiento.ipynb`) terminó con un modelo evaluado una sola vez en su test, dando **29.46%** de error sobre la mediana — ese resultado quedó consumido y no se vuelve a tocar.

En la Ronda 2 se repite el ciclo completo con un split nuevo, pero esta vez con un enfoque de **mejora iterativa guiada por diagnóstico**: en vez de solo comparar modelos y ajustar hiperparámetros, se analiza _dónde_ y _por qué_ falla el modelo antes de decidir qué probar.

**Qué se logró en este notebook:**

1. Congelar un test nuevo para la Ronda 2 (`random_state=52`).
2. Re-comparar modelos candidatos y fijar una línea base con `GridSearchCV`.
3. Diagnosticar el error con `cross_val_predict` (por zona, por rango de precio, por residuales).
4. Probar 4 hipótesis de mejora, una a la vez, siempre validando con CV.
5. Validar el resultado combinado con predicciones reales fuera de muestra.
6. Analizar los residuales del modelo mejorado por 4 ángulos distintos.
7. Congelar el modelo final (dos modelos: interior y costa) y evaluarlo **una sola vez** en el test de la Ronda 2.

**La métrica:** RMSE (en dólares) convertido a **% de error sobre la mediana de `y_train_set`** ($170,700), para que el número sea comparable entre todas las etapas del proyecto.

---

### Conceptos clave en palabras simples

| Concepto                       | Qué es (simple)                                                                                                                                                                                                                         |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **cross_val_score**            | Te da solo el RMSE resumido (promedio) de cada vuelta de validación cruzada.                                                                                                                                                            |
| **cross_val_predict**          | Te da la **predicción individual** de cada fila cuando estuvo en el fold de validación — permite auditar caso por caso, no solo ver un promedio.                                                                                        |
| **Segmentación de modelos**    | En vez de un solo modelo para todos los datos, se entrenan **modelos independientes** para subgrupos distintos (aquí: interior vs. costa), cada uno con su propio `ColumnTransformer` y su propio ajuste de hiperparámetros.            |
| **RMSE combinado (RMSE² × n)** | Para comparar un modelo segmentado contra uno único de forma justa, no se promedian los RMSE de cada segmento — se recupera la suma de errores al cuadrado de cada uno, se suman, se dividen entre el total de filas y se saca la raíz. |
| **Congelar el modelo**         | Entrenar una última vez con el 100% del train (ya no solo folds de CV) usando los hiperparámetros ya decididos, y guardar ese resultado con `joblib` para no depender de reentrenar.                                                    |

> **Regla de oro del proyecto:** el test de cada ronda se usa **una sola vez**. Si hace falta iterar después de evaluarlo, se abre un split nuevo — no se vuelve a tocar el mismo test.

---

### Fase 0-2 — Congelar el test nuevo y la línea base

**Qué le decimos a Python:**

> "Python, separa un 20% de los datos filtrados como test de la Ronda 2, con una semilla fija (52) y estratificando por nivel de ingreso, y guárdalo aparte para no volver a tocarlo hasta el final."

**Explicación técnica:**
`train_test_split(X_train_df, y_train_df, test_size=0.2, random_state=52, stratify=X_train_df["income_cat"])`, sobre los datos ya filtrados de outliers estructurales (`X_train_filtrado.csv`/`y_train_filtrado.csv`, heredados de la Ronda 1). Train quedó en 11,871 filas, test en 2,968. Se re-compararon los mismos modelos candidatos de la Ronda 1 (Lineal, Polynomial, Ridge, Ridge+Poly, Lasso, ElasticNet) con CV sobre este nuevo split, y se corrió un `GridSearchCV` de línea base (degree + alpha) que dio **30.26%**.

---

### Fase 3 — Diagnóstico del error

**Qué le decimos a Python:**

> "Python, en vez de solo decirme el RMSE promedio, dame la predicción real de cada casa cuando estuvo en el fold de validación — así puedo ver dónde falla el modelo, no solo cuánto falla en promedio."

**Explicación técnica:**
`cross_val_predict(pipeline, X_train_set, y_train_set, cv=10)` sobre el modelo Ridge+Polynomial (degree=3, alpha=200) de la línea base. Con esas predicciones se armó un DataFrame (`evaluando_df`) con `precio_real`, `predicción`, `residual` y `error_absoluto`, y se agrupó por `ocean_proximity` y por rango de precio (cuartiles con `pd.cut`).

**Hallazgos:**

1. **El error relativo es mayor en zonas baratas, no en zonas caras.** INLAND tenía el error absoluto más bajo en dólares, pero el error relativo más alto (23.66% de su propio precio) — comparar solo en dólares era engañoso sin normalizar por precio.
2. **El rango de precio "Muy Alto" tenía casi el doble de error** que los otros tres rangos, a pesar de tamaños de muestra casi idénticos — probable límite del dataset (describe grupos censales, no viviendas individuales).
3. **El modelo producía predicciones negativas** (mínimo -$48,005, imposible en la realidad) y **sobre-predecía por encima del techo de $500,000** del dataset — comportamiento esperado de `PolynomialFeatures`, que no tiene límite natural.

**Lo aprendido:** comparar error en dólares entre grupos con precios promedio muy distintos (como zonas) es engañoso. Siempre normalizar por el precio de referencia de cada grupo antes de comparar.

---

### Fase 4 — Hipótesis de mejora (probadas una a la vez, siempre con CV)

#### Hipótesis 1: features de dominio

Se crearon `rooms_per_household`, `bedrooms_ratio` y `population_per_household` a partir de las columnas totales del grupo censal. La primera corrida de CV dio **35.85%** (peor que la línea base) — pero 9 de 10 folds estaban bien, y uno se disparaba a 145,309 de RMSE. Se rastreó a una fila con 1,243 "personas por hogar" (error de captura de datos). Al filtrar las 16 filas con `population_per_household > 10`, el resultado mejoró a **29.99%**.

**Lo aprendido:** una feature nueva puede parecer mala cuando el problema real es un outlier aislado — verificar antes de descartar la hipótesis.

#### Hipótesis 2: distancia a centros urbanos

Se calculó la distancia euclidiana de cada casa a San Francisco, Los Ángeles y San Diego, y se tomó el mínimo de las tres como `distancia_ciudad` (evitando meter 3 columnas correlacionadas = multicolinealidad). Resultado: **28.64%** con degree=3.

#### Hipótesis 3: grado del polinomio y alpha

Con las features nuevas, degree=3 (28.64%) seguía ganando a degree=2 (29.80%) sin evidencia de sobreajuste en CV. Se decidió quedarse con **degree=2** por margen de maniobra para hipótesis futuras. `GridSearchCV` de alpha dio **alpha=0.01**, resultando en **29.25%**.

**Lo aprendido:** que un modelo más complejo (degree mayor) dé mejor resultado en CV no es automáticamente señal de sobreajuste — hay que revisar la desviación entre folds, no solo asumir.

#### Hipótesis 4: segmentación interior/costa

En vez de un modelo único, se entrenaron **dos modelos independientes** según `ocean_proximity == 'INLAND'` (interior: 4,055 filas, costa: 7,800), cada uno con su propio `ColumnTransformer` y su propio `GridSearchCV` de alpha (degree=2 fijo).

| Segmento | RMSE    | % sobre su mediana | Alpha ganador |
| -------- | ------- | ------------------ | ------------- |
| Interior | $37,499 | 34.63%             | 50            |
| Costa    | $53,676 | 25.92%             | 0.001         |

Combinando ambos RMSE con la fórmula `RMSE² × n` (no un promedio simple), el resultado combinado fue **28.56%** — una mejora de 0.69 puntos sobre el modelo único.

---

### Validación del resultado con predicciones reales

El 28.56% de Hipótesis 4 se obtuvo combinando matemáticamente dos `best_score_`. Para confirmarlo, se corrió `cross_val_predict` por separado para cada segmento (con los alphas ganadores reales: 50 interior, 0.001 costa — los pipelines de prueba se habían quedado con alpha=0.01 y hubo que reconstruirlos), se concatenaron las predicciones y los valores reales de ambos segmentos, y se calculó un único RMSE real con `root_mean_squared_error`.

**Resultado: 28.60%** — a solo 0.04 puntos del cálculo matemático, confirmando que la combinación `RMSE² × n` era una aproximación confiable, no un atajo riesgoso.

---

### Análisis de residuales del modelo segmentado

Con el modelo validado, se analizaron sus residuales por 4 ángulos, usando el mismo DataFrame de predicciones reales combinadas:

1. **Zona:** la segmentación cumplió su propósito — el error relativo de INLAND bajó de 23.66% a 20.40% (la mayor mejora de las 4 categorías).
2. **Rango de precio:** mismo patrón que Fase 3 (el error sigue creciendo con el precio); 3 de 4 rangos mejoraron, uno ("Alto") empeoró levemente.
3. **Ingreso:** el error también sube con el ingreso, pero no es una pista nueva — `median_income` y `precio_real` tienen correlación de 0.66, es en gran parte el mismo patrón de precio visto de otro ángulo.
4. **Lat/long:** el error no está disperso de forma pareja — hay agrupamientos cerca de la Bahía de San Francisco y en una franja costa-central/valle-central.

**Decisión:** no se abrió una Hipótesis 5 (clustering geográfico con KMeans, la técnica clásica para esto). El patrón geográfico es real pero de magnitud incierta, y el resultado ya estaba cerca del margen de referencia (~27.8%, benchmark externo). Se congeló el modelo tal como estaba.

---

### Fase 5 — Congelar el modelo final

**Qué le decimos a Python:**

> "Python, ya decidiste los hiperparámetros — ahora entrena cada modelo una última vez con el 100% de los datos de su segmento (no solo 90% como en cada fold de CV), y guarda el resultado en disco para no depender de reentrenar cada vez que abra el notebook."

**Explicación técnica:**
`pipeline_interior_new.fit(X_interior, y_interior)` y `pipeline_costa_new.fit(X_costa, y_costa)` — entrenamiento simple, sin CV, usando el 100% de cada segmento. Los objetos entrenados se guardaron con `joblib.dump()` en `artifacts/pipeline_interior_new.pkl` y `artifacts/pipeline_costa_new.pkl`, y las celdas de entrenamiento se comentaron para que futuras ejecuciones carguen (`joblib.load()`) el modelo congelado en vez de reentrenar.

**Lo aprendido:** la validación cruzada nunca entrena con el 100% de los datos — cada fold usa solo el 90%. Una vez decidida la configuración final, hay que reentrenar una última vez con todo el train disponible.

---

### Fase 6 — Evaluación final en el test de la Ronda 2

**Qué le decimos a Python:**

> "Python, aplícale al test exactamente las mismas transformaciones que le hiciste al train (las features de dominio, el filtro de outliers, la distancia a ciudades), divídelo igual que al train, y dame la predicción de cada modelo congelado — sin reentrenar nada."

**Explicación técnica:**
Se cargó `X_test.csv`/`y_test.csv` (el split congelado en Fase 0), se replicaron a mano las columnas de Hipótesis 1 y 2 (no estaban dentro del `ColumnTransformer`, así que no se propagan solas), se filtraron 6 filas con `population_per_household > 10` (vs. 16 en train — menor proporción, pero el mismo tipo de outlier), se dividió por `ocean_proximity == 'INLAND'` (1,028 interior, 1,934 costa), y se predijo con `pipeline_interior_new.predict()` y `pipeline_costa_new.predict()` — **sin volver a usar `ColumnTransformer`, `cross_val_score` ni `GridSearchCV`**, porque toda esa maquinaria ya cumplió su función en train; el pipeline ya entrenado aplica su `.transform()` aprendido automáticamente al llamar `.predict()`.

**Resultado final: 28.68%** sobre datos nunca antes vistos por el modelo.

| Etapa                                            | % sobre mediana |
| ------------------------------------------------ | --------------- |
| Línea base (Fase 2)                              | 30.26%          |
| Hipótesis 4 (segmentado, combinación matemática) | 28.56%          |
| Validación con `cross_val_predict` (train)       | 28.60%          |
| **Test final (Ronda 2)**                         | **28.68%**      |
| Referencia externa (benchmark RandomForest)      | ~27.8%          |

La diferencia entre la validación en train (28.60%) y el resultado en test (28.68%) es de solo 0.08 puntos — señal clara de que el modelo **generaliza bien** y no estaba sobreajustado, a pesar de 4 hipótesis probadas y ajustadas sobre ese mismo train.

> ### Lección reutilizable: verificar filas, no solo columnas
>
> Al replicar las transformaciones en el test, una celda que revisaba `X_test_set.shape` **antes** de aplicar un filtro de outliers mostró el mismo número de filas que **después** del filtro — pareciendo que el test no tenía outliers. La causa real: en Jupyter, la **posición** de una celda en el notebook no garantiza su **orden real de ejecución**. Esa celda de verificación se había corrido manualmente _después_ del filtro (visible en su número de `execution_count`, mayor al de la celda del filtro), mostrando un resultado ya filtrado como si fuera el "antes".
>
> **La columna (el cálculo) estaba bien; la fila (el conteo) mentía por el orden de ejecución, no por el código.** La forma confiable de verificar es comparar los números de `execution_count` de las celdas involucradas, o simplemente reiniciar el kernel y correr todo en orden (`Restart & Run All`) antes de confiar en cualquier comparación "antes/después".

---

### Conclusión general de la Ronda 2

Partiendo de una línea base de 30.26%, el proceso de diagnóstico guiado (en vez de prueba y error a ciegas) llevó el modelo a **28.68%** en test real — una mejora de 1.58 puntos porcentuales, validada tanto en train (predicciones reales fuera de muestra) como en un test nunca antes visto, y razonablemente cerca del límite de referencia externo del dataset (~27.8%). El modelo final está compuesto por **dos modelos Ridge+Polynomial independientes** (interior y costa de California), cada uno con sus propias features, su propio preprocesamiento y su propio alpha, reflejando que ambas zonas tienen dinámicas de precio distintas que un solo modelo no capturaba bien.
