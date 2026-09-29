# Titanic Survival Prediction with Neural Networks

Proyecto de práctica de **Machine Learning y Redes Neuronales Artificiales (RNA)** desarrollado utilizando el dataset Titanic de `seaborn`.

El objetivo principal fue experimentar con diferentes arquitecturas de redes neuronales para una tarea de **clasificación binaria**, comparando su comportamiento mediante validación cruzada y distintas métricas de evaluación.

> **Nota:** Este proyecto tiene fines educativos y fue desarrollado principalmente para practicar conceptos relacionados con redes neuronales, preprocesamiento y evaluación de modelos.

## Tecnologías utilizadas

* Python 3.12.3
* TensorFlow 2.21.0
* Keras 3.15.1
* Scikit-learn 1.8.0
* Imbalanced-learn 0.14.2
* Pandas 3.0.2
* NumPy 2.4.4
* Matplotlib 3.10.8
* Seaborn 0.13.2
* Jupyter Notebook

## Dataset

Se utiliza el dataset **Titanic**, disponible directamente mediante Seaborn.

El objetivo es predecir la variable:

* `survived`: supervivencia del pasajero.

  * `0`: No sobrevivió
  * `1`: Sobrevivió

Las variables utilizadas como características son:

* `pclass`: clase del pasajero.
* `sex`: sexo del pasajero.
* `age`: edad.
* `sibsp`: cantidad de hermanos/cónyuges a bordo.
* `parch`: cantidad de padres/hijos a bordo.
* `fare`: tarifa pagada.
* `embarked`: puerto de embarque.

Se excluyen variables que podrían introducir información redundante o fuga de información respecto al objetivo, como `alive`.

## Preprocesamiento

El preprocesamiento se realiza teniendo en cuenta la separación entre los datos de entrenamiento y prueba para evitar contaminar el conjunto de evaluación externo.

Las principales etapas son:

1. Selección de las variables utilizadas.
2. Eliminación de registros sin información en `embarked`.
3. Codificación binaria de `sex`.
4. Codificación **One-Hot** de `embarked`.
5. Imputación de valores faltantes de `age` utilizando la mediana.
6. Estandarización mediante `StandardScaler`.
7. Balanceo de las clases mediante **SMOTE**.

Durante la validación cruzada, las transformaciones se ajustan utilizando los datos de entrenamiento de cada fold. El conjunto de prueba externo no participa en estas transformaciones.

SMOTE también se aplica únicamente sobre los datos de entrenamiento, manteniendo los datos de validación y prueba con su distribución original.

## Reproducibilidad

Se establece una semilla fija mediante:

```python
keras.utils.set_random_seed(42)
```

Esto permite obtener resultados reproducibles bajo el mismo entorno y configuración de ejecución.

## Modelos

Se implementan y comparan tres arquitecturas de redes neuronales.

### Modelo 1

* Capa de entrada.
* Dense de 16 neuronas con activación ReLU.
* Capa de salida de 1 neurona con activación Sigmoid.
* Optimizador: Adam.
* Learning rate: `0.01`.

### Modelo 2

* Capa de entrada.
* Dense de 32 neuronas con activación Tanh.
* Dense de 16 neuronas con activación Tanh.
* Capa de salida de 1 neurona con activación Sigmoid.
* Optimizador: Adam.
* Learning rate: `0.001`.

### Modelo 3

* Capa de entrada.
* Dense de 64 neuronas con activación ReLU.
* Dropout de `0.3`.
* Dense de 32 neuronas con activación ReLU.
* Dropout de `0.2`.
* Capa de salida de 1 neurona con activación Sigmoid.
* Optimizador: SGD.
* Learning rate: `0.01`.
* Momentum: `0.9`.

## Estrategia de evaluación

Para evaluar los modelos se utiliza **Stratified K-Fold Cross-Validation** con:

* 5 folds.
* Mezcla aleatoria de los datos.
* `random_state=42`.
* Distribución de clases preservada mediante estratificación.

En cada fold se realiza el siguiente procedimiento:

1. Separación de los datos de entrenamiento y prueba.
2. Imputación de `age`.
3. Estandarización de las características.
4. Separación de un 15 % del entrenamiento para validación.
5. Aplicación de SMOTE únicamente sobre el conjunto de entrenamiento.
6. Entrenamiento de la red neuronal.
7. Aplicación de `EarlyStopping`.
8. Evaluación sobre el conjunto de prueba del fold.

Se utiliza `EarlyStopping` con una paciencia de 10 épocas y restauración de los mejores pesos.

El entrenamiento tiene un máximo de 100 épocas y utiliza un `batch_size` de 32.

## Métricas

Para comparar los modelos se calculan:

* Accuracy: porcentaje de predicciones correctamente clasificadas sobre el total
* Precision: proporción de predicciones positivas que fueron correctas.
* Recall: proporción de casos positivos reales que fueron identificados correctamente.
* F1-Score: media armónica entre Precision y Recall.

Los resultados obtenidos en los cinco folds se resumen mediante:

**media ± desviación estándar**

Esto permite observar tanto el rendimiento promedio como la variabilidad del modelo entre diferentes particiones de los datos.

## Visualización de resultados

El notebook genera dos visualizaciones principales.

### Matrices de confusión

Se generan matrices de confusión normalizadas por fila para cada modelo.

Además del porcentaje correspondiente, se muestran los valores absolutos de las predicciones.

Archivo generado:

```text
matrices_confusion.png
```

### Comparación de modelos

Se genera un gráfico comparativo de Accuracy, Precision, Recall y F1-Score para los tres modelos.

Las barras representan el rendimiento promedio y las barras de error representan la **desviación estándar obtenida durante los cinco folds**.

Archivo generado:

```text
comparacion_modelos.png
```

## Estructura del proyecto

```text
├── titanic_neural_networks.ipynb
├── matrices_confusion.png
├── comparacion_modelos.png
├── requirements.txt
├── README.md
└── .gitignore
```

## Instalación y ejecución

### 1. Clonar el repositorio

Clonar el repositorio desde GitHub o descargarlo como archivo ZIP.

### 2. Crear un entorno virtual

Desde la carpeta del proyecto:

```bash
python -m venv .venv
```

### 3. Activar el entorno virtual

En Windows:

```bash
.venv\Scripts\activate
```

### 4. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 5. Iniciar Jupyter Notebook

```bash
jupyter notebook
```

### 6. Ejecutar el notebook

Abrir:

```text
titanic_neural_networks.ipynb
```

y ejecutar todas las celdas en orden.

Al finalizar se generarán:

* Las métricas de evaluación de los tres modelos.
* El resumen de media y desviación estándar.
* Las matrices de confusión.
* El gráfico comparativo de métricas.

## Objetivos del proyecto

* Practicar el desarrollo de redes neuronales con TensorFlow/Keras.
* Aplicar técnicas de preprocesamiento de datos.
* Trabajar con variables categóricas mediante One-Hot Encoding.
* Gestionar valores faltantes mediante imputación.
* Aplicar estandarización de características.
* Trabajar con conjuntos de datos desbalanceados mediante SMOTE.
* Evitar la contaminación del conjunto de prueba durante la validación.
* Utilizar validación cruzada estratificada.
* Implementar Early Stopping.
* Comparar diferentes arquitecturas y optimizadores.
* Analizar Accuracy, Precision, Recall y F1-Score.
* Interpretar matrices de confusión.
* Analizar la variabilidad de los resultados mediante media y desviación estándar.
* Favorecer la reproducibilidad mediante semillas y versiones de dependencias.

