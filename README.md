# Boston Housing Data Analysis

Este proyecto se centra en el análisis del conjunto de datos de Boston Housing utilizando el modelo CRISP-DM (Cross-Industry Standard Process for Data Mining). El análisis se realiza mediante un Jupyter Notebook (`boston-houses-data-analysis.ipynb`) y se apoya en módulos Python (`data_generator.py`, `analysis.py`, `visualization.py`). Además, se incluye un notebook sandbox (`sandbox.ipynb`) para pruebas y experimentación.

## Estructura del Proyecto

- **boston-houses-data-analysis.ipynb**: Contiene el análisis principal del conjunto de datos de Boston Housing. Sigue las fases del modelo CRISP-DM, desde la comprensión de los datos hasta la evaluación del modelo.
- **sandbox.ipynb**: Un espacio para pruebas y experimentación con el código y los datos.
- **data_generator.py**: Módulo para la carga, limpieza, preprocesamiento y generación de subconjuntos de datos.
- **analysis.py**: Módulo para el entrenamiento, evaluación y optimización de modelos de regresión (Linear, Ridge y Lasso).
- **visualization.py**: Módulo para la visualización de datos y resultados, incluyendo matrices de correlación y curvas de aprendizaje.

## Descripción del Conjunto de Datos

El conjunto de datos de Boston Housing contiene información sobre diversas características de viviendas en diferentes áreas de Boston. Este conjunto de datos es ampliamente utilizado para tareas de regresión y análisis de precios de viviendas.

### Características del Conjunto de Datos

- **CRIM**: Tasa de criminalidad per cápita por ciudad.
- **ZN**: Proporción de terrenos residenciales zonificados para lotes de más de 25,000 pies cuadrados.
- **INDUS**: Proporción de acres comerciales no minoristas por ciudad.
- **CHAS**: Variable ficticia del río Charles (1 si el tramo limita con el río; 0 en caso contrario).
- **NOX**: Concentración de óxidos nítricos (partes por 10 millones).
- **RM**: Número medio de habitaciones por vivienda.
- **AGE**: Proporción de unidades ocupadas por sus propietarios construidas antes de 1940.
- **DIS**: Distancias ponderadas a cinco centros de empleo de Boston.
- **RAD**: Índice de accesibilidad a carreteras radiales.
- **TAX**: Tasa de impuesto a la propiedad por cada $10,000.
- **PTRATIO**: Proporción alumno-maestro por ciudad.
- **B**: Proporción de personas de ascendencia afroamericana por ciudad.
- **LSTAT**: Porcentaje de población de estatus socioeconómico bajo.
- **MEDV**: Valor medio de las viviendas ocupadas por sus propietarios en $1000's (variable objetivo).

## Análisis y Modelado

El análisis sigue las fases del modelo CRISP-DM:

1. **Comprensión del Negocio**: Entender el contexto y los objetivos del análisis.
2. **Comprensión de los Datos**: Explorar y describir los datos, incluyendo estadísticas básicas y visualizaciones.
3. **Preparación de los Datos**: Limpiar y preprocesar los datos, manejar valores nulos y crear subconjuntos de datos.
4. **Modelado**: Entrenar modelos de regresión (Linear, Ridge, Lasso) y optimizar hiperparámetros.
5. **Evaluación**: Evaluar el desempeño de los modelos utilizando métricas como MSE y R².
6. **Despliegue**: (Si es aplicable) Implementar el modelo en un entorno de producción.

### Resultados Clave

- **Correlación**: LSTAT (-0.74) y RM (0.70) son variables cruciales para predecir el valor de las viviendas.
- **Multicolinealidad**: Alta correlación entre TAX y RAD (0.91) sugiere eliminación o combinación de estas variables.
- **Underfitting**: Los modelos muestran signos de underfitting, sugiriendo la necesidad de modelos más complejos o más características.
