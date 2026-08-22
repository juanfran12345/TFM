# Análisis y Predicción de la Contaminación Atmosférica en Ciudades Españolas

Repositorio correspondiente al desarrollo de un **Trabajo Fin de Máster (TFM)** centrado en el análisis y predicción de las concentraciones de diferentes contaminantes atmosféricos en ciudades españolas.

El proyecto integra información histórica de **calidad del aire**, **meteorología**, **imágenes satelitales**, **características geoespaciales** y **variables socioeconómicas** con el objetivo de construir y comparar diferentes modelos de predicción de series temporales.

Las cuatro familias de modelos estudiadas son:

- **Prophet**
- **SARIMAX**
- **LSTM**
- **NeuralProphet**

El periodo principal de estudio comprende los años **2013-2024**.

---

## Índice

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Ciudades Analizadas](#ciudades-analizadas)
- [Contaminantes Analizados](#contaminantes-analizados)
- [Fuentes de Datos](#fuentes-de-datos)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Características Principales](#características-principales)
- [Ingesta y Preprocesamiento](#ingesta-y-preprocesamiento)
- [Análisis Exploratorio de los Datos](#análisis-exploratorio-de-los-datos)
- [Selección de Ciudades mediante Clustering](#selección-de-ciudades-mediante-clustering)
- [Modelado](#modelado)
- [Métricas de Evaluación](#métricas-de-evaluación)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Orden de Ejecución](#orden-de-ejecución)
- [Datasets Principales](#datasets-principales)
- [Figuras](#figuras)
- [Funciones Auxiliares](#funciones-auxiliares)
- [Reproducibilidad](#reproducibilidad)
- [Consideraciones Computacionales](#consideraciones-computacionales)
- [Solución de Problemas](#solución-de-problemas)

---

# Descripción del Proyecto

La contaminación atmosférica constituye uno de los principales problemas ambientales y de salud pública actuales. Disponer de herramientas capaces de anticipar su evolución puede resultar de gran utilidad para la vigilancia y gestión de la calidad del aire.

Este proyecto desarrolla un flujo completo de trabajo que incluye:

1. **Recolección de datos** desde diferentes fuentes.
2. **Preprocesamiento y limpieza** de los registros.
3. **Integración de información** de distinta naturaleza y granularidad temporal.
4. **Análisis exploratorio de los datos**.
5. **Caracterización de ciudades mediante clustering**.
6. **Selección de ciudades representativas**.
7. **Entrenamiento y optimización de modelos de series temporales**.
8. **Comparación del rendimiento de los modelos**.
9. **Evaluación de su capacidad de generalización**.

Los datos utilizados incluyen información sobre:

- Concentraciones de contaminantes atmosféricos.
- Condiciones meteorológicas.
- Vegetación.
- Superficie construida.
- Características geográficas.
- Población.
- PIB.
- Número de vehículos.

---

# Ciudades Analizadas

Durante las fases de recolección, preprocesamiento y análisis exploratorio se estudian **12 ciudades españolas**:

- A Coruña
- Albacete
- Alicante/Alacant
- Barcelona
- Bilbao
- Madrid
- Murcia
- Santa Cruz de Tenerife
- Sevilla
- Valencia
- Valladolid
- Zaragoza

Posteriormente, mediante técnicas de **clustering**, se identifican diferentes perfiles de ciudad.

Para reducir el coste computacional del modelado manteniendo representados perfiles urbanos diferentes, se seleccionan finalmente cuatro ciudades:

- **Barcelona**
- **Murcia**
- **Sevilla**
- **Madrid**

Estas cuatro ciudades son las utilizadas durante la fase completa de entrenamiento y comparación de modelos.

---

# Contaminantes Analizados

Durante el modelado se estudian los siguientes ocho contaminantes:

| Contaminante | Descripción |
|---|---|
| **CO** | Monóxido de carbono |
| **NO** | Óxido nítrico |
| **NO2** | Dióxido de nitrógeno |
| **NOx** | Óxidos de nitrógeno |
| **C6H6** | Benceno |
| **O3** | Ozono |
| **PM10** | Partículas en suspensión de diámetro inferior o igual a 10 µm |
| **SO2** | Dióxido de azufre |

Los archivos originales obtenidos de la European Environment Agency pueden contener adicionalmente registros de otros contaminantes, como **PM2.5**, aunque estos no forman parte de las ocho variables objetivo utilizadas en el proceso completo de modelado.

---

# Fuentes de Datos

El proyecto integra información procedente de diferentes fuentes.

## European Environment Agency (EEA)

La **European Environment Agency (EEA)** constituye la fuente principal de los registros históricos de contaminación atmosférica.

Los datos originales descargados se almacenan en formato Parquet dentro de:

```text
datasets/eea_parquet/
```

Los archivos se encuentran organizados por ciudad, contaminante y estación de medida.

---

## Open-Meteo

Los datos meteorológicos se obtienen mediante la API de **Open-Meteo**.

Entre las variables utilizadas se encuentran:

- Temperatura.
- Humedad relativa.
- Precipitación.
- Lluvia.
- Nieve.
- Presión atmosférica.
- Cobertura nubosa.
- Velocidad del viento.
- Radiación solar.
- Altura de la capa límite planetaria (PBLH).

Los archivos obtenidos se almacenan en:

```text
datasets/archivos_clima/
```

---

## NASA AppEEARS y MODIS

La plataforma **AppEEARS** permite obtener información procedente de productos MODIS.

En el proyecto se utilizan dos índices:

- **NDVI**: Normalized Difference Vegetation Index.
- **NDBI**: Normalized Difference Built-up Index.

Los archivos originales, las peticiones, los metadatos y los resultados procesados se encuentran en:

```text
datasets/AppEEARS/
```

Entre los archivos procesados principales se encuentran:

```text
índice_vegetación.csv
índice_edificación.csv
```

---

## Eurostat

Las variables socioeconómicas se obtienen principalmente de **Eurostat**.

Se utilizan:

- Población.
- PIB.
- Número de vehículos por cada 1000 habitantes.

Los archivos correspondientes se almacenan en:

```text
datasets/archivos_socioeconómicos/
```

---

## Información geoespacial

También se construye un conjunto de datos con características propias de cada ciudad:

- Latitud.
- Longitud.
- Altitud.
- Superficie.
- Distancia al mar.
- Factor cuenca o Topographic Position Index (TPI).

El resultado de esta fase se almacena en:

```text
datasets/archivo_ciudades/ciudades.csv
```

---

# Estructura del Proyecto

La estructura principal del repositorio es la siguiente:

```text
TFM/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── datasets/
│   │
│   ├── AppEEARS/
│   │   ├── índice_edificación.csv
│   │   ├── índice_vegetación.csv
│   │   ├── TFM-Urban-Indices-Spain-request.json
│   │   ├── TFM-Urban-Indices-Spain-MOD09A1-061-results.csv
│   │   ├── TFM-Urban-Indices-Spain-MOD13Q1-061-results.csv
│   │   └── ...
│   │
│   ├── archivo_ciudades/
│   │   └── ciudades.csv
│   │
│   ├── archivo_ciudades_socioeconomicos/
│   │   └── dataset_ciudades_socioeconomico.csv
│   │
│   ├── archivos_clima/
│   │   ├── A_Coruña/
│   │   ├── Albacete/
│   │   ├── Alicante_Alacant/
│   │   ├── Barcelona/
│   │   ├── Bilbao/
│   │   ├── Madrid/
│   │   ├── Murcia/
│   │   ├── Santa_Cruz_de_Tenerife/
│   │   ├── Sevilla/
│   │   ├── Valencia/
│   │   ├── Valladolid/
│   │   └── Zaragoza/
│   │
│   ├── archivos_cont_clima/
│   │   ├── A_Coruña.csv
│   │   ├── Albacete.csv
│   │   ├── Alicante-Alacant.csv
│   │   ├── Barcelona.csv
│   │   ├── Bilbao.csv
│   │   ├── Madrid.csv
│   │   ├── Murcia.csv
│   │   ├── Santa_Cruz_de_Tenerife.csv
│   │   ├── Sevilla.csv
│   │   ├── Valencia.csv
│   │   ├── Valladolid.csv
│   │   └── Zaragoza.csv
│   │
│   ├── archivos_cont_clima_indices/
│   │   ├── A_Coruña.csv
│   │   ├── Albacete.csv
│   │   ├── Alicante-Alacant.csv
│   │   ├── Barcelona.csv
│   │   ├── Bilbao.csv
│   │   ├── Madrid.csv
│   │   ├── Murcia.csv
│   │   ├── Santa_Cruz_de_Tenerife.csv
│   │   ├── Sevilla.csv
│   │   ├── Valencia.csv
│   │   ├── Valladolid.csv
│   │   └── Zaragoza.csv
│   │
│   ├── archivos_csv/
│   │   └── ES/
│   │       ├── A_Coruña/
│   │       ├── Albacete/
│   │       ├── Alicante-Alacant/
│   │       ├── Barcelona/
│   │       ├── Bilbao/
│   │       ├── Madrid/
│   │       ├── Murcia/
│   │       ├── Santa_Cruz_de_Tenerife/
│   │       ├── Sevilla/
│   │       ├── Valencia/
│   │       ├── Valladolid/
│   │       └── Zaragoza/
│   │
│   ├── archivos_socioeconómicos/
│   │   ├── población.csv
│   │   ├── PIB.csv
│   │   ├── vehículos.csv
│   │   └── dataset_socioeconómico.csv
│   │
│   ├── eda_archivos_cont_clima_indices/
│   │   └── dataset_cont_clima_indices_limpio.csv
│   │
│   └── eea_parquet/
│       └── ES/
│           ├── A_Coruña/
│           ├── Albacete/
│           ├── Alicante-Alacant/
│           ├── Barcelona/
│           ├── Bilbao/
│           ├── Madrid/
│           ├── Murcia/
│           ├── Santa_Cruz_de_Tenerife/
│           ├── Sevilla/
│           ├── Valencia/
│           ├── Valladolid/
│           └── Zaragoza/
│
├── figuras/
│   │
│   ├── EDA/
│   │   ├── correlaciones/
│   │   ├── estacionalidad anual/
│   │   ├── estacionalidad mensual/
│   │   ├── estacionalidad horaria/
│   │   └── heatmap_media_contaminantes_por_ciudad.png
│   │
│   ├── EDA2/
│   │   └── clusters_ciudades_pca.png
│   │
│   ├── Modelado/
│   │   ├── Barcelona/
│   │   ├── Madrid/
│   │   ├── Murcia/
│   │   └── Sevilla/
│   │
│   └── extra/
│       ├── hiperparametros_lstm_barras.png
│       ├── hiperparametros_neuralprophet_barras.png
│       ├── hiperparametros_prophet_barras.png
│       ├── hiperparametros_sarimax_barras.png
│       ├── mapa_calor_mape_lstm.png
│       ├── mapa_calor_mape_neuralprophet.png
│       ├── mapa_calor_mape_prophet.png
│       ├── mapa_calor_mape_sarimax.png
│       ├── mapa_calor_nrmse_lstm.png
│       ├── mapa_calor_nrmse_neuralprophet.png
│       ├── mapa_calor_nrmse_prophet.png
│       ├── mapa_calor_nrmse_sarimax.png
│       ├── mejor_modelo_mape_ciudad_contaminante.png
│       └── mapa_calor_mape_generalizacion.png
│
└── scripts/
    │
    ├── Ingesta y Preprocesamiento de datos/
    │   ├── 1.Ciudades.ipynb
    │   ├── 2.Air Quality dataset.ipynb
    │   ├── 3.Open Meteo.ipynb
    │   ├── 4.Unión.ipynb
    │   ├── 5.AppEEARS.ipynb
    │   ├── 6.Unión de registros demográficos.ipynb
    │   ├── 7.Interpolación de contaminantes, registros atmosfericos e indices.ipynb
    │   └── 8.Unión de ciudades y datos demográficos.ipynb
    │
    ├── Exploración de los datos/
    │   ├── EDA.ipynb
    │   └── EDA2.ipynb
    │
    ├── Modelado/
    │   ├── 1. Madrid/
    │   ├── 2. Barcelona/
    │   ├── 3. Murcia/
    │   └── 4. Sevilla/
    │
    ├── utils.py
    └── extra.ipynb
```

> El árbol anterior muestra únicamente los elementos principales. Algunas carpetas, especialmente `eea_parquet` y `Modelado`, contienen un número elevado de archivos que se han omitido para facilitar la lectura.

---

# Características Principales

## Integración de múltiples fuentes de información

El proyecto construye un conjunto de datos combinando información de naturaleza muy diferente:

1. Concentraciones históricas de contaminantes.
2. Variables meteorológicas.
3. Índices de vegetación y superficie construida.
4. Información geográfica.
5. Información socioeconómica.

---

## Series temporales horarias

El conjunto de datos principal presenta una frecuencia **horaria**.

El periodo analizado comprende:

```text
01/01/2013 - 31/12/2024
```

Para las ciudades con cobertura completa, esto supone **105.192 observaciones horarias**.

---

## Aprendizaje no supervisado

Se utilizan técnicas de aprendizaje no supervisado para caracterizar las ciudades:

- **K-Means**
- **Silhouette Score**
- **Análisis de Componentes Principales (PCA)**

Estas técnicas permiten reducir las doce ciudades iniciales a cuatro ciudades representativas para el proceso completo de modelado.

---

## Cuatro familias de modelos

Para cada combinación de ciudad y contaminante se comparan:

- Prophet
- SARIMAX
- LSTM
- NeuralProphet

La estructura permite estudiar:

```text
4 ciudades
× 8 contaminantes
× 4 familias de modelos
= 128 combinaciones de modelado
```

---

# Ingesta y Preprocesamiento

Los notebooks correspondientes a esta fase se encuentran en:

```text
scripts/Ingesta y Preprocesamiento de datos/
```

Los archivos están numerados siguiendo el orden general del procesamiento.

## 1.Ciudades.ipynb

Construye el conjunto de datos con las características geoespaciales de las ciudades.

Entre las variables obtenidas se encuentran:

- Latitud.
- Longitud.
- Superficie.
- Altitud.
- Distancia al mar.
- TPI.

Resultado principal:

```text
datasets/archivo_ciudades/ciudades.csv
```

---

## 2.Air Quality dataset.ipynb

Procesa los datos originales de contaminación atmosférica obtenidos de la EEA.

Los archivos originales se encuentran en:

```text
datasets/eea_parquet/
```

Durante esta fase se procesan los diferentes registros disponibles para cada ciudad, contaminante y estación.

---

## 3.Open Meteo.ipynb

Obtiene los datos meteorológicos mediante Open-Meteo para las doce ciudades analizadas.

Los resultados se almacenan en:

```text
datasets/archivos_clima/
```

---

## 4.Unión.ipynb

Realiza la primera integración entre:

- Contaminantes atmosféricos.
- Variables meteorológicas.

El resultado es un archivo por ciudad dentro de:

```text
datasets/archivos_cont_clima/
```

---

## 5.AppEEARS.ipynb

Procesa los datos obtenidos mediante AppEEARS y los productos MODIS utilizados en el proyecto.

Se obtienen las series correspondientes a:

- NDVI.
- NDBI.

Los archivos asociados se encuentran en:

```text
datasets/AppEEARS/
```

---

## 6.Unión de registros demográficos.ipynb

Procesa e integra la información socioeconómica.

Las variables principales son:

- Población.
- PIB.
- Vehículos por cada 1000 habitantes.

Los archivos se encuentran en:

```text
datasets/archivos_socioeconómicos/
```

---

## 7.Interpolación de contaminantes, registros atmosfericos e indices.ipynb

Integra los índices NDVI y NDBI con los registros horarios de contaminación y meteorología.

Los datos presentan diferentes frecuencias:

- Contaminantes: horaria.
- Meteorología: horaria.
- NDVI: aproximadamente cada 16 días.
- NDBI: aproximadamente cada 8 días.

Por este motivo se realiza un proceso de interpolación temporal para obtener valores horarios compatibles con el resto de las variables.

El principal método utilizado es **PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)**.

Los resultados se almacenan en:

```text
datasets/archivos_cont_clima_indices/
```

---

## 8.Unión de ciudades y datos demográficos.ipynb

Integra las características:

- Geoespaciales.
- Socioeconómicas.

Resultado principal:

```text
datasets/archivo_ciudades_socioeconomicos/
└── dataset_ciudades_socioeconomico.csv
```

Este dataset se utiliza posteriormente en el análisis de clustering.

---

# Análisis Exploratorio de los Datos

Los notebooks utilizados para el análisis exploratorio se encuentran en:

```text
scripts/Exploración de los datos/
```

## EDA.ipynb

Notebook principal para el análisis de los registros de contaminación, meteorología e índices urbanos.

Entre los análisis realizados se encuentran:

1. Análisis estructural.
2. Cobertura temporal.
3. Detección de valores ausentes.
4. Detección y corrección de valores inválidos.
5. Estadísticos descriptivos.
6. Concentraciones medias por ciudad.
7. Evolución anual.
8. Evolución mensual.
9. Evolución horaria.
10. Matrices de correlación.

Las figuras generadas se almacenan principalmente en:

```text
figuras/EDA/
```

---

## EDA2.ipynb

Notebook destinado al análisis de las características geoespaciales y socioeconómicas de las ciudades.

Incluye:

- Estandarización de variables.
- K-Means.
- Comparación mediante Silhouette Score.
- PCA.
- Representación de clusters.
- Análisis de los loadings de las componentes principales.

La principal representación generada se encuentra en:

```text
figuras/EDA2/clusters_ciudades_pca.png
```

---

# Selección de Ciudades mediante Clustering

El clustering se realiza a partir del dataset:

```text
datasets/archivo_ciudades_socioeconomicos/
└── dataset_ciudades_socioeconomico.csv
```

Se utiliza el algoritmo **K-Means** sobre variables estandarizadas.

Se comparan diferentes valores del número de clusters utilizando el **Silhouette Score**.

La solución seleccionada divide las ciudades en cuatro grupos.

Posteriormente, se escoge una ciudad representativa de cada cluster teniendo en cuenta también la disponibilidad y cobertura de los contaminantes.

Las ciudades seleccionadas son:

| Cluster | Ciudad seleccionada |
|---|---|
| Cluster 1 | Barcelona |
| Cluster 2 | Murcia |
| Cluster 3 | Sevilla |
| Cluster 4 | Madrid |

Estas cuatro ciudades son las utilizadas durante el modelado completo.

---

# Modelado

Los notebooks correspondientes al modelado se encuentran en:

```text
scripts/Modelado/
```

La estructura está organizada primero por **ciudad** y después por **contaminante**.

Por ejemplo:

```text
scripts/Modelado/
└── 1. Madrid/
    └── 1. CO/
        ├── Prophet_CO_Madrid.ipynb
        ├── SARIMAX_CO_Madrid.ipynb
        ├── LSTM_CO_Madrid.ipynb
        └── NeuralProphet_CO_Madrid.ipynb
```

La misma estructura se repite para:

### Ciudades

```text
1. Madrid
2. Barcelona
3. Murcia
4. Sevilla
```

### Contaminantes

```text
1. CO
2. NO
3. NO2
4. NOx
5. C6H6
6. O3
7. PM10
8. SO2
```

---

## Prophet

Prophet es un modelo de series temporales basado principalmente en la descomposición de la serie en:

- Tendencia.
- Estacionalidad.
- Festivos.
- Regresores externos.

Para cada combinación ciudad-contaminante se realiza una búsqueda de hiperparámetros utilizando los conjuntos de entrenamiento y validación.

---

## SARIMAX

SARIMAX amplía los modelos SARIMA permitiendo incorporar variables exógenas.

Se estudian:

- Parte autorregresiva.
- Diferenciación.
- Media móvil.
- Componente estacional.
- Variables externas.

La búsqueda inicial de órdenes se realiza mediante `auto_arima`, considerando criterios como:

- AIC.
- AICc.
- BIC.

Posteriormente, los candidatos se comparan utilizando el conjunto de validación.

---

## LSTM

Las redes LSTM permiten aprender dependencias temporales mediante una arquitectura recurrente con memoria.

Antes del entrenamiento se realizan diferentes procesos:

- Escalado de los datos.
- Construcción de ventanas temporales.
- Creación de secuencias.
- Separación cronológica.
- Optimización de hiperparámetros.

Entre los hiperparámetros estudiados se encuentran:

- Número de unidades.
- Número de capas.
- Dropout.
- Optimizador.
- Número de épocas.
- Tamaño del batch.
- Longitud de la ventana temporal.

---

## NeuralProphet

NeuralProphet combina componentes interpretables de Prophet con arquitecturas basadas en redes neuronales.

El modelo permite incorporar:

- Tendencia.
- Estacionalidad.
- Festivos.
- Autorregresión.
- Regresores rezagados.
- Regresores conocidos en el futuro.

---

# División Temporal de los Datos

Al tratarse de series temporales, los datos **no se dividen aleatoriamente**.

Para las series con cobertura suficiente, la división general es:

| Conjunto | Periodo |
|---|---|
| Entrenamiento | 2013-2020 |
| Validación | 2021-2022 |
| Prueba | 2023-2024 |

Para contaminantes cuyos registros comienzan más tarde, estos intervalos se adaptan a la disponibilidad real de datos.

El conjunto de **prueba** se reserva para la evaluación final y no participa en la selección de hiperparámetros.

---

# Métricas de Evaluación

Se utilizan principalmente tres métricas.

## MSE

**Mean Squared Error** o error cuadrático medio:

```text
MSE = promedio((valor_real - valor_predicho)²)
```

Penaliza especialmente los errores de gran magnitud.

---

## MAPE

**Mean Absolute Percentage Error** o error porcentual absoluto medio.

Permite expresar el error de predicción en términos porcentuales.

Es especialmente útil para comparar contaminantes que presentan diferentes escalas de concentración.

---

## NRMSE

**Normalized Root Mean Squared Error**.

Normaliza el RMSE respecto a la magnitud de la variable observada y permite comparar errores entre series de diferente escala.

Además, al derivarse del error cuadrático, penaliza especialmente los errores puntuales elevados.

---

# Análisis Global de Resultados

El notebook:

```text
scripts/extra.ipynb
```

se utiliza para realizar diferentes análisis conjuntos de los resultados obtenidos durante el modelado.

Entre las figuras generadas se encuentran:

### Hiperparámetros

```text
figuras/extra/hiperparametros_prophet_barras.png
figuras/extra/hiperparametros_sarimax_barras.png
figuras/extra/hiperparametros_lstm_barras.png
figuras/extra/hiperparametros_neuralprophet_barras.png
```

### Errores MAPE

```text
figuras/extra/mapa_calor_mape_prophet.png
figuras/extra/mapa_calor_mape_sarimax.png
figuras/extra/mapa_calor_mape_lstm.png
figuras/extra/mapa_calor_mape_neuralprophet.png
```

### Errores NRMSE

```text
figuras/extra/mapa_calor_nrmse_prophet.png
figuras/extra/mapa_calor_nrmse_sarimax.png
figuras/extra/mapa_calor_nrmse_lstm.png
figuras/extra/mapa_calor_nrmse_neuralprophet.png
```

### Comparación final

```text
figuras/extra/mejor_modelo_mape_ciudad_contaminante.png
```

### Generalización

```text
figuras/extra/mapa_calor_mape_generalizacion.png
```

---

# Requisitos Previos

Para ejecutar el proyecto es necesario disponer de:

- **Python**
- **pip**
- **Jupyter Notebook o JupyterLab**
- **Git** (opcional, únicamente necesario para clonar el repositorio)

También se recomienda utilizar un **entorno virtual** para evitar conflictos entre las dependencias del proyecto.

---

# Instalación

## 1. Clonar o descargar el repositorio

Si el repositorio se encuentra disponible mediante Git:

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
```

Sustituir `<URL_DEL_REPOSITORIO>` y `<NOMBRE_DEL_REPOSITORIO>` por los valores correspondientes.

También es posible descargar el proyecto en formato ZIP y descomprimirlo.

---

## 2. Crear un entorno virtual

Se recomienda utilizar un entorno virtual para aislar las dependencias.

### Windows / PowerShell

```powershell
python -m venv .venv
```

Activar:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows / CMD

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Instalar las dependencias

Con el entorno virtual activado:

```bash
pip install -r requirements.txt
```

Las dependencias necesarias para ejecutar el proyecto se encuentran definidas en:

```text
requirements.txt
```

---

## 4. Iniciar Jupyter

Para utilizar los notebooks:

```bash
jupyter notebook
```

o:

```bash
jupyter lab
```

A continuación, navegar hasta el notebook que se desee ejecutar.

---

# Uso

El proyecto puede utilizarse de dos formas diferentes.

## Opción 1: Reproducir todo el proceso

Permite comenzar desde los datos originales y reproducir progresivamente:

```text
Extracción
    ↓
Preprocesamiento
    ↓
Integración
    ↓
Limpieza
    ↓
EDA
    ↓
Clustering
    ↓
Modelado
    ↓
Comparación de resultados
```

Para ello se deben ejecutar los notebooks siguiendo el orden descrito en la siguiente sección.

---

## Opción 2: Utilizar directamente los datasets procesados

Si no se desea repetir toda la fase de extracción e integración, se pueden utilizar directamente los datasets finales almacenados en el repositorio.

Para el EDA y modelado:

```text
datasets/eda_archivos_cont_clima_indices/
└── dataset_cont_clima_indices_limpio.csv
```

Para clustering:

```text
datasets/archivo_ciudades_socioeconomicos/
└── dataset_ciudades_socioeconomico.csv
```

Esta opción permite comenzar directamente con los análisis y modelos.

---

# Orden de Ejecución

Para reproducir el proyecto completo se recomienda seguir el siguiente orden.

## Fase 1: Ingesta y Preprocesamiento

```text
scripts/Ingesta y Preprocesamiento de datos/
```

Ejecutar:

```text
1.Ciudades.ipynb
        ↓
2.Air Quality dataset.ipynb
        ↓
3.Open Meteo.ipynb
        ↓
4.Unión.ipynb
        ↓
5.AppEEARS.ipynb
        ↓
6.Unión de registros demográficos.ipynb
        ↓
7.Interpolación de contaminantes, registros atmosfericos e indices.ipynb
        ↓
8.Unión de ciudades y datos demográficos.ipynb
```

---

## Fase 2: Análisis Exploratorio

Ejecutar:

```text
scripts/Exploración de los datos/EDA.ipynb
```

y posteriormente:

```text
scripts/Exploración de los datos/EDA2.ipynb
```

---

## Fase 3: Modelado

Seleccionar una ciudad:

```text
scripts/Modelado/1. Madrid/
scripts/Modelado/2. Barcelona/
scripts/Modelado/3. Murcia/
scripts/Modelado/4. Sevilla/
```

Después seleccionar un contaminante y ejecutar los cuatro modelos.

Por ejemplo:

```text
scripts/Modelado/1. Madrid/1. CO/
```

Contiene:

```text
Prophet_CO_Madrid.ipynb
SARIMAX_CO_Madrid.ipynb
LSTM_CO_Madrid.ipynb
NeuralProphet_CO_Madrid.ipynb
```

Los modelos correspondientes a distintas ciudades y contaminantes pueden ejecutarse independientemente una vez se dispone del dataset procesado.

---

## Fase 4: Comparación de Resultados

Finalmente se utiliza:

```text
scripts/extra.ipynb
```

para generar las comparaciones globales y las figuras resumen.

---

# Datasets Principales

El repositorio contiene numerosos datasets intermedios. Los más importantes para comprender o reutilizar el proyecto son los siguientes.

## Dataset principal de contaminación

```text
datasets/eda_archivos_cont_clima_indices/
└── dataset_cont_clima_indices_limpio.csv
```

Contiene los registros horarios utilizados para el análisis y modelado, integrando:

- Variables temporales.
- Contaminantes.
- Variables meteorológicas.
- NDVI.
- NDBI.

---

## Dataset de ciudades

```text
datasets/archivo_ciudades_socioeconomicos/
└── dataset_ciudades_socioeconomico.csv
```

Contiene una observación por ciudad con variables como:

- Latitud.
- Longitud.
- Altitud.
- Superficie.
- Distancia al mar.
- TPI.
- Población representativa.
- PIB representativo.
- Vehículos por cada 1000 habitantes.

Este archivo constituye la entrada principal del análisis de clustering.

---

# Figuras

Las figuras generadas durante el proyecto se almacenan en:

```text
figuras/
```

## EDA

```text
figuras/EDA/
```

Contiene:

- Matrices de correlación.
- Evoluciones anuales.
- Evoluciones mensuales.
- Evoluciones horarias.
- Concentraciones medias de contaminantes.

---

## Clustering y PCA

```text
figuras/EDA2/
```

Contiene principalmente:

```text
clusters_ciudades_pca.png
```

---

## Modelado

```text
figuras/Modelado/
```

Organizada por:

- Madrid.
- Barcelona.
- Murcia.
- Sevilla.

Incluye principalmente los análisis de estacionalidad realizados antes del entrenamiento de los modelos.

---

## Comparación Global

```text
figuras/extra/
```

Contiene:

- Mapas de calor de errores.
- Comparaciones entre modelos.
- Frecuencia de selección de hiperparámetros.
- Resultados de generalización.

---

# Funciones Auxiliares

El archivo:

```text
scripts/utils.py
```

contiene diferentes funciones reutilizadas a lo largo de los notebooks.

Su finalidad es evitar duplicar código y centralizar procedimientos comunes relacionados con:

- Carga de datos.
- Filtrado.
- Representación gráfica.
- Evaluación.
- Cálculo de métricas.
- Procesamiento auxiliar para los modelos.

Se recomienda **no modificar la ubicación de este archivo**, ya que diferentes notebooks pueden utilizar rutas relativas para importarlo.

---

# Reproducibilidad

El proyecto utiliza diferentes mecanismos para favorecer la reproducibilidad de los experimentos.

Entre ellos:

- División cronológica de los datos.
- Separación entre entrenamiento, validación y prueba.
- Ajuste de escaladores únicamente con los datos de entrenamiento.
- Uso de semillas aleatorias.
- Conservación del orden temporal en los modelos secuenciales.
- Centralización de funciones auxiliares.
- Almacenamiento de datasets intermedios.
- Conservación de las configuraciones evaluadas durante el modelado.

No obstante, algunos modelos de aprendizaje profundo pueden producir pequeñas variaciones dependiendo de:

- Hardware utilizado.
- Versiones de las librerías.
- Sistema operativo.
- Backend de cálculo.
- Operaciones no completamente deterministas.

---

# Consideraciones Computacionales

La ejecución completa del proyecto puede tener un coste computacional elevado.

Esto se debe principalmente a la cantidad de combinaciones existentes:

```text
4 ciudades
×
8 contaminantes
×
4 familias de modelos
```

Además, cada notebook puede evaluar numerosas combinaciones de hiperparámetros.

Los modelos con un mayor coste computacional son especialmente:

- LSTM.
- NeuralProphet.
- Determinadas configuraciones SARIMAX.

Por ello, antes de reproducir todo el proceso se recomienda comprobar el funcionamiento del entorno utilizando una única combinación.

Por ejemplo:

```text
scripts/Modelado/1. Madrid/1. CO/
```

---

# Flujo General del Proyecto

El flujo principal seguido durante el TFM puede representarse de la siguiente manera:

```text
                     ┌───────────────────────┐
                     │          EEA          │
                     │ Calidad del aire      │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Procesamiento de      │
                     │ contaminantes         │
                     └───────────┬───────────┘
                                 │
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
 ┌─────────────┐          ┌─────────────┐         ┌──────────────┐
 │ Open-Meteo  │          │  AppEEARS   │         │ Información  │
 │ Meteorología│          │ NDVI / NDBI │         │ socioeconómica│
 └──────┬──────┘          └──────┬──────┘         └──────┬───────┘
        │                        │                        │
        ▼                        ▼                        │
 ┌─────────────────────────────────────┐                 │
 │ Contaminación + Meteorología        │                 │
 │ + NDVI + NDBI                       │                 │
 └──────────────────┬──────────────────┘                 │
                    │                                    │
                    ▼                                    ▼
         ┌───────────────────────┐          ┌─────────────────────────┐
         │ Dataset horario limpio│          │ Dataset de ciudades     │
         └───────────┬───────────┘          │ geoespacial + socioecon.│
                     │                      └────────────┬────────────┘
                     ▼                                   │
               ┌───────────┐                             ▼
               │    EDA    │                      ┌──────────────┐
               └─────┬─────┘                      │ K-Means + PCA│
                     │                            └──────┬───────┘
                     │                                   │
                     │                                   ▼
                     │                         Selección de ciudades
                     │                                   │
                     └──────────────────┬────────────────┘
                                        │
                                        ▼
                                 ┌──────────────┐
                                 │   Modelado   │
                                 └──────┬───────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
          Prophet                   SARIMAX                    LSTM
              │                         │                         │
              └──────────────┬──────────┴───────────┬────────────┘
                             │                      │
                             │               NeuralProphet
                             │                      │
                             └──────────┬───────────┘
                                        │
                                        ▼
                              Comparación de modelos
                                        │
                                        ▼
                                MAPE / NRMSE
                                        │
                                        ▼
                          Evaluación de generalización
```

---

# Solución de Problemas

## Error `ModuleNotFoundError`

Comprobar que el entorno virtual se encuentra activado.

Después instalar nuevamente las dependencias:

```bash
pip install -r requirements.txt
```

---

## Error `FileNotFoundError`

Los notebooks utilizan rutas relativas.

Por ello, comprobar que:

1. Se mantiene la estructura original del repositorio.
2. El notebook no ha sido trasladado a otra carpeta.
3. Las carpetas `datasets`, `scripts` y `figuras` mantienen sus nombres y ubicaciones.

---

## Error al importar `utils.py`

El archivo se encuentra en:

```text
scripts/utils.py
```

Algunos notebooks ubicados en subdirectorios utilizan rutas relativas para acceder a este archivo.

Si el notebook se mueve de ubicación, estas rutas pueden dejar de funcionar.

---

## El kernel de Jupyter no utiliza el entorno correcto

Con el entorno virtual activado puede registrarse como kernel de Jupyter mediante:

```bash
pip install ipykernel
```

Después:

```bash
python -m ipykernel install --user --name tfm --display-name "Python - TFM"
```

A continuación, seleccionar el kernel correspondiente desde Jupyter Notebook o JupyterLab.

---

## Problemas de memoria durante el modelado

Algunos modelos requieren una cantidad considerable de memoria.

En este caso se recomienda:

- Ejecutar un único notebook cada vez.
- Reiniciar el kernel después de experimentos pesados.
- Liberar modelos y variables que ya no sean necesarios.
- Evitar ejecutar simultáneamente varios modelos LSTM o NeuralProphet.

---

# Desactivar el Entorno Virtual

Cuando se termine de trabajar con el proyecto:

```bash
deactivate
```

---

# Resumen

El repositorio permite reproducir el flujo completo desarrollado durante el Trabajo Fin de Máster:

```text
Extracción de datos
        ↓
Preprocesamiento
        ↓
Integración de fuentes
        ↓
Limpieza
        ↓
Análisis exploratorio
        ↓
Clustering y PCA
        ↓
Selección de ciudades
        ↓
Prophet / SARIMAX / LSTM / NeuralProphet
        ↓
Optimización de hiperparámetros
        ↓
Evaluación
        ↓
Comparación de modelos
        ↓
Evaluación de generalización
```

La estructura del repositorio conserva tanto los datos originales como los productos intermedios y finales, permitiendo seguir el proceso completo desde la adquisición de los datos hasta la comparación final de los modelos predictivos.