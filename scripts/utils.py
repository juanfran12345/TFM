import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

import numpy as np
import pandas as pd
import itertools
from sklearn import metrics
from pathlib import Path
import random

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics
import torch
from neuralprophet import NeuralProphet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("fbprophet").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
plt.style.use("fivethirtyeight")
plt.rcParams["lines.linewidth"] = 1.5
light_style = {
    "figure.facecolor": "#d9effb",   
    "axes.facecolor": "#d9effb",
    "savefig.facecolor": "#d9effb",
    "axes.grid": True,
    "axes.grid.which": "both",
    "axes.spines.left": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.spines.bottom": True,
    "grid.color": "#a9d3f2",
    "grid.linewidth": "0.8",
    "text.color": "#333333",
    "axes.labelcolor": "#333333",
    "axes.labelweight": "black",      
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "font.size": 12,
    "axes.titleweight": "bold",      
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
}
plt.rcParams.update(light_style)
rcParams["figure.figsize"] = (18, 7)

#############################################################################################
##################################### FUNCIONES COMUNES #####################################
#############################################################################################

def CARGA_Y_FILTRO(contaminante, ciudad):
    """
    Carga el dataset limpio y filtra los datos por ciudad y contaminante.

    Solo conserva las filas en las que el contaminante seleccionado contiene
    un valor numérico válido.

    Parámetros
    ----------
    contaminante : str
        Nombre exacto de la columna del contaminante.
        Ejemplo: 'CO (mg.m-3)'.

    ciudad : str
        Nombre de la ciudad que se desea seleccionar.
        Ejemplo: 'Madrid'.

    Retorna
    -------
    pandas.DataFrame
        DataFrame filtrado por ciudad y por observaciones válidas del
        contaminante, con las columnas necesarias para el modelado.
    """

    # ==========================================================================
    # COMPROBACIONES INICIALES
    # ==========================================================================

    if not isinstance(contaminante, str):
        raise TypeError(
            "El argumento 'contaminante' debe ser una cadena de texto."
        )

    if not isinstance(ciudad, str):
        raise TypeError(
            "El argumento 'ciudad' debe ser una cadena de texto."
        )

    if not contaminante.strip():
        raise ValueError(
            "El argumento 'contaminante' no puede estar vacío."
        )

    if not ciudad.strip():
        raise ValueError(
            "El argumento 'ciudad' no puede estar vacío."
        )

    # ==========================================================================
    # DEFINICIÓN DE LAS RUTAS
    # ==========================================================================

    # Ruta raíz del repositorio.
    # utils.py se encuentra dentro de la carpeta scripts.
    BASE_PATH = Path(__file__).resolve().parent.parent

    # Carpeta que contiene el dataset limpio
    FOLDER_DATA = (
        BASE_PATH
        / "datasets"
        / "eda_archivos_cont_clima_indices"
    )

    # Ruta completa del archivo
    ruta_dataset = (
        FOLDER_DATA
        / "dataset_cont_clima_indices_limpio.csv"
    )

    # Comprobar que el archivo existe
    if not ruta_dataset.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el dataset en:\n{ruta_dataset}"
        )

    # ==========================================================================
    # CARGA DEL DATASET
    # ==========================================================================

    df = pd.read_csv(
        ruta_dataset
    )

    if df.empty:
        raise ValueError(
            "El dataset está vacío."
        )

    # Comprobar que el contaminante indicado existe
    if contaminante not in df.columns:
        raise ValueError(
            f"La columna '{contaminante}' no existe en el dataset.\n"
            f"Columnas disponibles:\n{df.columns.tolist()}"
        )

    # ==========================================================================
    # SELECCIÓN DE COLUMNAS
    # ==========================================================================

    columnas = [
        "Start",
        "End",
        contaminante,
        "city",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "snowfall",
        "surface_pressure",
        "cloudcover",
        "windspeed_10m",
        "shortwave_radiation",
        "boundary_layer_height",
        "NDVI",
        "NDBI",
        "Año",
        "Mes",
        "Dia",
        "Dia_semana",
        "Hora"
    ]

    columnas_no_encontradas = [
        columna
        for columna in columnas
        if columna not in df.columns
    ]

    if columnas_no_encontradas:
        raise ValueError(
            "No se han encontrado las siguientes columnas en el dataset:\n"
            f"{columnas_no_encontradas}"
        )

    # ==========================================================================
    # FILTRADO POR CIUDAD
    # ==========================================================================

    mascara_ciudad = (
        df["city"]
        .astype(str)
        .str.strip()
        .str.casefold()
        == ciudad.strip().casefold()
    )

    df_filtrado = df.loc[
        mascara_ciudad,
        columnas
    ].copy()

    if df_filtrado.empty:

        ciudades_disponibles = sorted(
            df["city"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

        raise ValueError(
            f"No se han encontrado registros para la ciudad '{ciudad}'.\n"
            f"Ciudades disponibles:\n{ciudades_disponibles}"
        )

    # ==========================================================================
    # FILTRADO POR VALORES VÁLIDOS DEL CONTAMINANTE
    # ==========================================================================

    # Convertimos el contaminante a formato numérico. Los valores que no puedan
    # convertirse se sustituyen por NaN.
    df_filtrado[contaminante] = pd.to_numeric(
        df_filtrado[contaminante],
        errors="coerce"
    )

    # Conservamos únicamente las filas en las que el contaminante tiene valor.
    df_filtrado = df_filtrado.dropna(
        subset=[
            contaminante
        ]
    ).copy()

    if df_filtrado.empty:
        raise ValueError(
            f"La ciudad '{ciudad}' tiene registros en el dataset, pero no "
            f"contiene valores válidos para el contaminante '{contaminante}'."
        )

    # ==========================================================================
    # ORDENACIÓN Y REINICIO DEL ÍNDICE
    # ==========================================================================

    # Ordenamos cronológicamente cuando la columna Start está disponible.
    df_filtrado["Start"] = pd.to_datetime(
        df_filtrado["Start"],
        errors="coerce"
    )

    df_filtrado["End"] = pd.to_datetime(
        df_filtrado["End"],
        errors="coerce"
    )

    df_filtrado = df_filtrado.sort_values(
        by="Start"
    )

    df_filtrado.reset_index(
        drop=True,
        inplace=True
    )

    return df_filtrado


def GRAFICAR_ESTACIONALIDADES(
    df,
    contaminante,
    ciudad,
    nombre_contaminante=None,
    unidad=None
):
    """
    Representa las estacionalidades anual, semanal y diaria de un contaminante,
    diferenciando los perfiles de cada año disponible.

    Solo se consideran los años que contienen observaciones válidas para el
    contaminante seleccionado.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame previamente filtrado por ciudad. Debe contener las columnas
        'Año', 'Mes', 'Dia_semana', 'Hora' y la columna del contaminante.

    contaminante : str
        Nombre exacto de la columna del contaminante en el DataFrame.
        Ejemplo: 'CO (mg.m-3)'.

    ciudad : str
        Nombre de la ciudad que aparecerá en el título general.
        Ejemplo: 'Madrid'.

    nombre_contaminante : str, opcional
        Nombre abreviado utilizado en los títulos.
        Ejemplo: 'CO'. Si no se indica, se utiliza el texto anterior al primer
        paréntesis del nombre de la columna.

    unidad : str, opcional
        Unidad que aparecerá en el eje vertical.
        Ejemplo: 'mg/m³'. Si no se indica, se intenta extraer de la columna.

    Retorna
    -------
    tuple
        Figura de Matplotlib y array de ejes.
    """

    # ==========================================================================
    # COMPROBACIONES INICIALES
    # ==========================================================================

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "El argumento 'df' debe ser un DataFrame de pandas."
        )

    if not isinstance(contaminante, str):
        raise TypeError(
            "El argumento 'contaminante' debe ser una cadena de texto."
        )

    if not isinstance(ciudad, str):
        raise TypeError(
            "El argumento 'ciudad' debe ser una cadena de texto."
        )

    columnas_necesarias = [
        contaminante,
        "Año",
        "Mes",
        "Dia_semana",
        "Hora"
    ]

    columnas_no_encontradas = [
        columna
        for columna in columnas_necesarias
        if columna not in df.columns
    ]

    if columnas_no_encontradas:
        raise ValueError(
            "No se han encontrado las siguientes columnas en el DataFrame:\n"
            f"{columnas_no_encontradas}"
        )

    if df.empty:
        raise ValueError(
            "El DataFrame está vacío y no se pueden representar "
            "las estacionalidades."
        )

    # ==========================================================================
    # FILTRADO DE OBSERVACIONES VÁLIDAS
    # ==========================================================================

    # Conservamos únicamente las filas con un año válido y un valor válido
    # del contaminante. De esta forma, no aparecen en la leyenda años para
    # los que el contaminante no tiene observaciones.
    df_grafico = df.dropna(
        subset=[
            "Año",
            contaminante
        ]
    ).copy()

    if df_grafico.empty:
        raise ValueError(
            f"No existen observaciones válidas para el contaminante "
            f"'{contaminante}'."
        )

    # Convertimos el contaminante a formato numérico por seguridad
    df_grafico[contaminante] = pd.to_numeric(
        df_grafico[contaminante],
        errors="coerce"
    )

    # Eliminamos posibles valores que no hayan podido convertirse a número
    df_grafico = df_grafico.dropna(
        subset=[
            contaminante
        ]
    ).copy()

    if df_grafico.empty:
        raise ValueError(
            f"No existen valores numéricos válidos para el contaminante "
            f"'{contaminante}'."
        )

    # ==========================================================================
    # NOMBRE Y UNIDAD DEL CONTAMINANTE
    # ==========================================================================

    if nombre_contaminante is None:
        nombre_contaminante = contaminante.split("(")[0].strip()

    if unidad is None:

        if "(" in contaminante and ")" in contaminante:

            unidad = contaminante.split(
                "(",
                1
            )[1].rsplit(
                ")",
                1
            )[0]

            unidad = (
                unidad
                .replace("ug.m-3", "µg/m³")
                .replace("mg.m-3", "mg/m³")
                .replace(".m-3", "/m³")
                .replace("m-3", "m⁻³")
                .replace("ug", "µg")
            )

        else:
            unidad = ""

    etiqueta_eje_y = nombre_contaminante

    if unidad:
        etiqueta_eje_y = (
            f"{nombre_contaminante} ({unidad})"
        )

    # ==========================================================================
    # ETIQUETAS DE LOS EJES
    # ==========================================================================

    meses_nombres = [
        "Ene",
        "Feb",
        "Mar",
        "Abr",
        "May",
        "Jun",
        "Jul",
        "Ago",
        "Sep",
        "Oct",
        "Nov",
        "Dic"
    ]

    dias_nombres = [
        "Lun",
        "Mar",
        "Mié",
        "Jue",
        "Vie",
        "Sáb",
        "Dom"
    ]

    # ==========================================================================
    # AÑOS DISPONIBLES
    # ==========================================================================

    # Se obtienen únicamente a partir de las observaciones válidas del
    # contaminante seleccionado.
    años = sorted(
        df_grafico["Año"].unique()
    )

    if not años:
        raise ValueError(
            "No existen años con observaciones válidas del contaminante."
        )

    # Paleta con un color diferente para cada año
    colores = sns.color_palette(
        "husl",
        n_colors=len(años)
    )

    paleta_años = dict(
        zip(
            años,
            colores
        )
    )

    # ==========================================================================
    # CREACIÓN DE LA FIGURA
    # ==========================================================================

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(15, 15)
    )

    # ==========================================================================
    # 1. ESTACIONALIDAD ANUAL
    # ==========================================================================

    df_mes_año = (
        df_grafico
        .dropna(
            subset=[
                "Mes"
            ]
        )
        .groupby(
            [
                "Año",
                "Mes"
            ],
            as_index=False
        )[contaminante]
        .mean()
    )

    sns.lineplot(
        data=df_mes_año,
        x="Mes",
        y=contaminante,
        hue="Año",
        palette=paleta_años,
        marker="o",
        linewidth=1.7,
        alpha=0.85,
        ax=axes[0],
        legend=False
    )

    axes[0].set_title(
        f"Estacionalidad anual: perfil mensual del "
        f"{nombre_contaminante} por año",
        fontsize=13,
        fontweight="bold"
    )

    axes[0].set_xticks(
        range(
            1,
            13
        )
    )

    axes[0].set_xticklabels(
        meses_nombres
    )

    axes[0].set_xlabel(
        "Mes del año"
    )

    axes[0].set_ylabel(
        etiqueta_eje_y
    )

    axes[0].grid(
        True,
        linestyle=":",
        alpha=0.6
    )

    # ==========================================================================
    # 2. ESTACIONALIDAD SEMANAL
    # ==========================================================================

    df_dia_año = (
        df_grafico
        .dropna(
            subset=[
                "Dia_semana"
            ]
        )
        .groupby(
            [
                "Año",
                "Dia_semana"
            ],
            as_index=False
        )[contaminante]
        .mean()
    )

    sns.lineplot(
        data=df_dia_año,
        x="Dia_semana",
        y=contaminante,
        hue="Año",
        palette=paleta_años,
        marker="o",
        linewidth=1.7,
        alpha=0.85,
        ax=axes[1],
        legend=False
    )

    axes[1].set_title(
        f"Estacionalidad semanal: perfil del "
        f"{nombre_contaminante} por día de la semana y año",
        fontsize=13,
        fontweight="bold"
    )

    axes[1].set_xticks(
        range(7)
    )

    axes[1].set_xticklabels(
        dias_nombres
    )

    axes[1].set_xlabel(
        "Día de la semana"
    )

    axes[1].set_ylabel(
        etiqueta_eje_y
    )

    axes[1].grid(
        True,
        linestyle=":",
        alpha=0.6
    )

    # ==========================================================================
    # 3. ESTACIONALIDAD DIARIA
    # ==========================================================================

    df_hora_año = (
        df_grafico
        .dropna(
            subset=[
                "Hora"
            ]
        )
        .groupby(
            [
                "Año",
                "Hora"
            ],
            as_index=False
        )[contaminante]
        .mean()
    )

    sns.lineplot(
        data=df_hora_año,
        x="Hora",
        y=contaminante,
        hue="Año",
        palette=paleta_años,
        linewidth=1.7,
        alpha=0.85,
        ax=axes[2],
        legend=False
    )

    axes[2].set_title(
        f"Estacionalidad diaria: perfil horario del "
        f"{nombre_contaminante} por año",
        fontsize=13,
        fontweight="bold"
    )

    axes[2].set_xticks(
        range(24)
    )

    axes[2].set_xlabel(
        "Hora del día"
    )

    axes[2].set_ylabel(
        etiqueta_eje_y
    )

    axes[2].grid(
        True,
        linestyle=":",
        alpha=0.6
    )

    # ==========================================================================
    # LEYENDA COMÚN
    # ==========================================================================

    leyenda = [
        plt.Line2D(
            [0],
            [0],
            color=paleta_años[año],
            linewidth=2,
            label=str(año)
        )
        for año in años
    ]

    fig.legend(
        handles=leyenda,
        title="Año",
        bbox_to_anchor=(
            0.99,
            0.5
        ),
        loc="center left",
        frameon=True
    )

    # ==========================================================================
    # TÍTULO GENERAL Y AJUSTES FINALES
    # ==========================================================================

    fig.suptitle(
        f"Análisis de las estacionalidades del "
        f"{nombre_contaminante} en {ciudad}",
        fontsize=16,
        fontweight="bold",
        y=1.01
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            0.91,
            1
        ]
    )

    plt.show()

    return fig, axes


def EVALUAR_METRICAS(y_real, y_predicho, num_parametros):
    """
    Calcula y muestra las principales métricas de evaluación
    de un modelo de predicción.

    Parámetros
    ----------
    y_real : array-like
        Valores reales de la variable objetivo.

    y_predicho : array-like
        Valores estimados por el modelo.

    num_parametros : int
        Número de parámetros o variables explicativas utilizados
        para calcular el coeficiente de determinación ajustado.

    Retorna
    -------
    dict
        Diccionario con MAE, MSE, RMSE, NRMSE y MAPE.
    """

    # Número de observaciones
    n = len(y_real)

    # Cálculo de las métricas
    mae = metrics.mean_absolute_error(y_real, y_predicho)
    mse = metrics.mean_squared_error(y_real, y_predicho)
    rmse = np.sqrt(mse)

    # RMSE normalizado respecto a la media de los valores reales
    nrmse = (rmse / np.mean(y_real)) * 100

    def MAPE(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mape = MAPE(y_real, y_predicho)

    # Resultados
    resultados = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'NRMSE': nrmse,
        'MAPE': mape
    }

    # Mostramos los resultados
    print('Resultados de la evaluación del modelo')
    print('---------------------------------------')
    print(f'Error absoluto medio (MAE): {mae:.6f}')
    print(f'Error cuadrático medio (MSE): {mse:.6f}')
    print(f'Raíz del error cuadrático medio (RMSE): {rmse:.6f}')
    print(f'Error porcentual absoluto medio (MAPE): {mape:.2f} %')
    print(f'Raíz del error cuadrático medio normalizada (NRMSE): {nrmse:.2f} %')

    return resultados


def CALCULAR_VIF(datos, variables):
    """
    Calcula el factor de inflación de la varianza de un conjunto
    de variables explicativas.
    """

    X = datos[variables].replace([np.inf, -np.inf], np.nan)

    # Imputación mediante la mediana, únicamente para calcular el VIF
    imputador = SimpleImputer(strategy="median")
    X_imputado = imputador.fit_transform(X)

    # Estandarización de las variables
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X_imputado)

    vif = pd.DataFrame({
        "variable": variables,
        "VIF": [
            variance_inflation_factor(X_escalado, i)
            for i in range(X_escalado.shape[1])
        ]
    })

    return vif.sort_values("VIF", ascending=False).reset_index(drop=True)


#############################################################################################
##################################### FUNCIONES PROPHET #####################################
#############################################################################################

def BUSQUEDA_CONFIGURACIONES_PROPHET(
    configuraciones,
    variables_exogenas,
    train,
    validation,
    estacionalidades,
    capacidad=None,
    mostrar_resultado=True
):
    """
    Ajusta y evalúa diferentes configuraciones de Prophet.

    Las estacionalidades indicadas se aplican a todos los modelos, mientras
    que sus órdenes de Fourier se obtienen de cada configuración evaluada.

    Parámetros
    ----------
    configuraciones : list of dict
        Lista de configuraciones generadas a partir de param_grid.

    variables_exogenas : list of str
        Variables exógenas que se incorporarán al modelo.

    train : pandas.DataFrame
        Conjunto de entrenamiento.

    validation : pandas.DataFrame
        Conjunto de validación.

    estacionalidades : list of str
        Estacionalidades que deben utilizarse en todos los entrenamientos.

        Valores permitidos:
        - 'yearly'
        - 'weekly'
        - 'daily'

        El orden de Fourier de cada estacionalidad se obtiene de las claves
        correspondientes de cada configuración.

    capacidad : float, opcional
        Capacidad máxima para los modelos con crecimiento logístico.

    mostrar_resultado : bool, opcional
        Indica si se muestran los mejores parámetros.

    Retorna
    -------
    resultados_df : pandas.DataFrame
        Resultados de todas las configuraciones.

    resultados_correctos : pandas.DataFrame
        Configuraciones correctas ordenadas por MSE.

    mejor_resultado : pandas.Series
        Mejor configuración encontrada.

    mejor_modelo : prophet.Prophet
        Modelo entrenado con la mejor configuración.
    """

    # ==========================================================================
    # COMPROBACIONES INICIALES
    # ==========================================================================

    if not isinstance(configuraciones, list):
        raise TypeError(
            "El argumento 'configuraciones' debe ser una lista."
        )

    if not configuraciones:
        raise ValueError(
            "La lista de configuraciones está vacía."
        )

    if not isinstance(variables_exogenas, list):
        raise TypeError(
            "El argumento 'variables_exogenas' debe ser una lista."
        )

    if not isinstance(estacionalidades, list):
        raise TypeError(
            "El argumento 'estacionalidades' debe ser una lista."
        )

    if not isinstance(train, pd.DataFrame):
        raise TypeError(
            "El argumento 'train' debe ser un DataFrame de pandas."
        )

    if not isinstance(validation, pd.DataFrame):
        raise TypeError(
            "El argumento 'validation' debe ser un DataFrame de pandas."
        )

    estacionalidades_permitidas = {
        "yearly",
        "weekly",
        "daily"
    }

    estacionalidades_normalizadas = [
        estacionalidad.strip().lower()
        for estacionalidad in estacionalidades
    ]

    estacionalidades_no_validas = [
        estacionalidad
        for estacionalidad in estacionalidades_normalizadas
        if estacionalidad not in estacionalidades_permitidas
    ]

    if estacionalidades_no_validas:
        raise ValueError(
            "Se han indicado estacionalidades no válidas:\n"
            f"{estacionalidades_no_validas}\n\n"
            "Los valores permitidos son: "
            "'yearly', 'weekly' y 'daily'."
        )

    # Eliminar duplicados conservando el orden
    estacionalidades_normalizadas = list(
        dict.fromkeys(estacionalidades_normalizadas)
    )

    columnas_necesarias = [
        "ds",
        "y"
    ] + variables_exogenas

    columnas_faltantes_train = [
        columna
        for columna in columnas_necesarias
        if columna not in train.columns
    ]

    columnas_faltantes_validation = [
        columna
        for columna in columnas_necesarias
        if columna not in validation.columns
    ]

    if columnas_faltantes_train:
        raise ValueError(
            "Faltan las siguientes columnas en entrenamiento:\n"
            f"{columnas_faltantes_train}"
        )

    if columnas_faltantes_validation:
        raise ValueError(
            "Faltan las siguientes columnas en validación:\n"
            f"{columnas_faltantes_validation}"
        )

    utiliza_crecimiento_logistico = any(
        configuracion.get("growth") == "logistic"
        for configuracion in configuraciones
    )

    if utiliza_crecimiento_logistico and capacidad is None:
        raise ValueError(
            "Debes indicar 'capacidad' porque alguna configuración "
            "utiliza growth='logistic'."
        )

    parametros_obligatorios = [
        "changepoint_prior_scale",
        "changepoint_range",
        "seasonality_prior_scale",
        "seasonality_mode",
        "growth",
        "usar_festivos"
    ]

    if "yearly" in estacionalidades_normalizadas:
        parametros_obligatorios.append(
            "yearly_seasonality"
        )

    if "weekly" in estacionalidades_normalizadas:
        parametros_obligatorios.append(
            "weekly_seasonality"
        )

    if "daily" in estacionalidades_normalizadas:
        parametros_obligatorios.append(
            "daily_seasonality"
        )

    for indice, configuracion in enumerate(configuraciones):

        if not isinstance(configuracion, dict):
            raise TypeError(
                f"La configuración {indice} debe ser un diccionario."
            )

        parametros_faltantes = [
            parametro
            for parametro in parametros_obligatorios
            if parametro not in configuracion
        ]

        if parametros_faltantes:
            raise ValueError(
                f"En la configuración {indice} faltan los parámetros:\n"
                f"{parametros_faltantes}"
            )

    # ==========================================================================
    # AJUSTE Y EVALUACIÓN DE CONFIGURACIONES
    # ==========================================================================

    resultados_busqueda = []

    for params in configuraciones:

        usar_festivos = params["usar_festivos"]

        # Si la estacionalidad está activa, se utiliza el orden de Fourier
        # definido en la configuración. En caso contrario, se desactiva.
        yearly_seasonality = (
            params["yearly_seasonality"]
            if "yearly" in estacionalidades_normalizadas
            else False
        )

        weekly_seasonality = (
            params["weekly_seasonality"]
            if "weekly" in estacionalidades_normalizadas
            else False
        )

        daily_seasonality = (
            params["daily_seasonality"]
            if "daily" in estacionalidades_normalizadas
            else False
        )

        parametros_modelo = {
            "changepoint_prior_scale": params[
                "changepoint_prior_scale"
            ],
            "changepoint_range": params[
                "changepoint_range"
            ],
            "seasonality_prior_scale": params[
                "seasonality_prior_scale"
            ],
            "seasonality_mode": params[
                "seasonality_mode"
            ],
            "growth": params[
                "growth"
            ]
        }

        try:
            modelo = Prophet(
                **parametros_modelo,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality
            )

            for variable in variables_exogenas:
                modelo.add_regressor(
                    variable,
                    standardize=True
                )

            if usar_festivos:
                modelo.add_country_holidays(
                    country_name="ES"
                )

            datos_entrenamiento = train.copy()
            datos_validacion = validation.copy()

            if params["growth"] == "logistic":
                datos_entrenamiento["cap"] = capacidad
                datos_entrenamiento["floor"] = 0

                datos_validacion["cap"] = capacidad
                datos_validacion["floor"] = 0

            modelo.fit(
                datos_entrenamiento
            )

            columnas_prediccion = [
                "ds"
            ] + variables_exogenas

            if params["growth"] == "logistic":
                columnas_prediccion += [
                    "cap",
                    "floor"
                ]

            fechas_validacion = datos_validacion[
                columnas_prediccion
            ].copy()

            prediccion = modelo.predict(
                fechas_validacion
            )

            mse = mean_squared_error(
                datos_validacion["y"],
                prediccion["yhat"]
            )

            resultado = params.copy()

            # Guardamos los órdenes realmente utilizados
            resultado["yearly_seasonality"] = yearly_seasonality
            resultado["weekly_seasonality"] = weekly_seasonality
            resultado["daily_seasonality"] = daily_seasonality

            resultado["MSE_validacion"] = mse
            resultado["estado"] = "Correcto"

            resultados_busqueda.append(
                resultado
            )

        except Exception as error:

            resultado = params.copy()

            resultado["yearly_seasonality"] = yearly_seasonality
            resultado["weekly_seasonality"] = weekly_seasonality
            resultado["daily_seasonality"] = daily_seasonality

            resultado["MSE_validacion"] = np.nan
            resultado["estado"] = str(error)

            resultados_busqueda.append(
                resultado
            )

    # ==========================================================================
    # SELECCIÓN DE LA MEJOR CONFIGURACIÓN
    # ==========================================================================

    resultados_df = pd.DataFrame(
        resultados_busqueda
    )

    resultados_correctos = resultados_df[
        resultados_df["estado"] == "Correcto"
    ].copy()

    resultados_correctos = resultados_correctos.sort_values(
        by="MSE_validacion",
        ascending=True
    ).reset_index(
        drop=True
    )

    if resultados_correctos.empty:

        errores = resultados_df[
            ["estado"]
        ].drop_duplicates()

        raise RuntimeError(
            "Ninguna configuración se ha ejecutado correctamente.\n\n"
            f"Errores encontrados:\n{errores.to_string(index=False)}"
        )

    mejor_resultado = resultados_correctos.iloc[0].copy()

    # ==========================================================================
    # ENTRENAMIENTO DEL MEJOR MODELO
    # ==========================================================================

    yearly_mejor = (
        int(mejor_resultado["yearly_seasonality"])
        if "yearly" in estacionalidades_normalizadas
        else False
    )

    weekly_mejor = (
        int(mejor_resultado["weekly_seasonality"])
        if "weekly" in estacionalidades_normalizadas
        else False
    )

    daily_mejor = (
        int(mejor_resultado["daily_seasonality"])
        if "daily" in estacionalidades_normalizadas
        else False
    )

    parametros_mejor_modelo = {
        "changepoint_prior_scale": mejor_resultado[
            "changepoint_prior_scale"
        ],
        "changepoint_range": mejor_resultado[
            "changepoint_range"
        ],
        "seasonality_prior_scale": mejor_resultado[
            "seasonality_prior_scale"
        ],
        "seasonality_mode": mejor_resultado[
            "seasonality_mode"
        ],
        "growth": mejor_resultado[
            "growth"
        ]
    }

    mejor_modelo = Prophet(
        **parametros_mejor_modelo,
        yearly_seasonality=yearly_mejor,
        weekly_seasonality=weekly_mejor,
        daily_seasonality=daily_mejor
    )

    for variable in variables_exogenas:
        mejor_modelo.add_regressor(
            variable,
            standardize=True
        )

    if mejor_resultado["usar_festivos"]:
        mejor_modelo.add_country_holidays(
            country_name="ES"
        )

    datos_mejor_modelo = train.copy()

    if mejor_resultado["growth"] == "logistic":
        datos_mejor_modelo["cap"] = capacidad
        datos_mejor_modelo["floor"] = 0

    mejor_modelo.fit(
        datos_mejor_modelo
    )

    # ==========================================================================
    # PRESENTACIÓN DEL RESULTADO
    # ==========================================================================

    if mostrar_resultado:

        print("Estacionalidades utilizadas:\n")

        if estacionalidades_normalizadas:
            for estacionalidad in estacionalidades_normalizadas:
                print(f"- {estacionalidad}")
        else:
            print()

        print("\nMejores parámetros:\n")

        for parametro in parametros_obligatorios:
            valor = mejor_resultado[parametro]

            if (
                parametro == "yearly_seasonality"
                and "yearly" not in estacionalidades_normalizadas
            ):
                valor = "No incluida"

            elif (
                parametro == "weekly_seasonality"
                and "weekly" not in estacionalidades_normalizadas
            ):
                valor = "No incluida"

            elif (
                parametro == "daily_seasonality"
                and "daily" not in estacionalidades_normalizadas
            ):
                valor = "No incluida"

            print(
                f"{parametro}: {valor}"
            )

        print(
            f"\nMSE de validación: "
            f"{mejor_resultado['MSE_validacion']:.8f}"
        )

    return (
        resultados_df,
        resultados_correctos,
        mejor_resultado,
        mejor_modelo
    )


def ENTRENAR_EVALUAR_PROPHET(
    train,
    validation,
    variables_exogenas,
    mejores_parametros,
    capacidad=None,
    factor_capacidad=1.20,
    floor=0,
    mostrar_resultado=True
):
    """
    Entrena y evalúa un modelo Prophet a partir de unos hiperparámetros
    previamente seleccionados.

    Las estacionalidades anual, semanal y diaria se activan únicamente
    cuando aparecen en el diccionario 'mejores_parametros'. Si alguna de
    ellas no aparece, se desactiva mediante False.

    Parámetros
    ----------
    train : pandas.DataFrame
        Conjunto de entrenamiento. Debe contener las columnas 'ds', 'y'
        y las variables exógenas indicadas.

    validation : pandas.DataFrame
        Conjunto de validación. Debe contener las columnas 'ds', 'y'
        y las variables exógenas indicadas.

    variables_exogenas : list of str
        Lista con los nombres de las variables exógenas que se incorporarán
        al modelo mediante add_regressor.

    mejores_parametros : dict
        Diccionario con los mejores hiperparámetros del modelo.

        Debe contener:

        - changepoint_prior_scale
        - changepoint_range
        - seasonality_prior_scale
        - seasonality_mode
        - growth
        - usar_festivos

        Puede contener:

        - yearly_seasonality
        - weekly_seasonality
        - daily_seasonality

        Si alguna estacionalidad no aparece, se establece como False.

    capacidad : float, opcional
        Capacidad máxima utilizada cuando growth='logistic'. Si no se
        proporciona, se calcula como:

        train['y'].max() * factor_capacidad

    factor_capacidad : float, opcional
        Factor multiplicativo utilizado para calcular la capacidad cuando
        growth='logistic' y no se proporciona una capacidad explícita.
        Por defecto es 1.20.

    floor : float, opcional
        Límite inferior utilizado cuando growth='logistic'.
        Por defecto es 0.

    mostrar_resultado : bool, opcional
        Si es True, muestra los parámetros utilizados y las métricas.
        Por defecto es True.

    Retorna
    -------
    modelo_prophet : prophet.Prophet
        Modelo Prophet entrenado.

    prediccion_validacion : pandas.DataFrame
        DataFrame generado por Prophet con las predicciones sobre validación.

    metricas_validacion : dict
        Diccionario con las métricas calculadas mediante EVALUAR_METRICAS.

    train_prophet_final : pandas.DataFrame
        Copia del conjunto de entrenamiento utilizada por Prophet.

    validation_prophet_final : pandas.DataFrame
        Copia del conjunto de validación utilizada para la predicción.
    """

    # ==========================================================================
    # COMPROBACIONES INICIALES
    # ==========================================================================

    if not isinstance(train, pd.DataFrame):
        raise TypeError(
            "El argumento 'train' debe ser un DataFrame de pandas."
        )

    if not isinstance(validation, pd.DataFrame):
        raise TypeError(
            "El argumento 'validation' debe ser un DataFrame de pandas."
        )

    if not isinstance(variables_exogenas, list):
        raise TypeError(
            "El argumento 'variables_exogenas' debe ser una lista."
        )

    if not isinstance(mejores_parametros, dict):
        raise TypeError(
            "El argumento 'mejores_parametros' debe ser un diccionario."
        )

    if not isinstance(factor_capacidad, (int, float)):
        raise TypeError(
            "El argumento 'factor_capacidad' debe ser numérico."
        )

    if factor_capacidad <= 0:
        raise ValueError(
            "El argumento 'factor_capacidad' debe ser mayor que cero."
        )

    parametros_obligatorios = [
        "changepoint_prior_scale",
        "changepoint_range",
        "seasonality_prior_scale",
        "seasonality_mode",
        "growth",
        "usar_festivos"
    ]

    parametros_faltantes = [
        parametro
        for parametro in parametros_obligatorios
        if parametro not in mejores_parametros
    ]

    if parametros_faltantes:
        raise ValueError(
            "Faltan los siguientes parámetros obligatorios en "
            f"'mejores_parametros':\n{parametros_faltantes}"
        )

    columnas_necesarias = [
        "ds",
        "y"
    ] + variables_exogenas

    columnas_faltantes_train = [
        columna
        for columna in columnas_necesarias
        if columna not in train.columns
    ]

    columnas_faltantes_validation = [
        columna
        for columna in columnas_necesarias
        if columna not in validation.columns
    ]

    if columnas_faltantes_train:
        raise ValueError(
            "Faltan las siguientes columnas en el conjunto de "
            f"entrenamiento:\n{columnas_faltantes_train}"
        )

    if columnas_faltantes_validation:
        raise ValueError(
            "Faltan las siguientes columnas en el conjunto de "
            f"validación:\n{columnas_faltantes_validation}"
        )

    valores_growth_permitidos = {
        "linear",
        "logistic",
        "flat"
    }

    growth = mejores_parametros["growth"]

    if growth not in valores_growth_permitidos:
        raise ValueError(
            f"El valor de 'growth' no es válido: {growth}.\n"
            "Los valores permitidos son: 'linear', 'logistic' y 'flat'."
        )

    if not isinstance(
        mejores_parametros["usar_festivos"],
        (bool, np.bool_)
    ):
        raise TypeError(
            "El parámetro 'usar_festivos' debe ser booleano."
        )

    # ==========================================================================
    # CORRECCIÓN OPCIONAL DE LA ERRATA 'WEAKLY'
    # ==========================================================================

    mejores_parametros = mejores_parametros.copy()

    if (
        "weakly_seasonality" in mejores_parametros
        and "weekly_seasonality" not in mejores_parametros
    ):
        mejores_parametros["weekly_seasonality"] = (
            mejores_parametros.pop("weakly_seasonality")
        )

    # ==========================================================================
    # ESTACIONALIDADES
    # ==========================================================================

    yearly_seasonality = mejores_parametros.get(
        "yearly_seasonality",
        False
    )

    weekly_seasonality = mejores_parametros.get(
        "weekly_seasonality",
        False
    )

    daily_seasonality = mejores_parametros.get(
        "daily_seasonality",
        False
    )

    estacionalidades = {
        "yearly_seasonality": yearly_seasonality,
        "weekly_seasonality": weekly_seasonality,
        "daily_seasonality": daily_seasonality
    }

    for nombre, valor in estacionalidades.items():

        if isinstance(valor, (bool, np.bool_)):
            continue

        if not isinstance(valor, (int, np.integer)):
            raise TypeError(
                f"El parámetro '{nombre}' debe ser un entero positivo "
                "o False."
            )

        if valor <= 0:
            raise ValueError(
                f"El parámetro '{nombre}' debe ser mayor que cero "
                "cuando la estacionalidad está activada."
            )

    # ==========================================================================
    # COPIA DE LOS CONJUNTOS
    # ==========================================================================

    train_prophet_final = train.copy()
    validation_prophet_final = validation.copy()

    # ==========================================================================
    # CRECIMIENTO LOGÍSTICO
    # ==========================================================================

    if growth == "logistic":

        if capacidad is None:
            capacidad = (
                train_prophet_final["y"].max()
                * factor_capacidad
            )

        if not isinstance(capacidad, (int, float, np.number)):
            raise TypeError(
                "El argumento 'capacidad' debe ser numérico."
            )

        if capacidad <= floor:
            raise ValueError(
                "La capacidad debe ser mayor que el valor de 'floor'."
            )

        if train_prophet_final["y"].max() > capacidad:
            raise ValueError(
                "La capacidad es inferior al máximo observado en el "
                "conjunto de entrenamiento."
            )

        train_prophet_final["cap"] = capacidad
        train_prophet_final["floor"] = floor

        validation_prophet_final["cap"] = capacidad
        validation_prophet_final["floor"] = floor

    # ==========================================================================
    # CREACIÓN DEL MODELO
    # ==========================================================================

    modelo_prophet = Prophet(
        changepoint_prior_scale=mejores_parametros[
            "changepoint_prior_scale"
        ],
        changepoint_range=mejores_parametros[
            "changepoint_range"
        ],
        seasonality_prior_scale=mejores_parametros[
            "seasonality_prior_scale"
        ],
        seasonality_mode=mejores_parametros[
            "seasonality_mode"
        ],
        growth=growth,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality
    )

    # ==========================================================================
    # INCORPORACIÓN DE VARIABLES EXÓGENAS
    # ==========================================================================

    for variable in variables_exogenas:
        modelo_prophet.add_regressor(
            variable,
            standardize=True
        )

    # ==========================================================================
    # INCORPORACIÓN DE FESTIVOS
    # ==========================================================================

    if mejores_parametros["usar_festivos"]:
        modelo_prophet.add_country_holidays(
            country_name="ES"
        )

    # ==========================================================================
    # ENTRENAMIENTO
    # ==========================================================================

    modelo_prophet.fit(
        train_prophet_final
    )

    # ==========================================================================
    # PREPARACIÓN DE LOS DATOS DE VALIDACIÓN
    # ==========================================================================

    columnas_prediccion = [
        "ds"
    ] + variables_exogenas

    if growth == "logistic":
        columnas_prediccion += [
            "cap",
            "floor"
        ]

    fechas_validacion = validation_prophet_final[
        columnas_prediccion
    ].copy()

    # ==========================================================================
    # PREDICCIÓN
    # ==========================================================================

    prediccion_validacion = modelo_prophet.predict(
        fechas_validacion
    )

    # ==========================================================================
    # EVALUACIÓN
    # ==========================================================================

    num_parametros = len(variables_exogenas)

    metricas_validacion = EVALUAR_METRICAS(
        y_real=validation_prophet_final["y"],
        y_predicho=prediccion_validacion["yhat"],
        num_parametros=num_parametros
    )

    return (
        modelo_prophet,
        prediccion_validacion,
        metricas_validacion,
        train_prophet_final,
        validation_prophet_final
    )


#############################################################################################
################################## FUNCIONES NEURALPROPHET ##################################
#############################################################################################

def BUSQUEDA_CONFIGURACIONES_NEURALPROPHET(
    configuraciones,
    train,
    validation,
    estacionalidades,
    regresores_futuros,
    regresores_retardados,
    mostrar_resultado=True
):
    """
    Ajusta y evalúa diferentes configuraciones de NeuralProphet.

    Las estacionalidades indicadas en el argumento 'estacionalidades'
    se aplican a todos los modelos. El número de términos de Fourier
    utilizado en cada entrenamiento se obtiene de cada configuración.

    Parámetros
    ----------
    configuraciones : list of dict
        Lista de configuraciones que se desean evaluar.

    train : pandas.DataFrame
        Conjunto de entrenamiento. Debe contener las columnas 'ds', 'y',
        los regresores futuros y los regresores retardados.

    validation : pandas.DataFrame
        Conjunto de validación. Debe contener las columnas 'ds', 'y',
        los regresores futuros y los regresores retardados.

    estacionalidades : list of str
        Lista con las estacionalidades que se aplicarán a todos los modelos.

        Valores permitidos:

        - 'yearly'
        - 'weekly'
        - 'daily'

        Ejemplo:

        ['yearly', 'weekly', 'daily']

    regresores_futuros : list of str
        Lista con los regresores cuyos valores son conocidos durante
        el periodo de predicción.

    regresores_retardados : list of str
        Lista con los regresores que se incorporarán mediante sus rezagos.

    mostrar_resultado : bool, opcional
        Si es True, muestra las estacionalidades utilizadas, los mejores
        parámetros y el MSE de validación.

    Retorna
    -------
    resultados_df : pandas.DataFrame
        DataFrame con los resultados de todas las configuraciones.

    resultados_correctos : pandas.DataFrame
        Configuraciones ejecutadas correctamente, ordenadas de menor
        a mayor MSE.

    mejor_resultado : pandas.Series or None
        Mejor configuración encontrada. Devuelve None si ninguna
        configuración se ejecuta correctamente.

    mejor_modelo : neuralprophet.NeuralProphet or None
        Modelo entrenado con la mejor configuración. Devuelve None si
        ninguna configuración se ejecuta correctamente.
    """

    # ==========================================================================
    # NORMALIZACIÓN DE LAS ESTACIONALIDADES
    # ==========================================================================

    estacionalidades_normalizadas = [
        estacionalidad.strip().lower()
        for estacionalidad in estacionalidades
    ]

    estacionalidades_normalizadas = list(
        dict.fromkeys(estacionalidades_normalizadas)
    )

    # ==========================================================================
    # PARÁMETROS OBLIGATORIOS
    # ==========================================================================

    parametros_obligatorios = [
        "changepoints_range",
        "trend_reg",
        "seasonality_mode",
        "seasonality_reg",
        "n_lags",
        "ar_layers",
        "lagged_reg_layers",
        "epochs",
        "batch_size",
        "usar_festivos"
    ]

    # El orden de Fourier solo es obligatorio para las estacionalidades
    # indicadas en el argumento estacionalidades.
    if "yearly" in estacionalidades_normalizadas:
        parametros_obligatorios.append(
            "yearly_seasonality"
        )

    if "weekly" in estacionalidades_normalizadas:
        parametros_obligatorios.append(
            "weekly_seasonality"
        )

    if "daily" in estacionalidades_normalizadas:
        parametros_obligatorios.append(
            "daily_seasonality"
        )

    for indice, configuracion in enumerate(configuraciones):

        parametros_faltantes = [
            parametro
            for parametro in parametros_obligatorios
            if parametro not in configuracion
        ]

        if parametros_faltantes:
            raise ValueError(
                f"En la configuración {indice} faltan los parámetros:\n"
                f"{parametros_faltantes}"
            )

    # ==========================================================================
    # AJUSTE Y EVALUACIÓN DE LAS CONFIGURACIONES
    # ==========================================================================

    resultados_busqueda = []

    for numero_configuracion, params in enumerate(
        configuraciones,
        start=1
    ):

        # ----------------------------------------------------------------------
        # Estacionalidades y órdenes de Fourier utilizados
        # ----------------------------------------------------------------------

        yearly_seasonality = (
            params["yearly_seasonality"]
            if "yearly" in estacionalidades_normalizadas
            else False
        )

        weekly_seasonality = (
            params["weekly_seasonality"]
            if "weekly" in estacionalidades_normalizadas
            else False
        )

        daily_seasonality = (
            params["daily_seasonality"]
            if "daily" in estacionalidades_normalizadas
            else False
        )

        # Indicador de utilización de festivos
        usar_festivos = params["usar_festivos"]

        # ----------------------------------------------------------------------
        # Parámetros admitidos por NeuralProphet
        # ----------------------------------------------------------------------

        parametros_modelo = {

            # Tendencia
            "growth": "linear",
            "changepoints_range": params["changepoints_range"],
            "trend_reg": params["trend_reg"],

            # Estacionalidad
            "seasonality_mode": params["seasonality_mode"],
            "seasonality_reg": params["seasonality_reg"],
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality,

            # Autorregresión
            "n_lags": params["n_lags"],
            "n_forecasts": 1,
            "ar_layers": params["ar_layers"],
            "lagged_reg_layers": params["lagged_reg_layers"],

            # Entrenamiento
            "epochs": params["epochs"],
            "batch_size": params["batch_size"]
        }

        try:

            # ------------------------------------------------------------------
            # Fijación de las semillas
            # ------------------------------------------------------------------

            np.random.seed(42)
            random.seed(42)
            torch.manual_seed(42)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

            # ------------------------------------------------------------------
            # Copia de los conjuntos
            # ------------------------------------------------------------------

            datos_entrenamiento = train.copy()
            datos_validacion = validation.copy()

            # ------------------------------------------------------------------
            # Conversión de las fechas
            # ------------------------------------------------------------------

            datos_entrenamiento["ds"] = pd.to_datetime(
                datos_entrenamiento["ds"]
            )

            datos_validacion["ds"] = pd.to_datetime(
                datos_validacion["ds"]
            )

            # ------------------------------------------------------------------
            # Ordenación cronológica
            # ------------------------------------------------------------------

            datos_entrenamiento = datos_entrenamiento.sort_values(
                by="ds"
            ).reset_index(
                drop=True
            )

            datos_validacion = datos_validacion.sort_values(
                by="ds"
            ).reset_index(
                drop=True
            )

            # ------------------------------------------------------------------
            # Creación del modelo
            # ------------------------------------------------------------------

            modelo = NeuralProphet(
                **parametros_modelo
            )

            # ------------------------------------------------------------------
            # Incorporación de los regresores retardados
            # ------------------------------------------------------------------

            for variable in regresores_retardados:

                modelo.add_lagged_regressor(
                    names=variable,
                    n_lags=params["n_lags"],
                    normalize="standardize"
                )

            # ------------------------------------------------------------------
            # Incorporación de los regresores futuros
            # ------------------------------------------------------------------

            for variable in regresores_futuros:

                modelo.add_future_regressor(
                    name=variable,
                    normalize="standardize",
                    mode=params["seasonality_mode"]
                )

            # ------------------------------------------------------------------
            # Incorporación de los festivos nacionales de España
            # ------------------------------------------------------------------

            if usar_festivos:

                modelo.add_country_holidays(
                    country_name="ES"
                )

            # ------------------------------------------------------------------
            # Entrenamiento
            # ------------------------------------------------------------------

            modelo.fit(
                datos_entrenamiento,
                freq="h",
                progress=None
            )

            # ------------------------------------------------------------------
            # Contexto necesario para construir los primeros rezagos
            # ------------------------------------------------------------------

            contexto_entrenamiento = datos_entrenamiento.tail(
                params["n_lags"]
            ).copy()

            datos_prediccion = pd.concat(
                [
                    contexto_entrenamiento,
                    datos_validacion
                ],
                ignore_index=True
            )

            datos_prediccion = datos_prediccion.sort_values(
                by="ds"
            ).reset_index(
                drop=True
            )

            # ------------------------------------------------------------------
            # Predicción
            # ------------------------------------------------------------------

            prediccion = modelo.predict(
                datos_prediccion
            )

            # ------------------------------------------------------------------
            # Predicciones correspondientes al periodo de validación
            # ------------------------------------------------------------------

            prediccion_validacion = prediccion[
                prediccion["ds"].isin(
                    datos_validacion["ds"]
                )
            ][
                [
                    "ds",
                    "yhat1"
                ]
            ].copy()

            # ------------------------------------------------------------------
            # Unión de valores reales y predicciones
            # ------------------------------------------------------------------

            comparacion_validacion = datos_validacion[
                [
                    "ds",
                    "y"
                ]
            ].merge(
                prediccion_validacion,
                on="ds",
                how="inner"
            )

            comparacion_validacion = comparacion_validacion.dropna(
                subset=[
                    "y",
                    "yhat1"
                ]
            ).reset_index(
                drop=True
            )

            if comparacion_validacion.empty:

                raise ValueError(
                    "NeuralProphet no ha generado predicciones válidas "
                    "para el conjunto de validación."
                )

            # ------------------------------------------------------------------
            # Cálculo del MSE
            # ------------------------------------------------------------------

            mse = mean_squared_error(
                comparacion_validacion["y"],
                comparacion_validacion["yhat1"]
            )

            # ------------------------------------------------------------------
            # Almacenamiento del resultado
            # ------------------------------------------------------------------

            resultado = params.copy()

            # Guardamos los órdenes de Fourier realmente utilizados.
            # Las estacionalidades ausentes se almacenan como False.
            resultado["yearly_seasonality"] = yearly_seasonality
            resultado["weekly_seasonality"] = weekly_seasonality
            resultado["daily_seasonality"] = daily_seasonality

            resultado["MSE_validacion"] = mse

            resultado["numero_predicciones"] = len(
                comparacion_validacion
            )

            resultado["estado"] = "Correcto"

            resultados_busqueda.append(
                resultado
            )

        except Exception as error:

            # ------------------------------------------------------------------
            # Almacenamiento del error
            # ------------------------------------------------------------------

            resultado = params.copy()

            resultado["yearly_seasonality"] = yearly_seasonality
            resultado["weekly_seasonality"] = weekly_seasonality
            resultado["daily_seasonality"] = daily_seasonality

            resultado["MSE_validacion"] = np.nan
            resultado["numero_predicciones"] = 0
            resultado["estado"] = str(error)

            resultados_busqueda.append(
                resultado
            )

            if mostrar_resultado:

                print(
                    f"Error en la configuración "
                    f"{numero_configuracion}:"
                )

                print(error)

    # ==========================================================================
    # CONVERSIÓN DE LOS RESULTADOS EN UN DATAFRAME
    # ==========================================================================

    resultados_df = pd.DataFrame(
        resultados_busqueda
    )

    # ==========================================================================
    # SELECCIÓN DE LAS CONFIGURACIONES CORRECTAS
    # ==========================================================================

    resultados_correctos = resultados_df[
        resultados_df["estado"] == "Correcto"
    ].copy()

    resultados_correctos = resultados_correctos.sort_values(
        by="MSE_validacion",
        ascending=True
    ).reset_index(
        drop=True
    )

    # ==========================================================================
    # CASO EN EL QUE NINGUNA CONFIGURACIÓN SEA CORRECTA
    # ==========================================================================

    if resultados_correctos.empty:

        mejor_resultado = None
        mejor_modelo = None

        if mostrar_resultado:

            print(
                "\nNo se ha podido ajustar correctamente "
                "ninguna configuración."
            )

            print(
                "\nErrores encontrados:\n"
            )

            for indice, fila in resultados_df.iterrows():

                print(
                    f"Configuración {indice + 1}: "
                    f"{fila['estado']}"
                )

        return (
            resultados_df,
            resultados_correctos,
            mejor_resultado,
            mejor_modelo
        )

    # ==========================================================================
    # SELECCIÓN DE LA MEJOR CONFIGURACIÓN
    # ==========================================================================

    mejor_resultado = resultados_correctos.iloc[
        0
    ].copy()

    mejores_parametros = mejor_resultado.to_dict()

    # ==========================================================================
    # ENTRENAMIENTO DEL MEJOR MODELO
    # ==========================================================================

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    datos_mejor_modelo = train.copy()

    datos_mejor_modelo["ds"] = pd.to_datetime(
        datos_mejor_modelo["ds"]
    )

    datos_mejor_modelo = datos_mejor_modelo.sort_values(
        by="ds"
    ).reset_index(
        drop=True
    )

    parametros_mejor_modelo = {

        # Tendencia
        "growth": "linear",
        "changepoints_range": mejores_parametros[
            "changepoints_range"
        ],
        "trend_reg": mejores_parametros[
            "trend_reg"
        ],

        # Estacionalidad
        "seasonality_mode": mejores_parametros[
            "seasonality_mode"
        ],
        "seasonality_reg": mejores_parametros[
            "seasonality_reg"
        ],
        "yearly_seasonality": mejores_parametros[
            "yearly_seasonality"
        ],
        "weekly_seasonality": mejores_parametros[
            "weekly_seasonality"
        ],
        "daily_seasonality": mejores_parametros[
            "daily_seasonality"
        ],

        # Autorregresión
        "n_lags": mejores_parametros[
            "n_lags"
        ],
        "n_forecasts": 1,
        "ar_layers": mejores_parametros[
            "ar_layers"
        ],
        "lagged_reg_layers": mejores_parametros[
            "lagged_reg_layers"
        ],

        # Entrenamiento
        "epochs": mejores_parametros[
            "epochs"
        ],
        "batch_size": mejores_parametros[
            "batch_size"
        ]
    }

    mejor_modelo = NeuralProphet(
        **parametros_mejor_modelo
    )

    for variable in regresores_retardados:

        mejor_modelo.add_lagged_regressor(
            names=variable,
            n_lags=mejores_parametros["n_lags"],
            normalize="standardize"
        )

    for variable in regresores_futuros:

        mejor_modelo.add_future_regressor(
            name=variable,
            normalize="standardize",
            mode=mejores_parametros["seasonality_mode"]
        )

    if mejores_parametros["usar_festivos"]:

        mejor_modelo.add_country_holidays(
            country_name="ES"
        )

    mejor_modelo.fit(
        datos_mejor_modelo,
        freq="h",
        progress=None
    )

    # ==========================================================================
    # PRESENTACIÓN DE LOS RESULTADOS
    # ==========================================================================

    if mostrar_resultado:

        print(
            "\nEstacionalidades utilizadas:"
        )

        if estacionalidades_normalizadas:

            for estacionalidad in estacionalidades_normalizadas:

                nombre_parametro = (
                    f"{estacionalidad}_seasonality"
                )

                print(
                    f"- {estacionalidad}: "
                    f"{mejor_resultado[nombre_parametro]}"
                )

        print(
            "\nMejores parámetros:\n"
        )

        parametros_mostrar = [
            "changepoints_range",
            "trend_reg",
            "seasonality_mode",
            "seasonality_reg"
        ]

        if "yearly" in estacionalidades_normalizadas:
            parametros_mostrar.append(
                "yearly_seasonality"
            )

        if "weekly" in estacionalidades_normalizadas:
            parametros_mostrar.append(
                "weekly_seasonality"
            )

        if "daily" in estacionalidades_normalizadas:
            parametros_mostrar.append(
                "daily_seasonality"
            )

        parametros_mostrar += [
            "n_lags",
            "ar_layers",
            "lagged_reg_layers",
            "epochs",
            "batch_size",
            "usar_festivos"
        ]

        for parametro in parametros_mostrar:

            print(
                f"{parametro}: "
                f"{mejor_resultado[parametro]}"
            )

        print(
            f"\nMSE de validación: "
            f"{mejor_resultado['MSE_validacion']:.8f}"
        )

    return (
        resultados_df,
        resultados_correctos,
        mejor_resultado,
        mejor_modelo
    )



def ENTRENAR_EVALUAR_NEURALPROPHET(
    train,
    validation,
    regresores_futuros,
    regresores_retardados,
    mejores_parametros
):
    """
    Entrena y evalúa un modelo NeuralProphet utilizando los mejores
    hiperparámetros obtenidos previamente.

    Las estacionalidades anual, semanal y diaria se activan únicamente
    cuando aparecen en el diccionario 'mejores_parametros'. Si alguna
    no aparece, se establece como False.

    Parámetros
    ----------
    train : pandas.DataFrame
        Conjunto de entrenamiento con las columnas 'ds', 'y' y los
        regresores utilizados.

    validation : pandas.DataFrame
        Conjunto de validación con las columnas 'ds', 'y' y los
        regresores utilizados.

    regresores_futuros : list
        Lista con los nombres de los regresores futuros.

    regresores_retardados : list
        Lista con los nombres de los regresores retardados.

    mejores_parametros : dict
        Diccionario con los mejores hiperparámetros del modelo.

    Retorna
    -------
    modelo_neuralprophet : NeuralProphet
        Modelo NeuralProphet entrenado.

    historial_entrenamiento : pandas.DataFrame
        Historial generado durante el entrenamiento.

    prediccion_validacion : pandas.DataFrame
        Predicciones correspondientes al periodo de validación.

    comparacion_validacion : pandas.DataFrame
        Valores reales y predichos del conjunto de validación.

    metricas_validacion : dict
        Métricas calculadas mediante EVALUAR_METRICAS.

    train_neuralprophet_final : pandas.DataFrame
        Copia preparada del conjunto de entrenamiento.

    validation_neuralprophet_final : pandas.DataFrame
        Copia preparada del conjunto de validación.
    """

    # ==========================================================================
    # COPIA DE LOS CONJUNTOS
    # ==========================================================================

    train_neuralprophet_final = train.copy()
    validation_neuralprophet_final = validation.copy()

    # ==========================================================================
    # PREPARACIÓN DE LOS DATOS
    # ==========================================================================

    # Conversión de las fechas
    train_neuralprophet_final["ds"] = pd.to_datetime(
        train_neuralprophet_final["ds"]
    )

    validation_neuralprophet_final["ds"] = pd.to_datetime(
        validation_neuralprophet_final["ds"]
    )

    # Ordenación cronológica
    train_neuralprophet_final = train_neuralprophet_final.sort_values(
        by="ds"
    ).reset_index(drop=True)

    validation_neuralprophet_final = (
        validation_neuralprophet_final.sort_values(
            by="ds"
        ).reset_index(drop=True)
    )

    # ==========================================================================
    # FIJACIÓN DE LAS SEMILLAS
    # ==========================================================================

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ==========================================================================
    # ESTACIONALIDADES
    # ==========================================================================

    # Si una estacionalidad no aparece entre los mejores parámetros,
    # se desactiva mediante False.
    yearly_seasonality = mejores_parametros.get(
        "yearly_seasonality",
        False
    )

    weekly_seasonality = mejores_parametros.get(
        "weekly_seasonality",
        False
    )

    daily_seasonality = mejores_parametros.get(
        "daily_seasonality",
        False
    )

    # ==========================================================================
    # CREACIÓN DEL MODELO
    # ==========================================================================

    modelo_neuralprophet = NeuralProphet(

        # Tendencia
        growth=mejores_parametros.get(
            "growth",
            "linear"
        ),
        changepoints_range=mejores_parametros[
            "changepoints_range"
        ],
        trend_reg=mejores_parametros[
            "trend_reg"
        ],

        # Estacionalidad
        seasonality_mode=mejores_parametros[
            "seasonality_mode"
        ],
        seasonality_reg=mejores_parametros[
            "seasonality_reg"
        ],
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,

        # Autorregresión
        n_lags=mejores_parametros[
            "n_lags"
        ],
        n_forecasts=1,
        ar_layers=mejores_parametros[
            "ar_layers"
        ],
        lagged_reg_layers=mejores_parametros[
            "lagged_reg_layers"
        ],

        # Entrenamiento
        epochs=mejores_parametros[
            "epochs"
        ],
        batch_size=mejores_parametros[
            "batch_size"
        ]
    )

    # ==========================================================================
    # INCORPORACIÓN DE LOS REGRESORES RETARDADOS
    # ==========================================================================

    for variable in regresores_retardados:

        modelo_neuralprophet.add_lagged_regressor(
            names=variable,
            n_lags=mejores_parametros["n_lags"],
            normalize="standardize"
        )

    # ==========================================================================
    # INCORPORACIÓN DE LOS REGRESORES FUTUROS
    # ==========================================================================

    for variable in regresores_futuros:

        modelo_neuralprophet.add_future_regressor(
            name=variable,
            normalize="standardize",
            mode=mejores_parametros["seasonality_mode"]
        )

    # ==========================================================================
    # INCORPORACIÓN DE LOS FESTIVOS NACIONALES DE ESPAÑA
    # ==========================================================================

    if mejores_parametros["usar_festivos"]:

        modelo_neuralprophet.add_country_holidays(
            country_name="ES"
        )

    # ==========================================================================
    # ENTRENAMIENTO DEL MODELO
    # ==========================================================================

    historial_entrenamiento = modelo_neuralprophet.fit(
        train_neuralprophet_final,
        freq="h",
        progress=None
    )

    # ==========================================================================
    # PREPARACIÓN DE LOS DATOS DE VALIDACIÓN
    # ==========================================================================

    # NeuralProphet necesita las últimas n_lags observaciones del conjunto
    # de entrenamiento para construir los primeros retardos de validación.
    contexto_entrenamiento = train_neuralprophet_final.tail(
        mejores_parametros["n_lags"]
    ).copy()

    datos_prediccion_validacion = pd.concat(
        [
            contexto_entrenamiento,
            validation_neuralprophet_final
        ],
        ignore_index=True
    )

    datos_prediccion_validacion = (
        datos_prediccion_validacion.sort_values(
            by="ds"
        ).reset_index(drop=True)
    )

    # ==========================================================================
    # PREDICCIÓN SOBRE EL CONJUNTO DE VALIDACIÓN
    # ==========================================================================

    prediccion_validacion_completa = modelo_neuralprophet.predict(
        datos_prediccion_validacion
    )

    # ==========================================================================
    # SELECCIÓN DE LAS PREDICCIONES DEL PERIODO DE VALIDACIÓN
    # ==========================================================================

    prediccion_validacion = prediccion_validacion_completa[
        prediccion_validacion_completa["ds"].isin(
            validation_neuralprophet_final["ds"]
        )
    ][
        [
            "ds",
            "yhat1"
        ]
    ].copy()

    # ==========================================================================
    # UNIÓN DE LOS VALORES REALES Y LAS PREDICCIONES
    # ==========================================================================

    comparacion_validacion = validation_neuralprophet_final[
        [
            "ds",
            "y"
        ]
    ].merge(
        prediccion_validacion,
        on="ds",
        how="inner"
    )

    comparacion_validacion = comparacion_validacion.dropna(
        subset=[
            "y",
            "yhat1"
        ]
    ).reset_index(drop=True)

    # ==========================================================================
    # EVALUACIÓN DEL MODELO
    # ==========================================================================

    num_parametros = (
        len(regresores_retardados)
        + len(regresores_futuros)
    )

    metricas_validacion = EVALUAR_METRICAS(
        y_real=comparacion_validacion["y"],
        y_predicho=comparacion_validacion["yhat1"],
        num_parametros=num_parametros
    )

    return (
        modelo_neuralprophet,
        historial_entrenamiento,
        prediccion_validacion,
        comparacion_validacion,
        metricas_validacion,
        train_neuralprophet_final,
        validation_neuralprophet_final
    )


#############################################################################################
###################################### FUNCIONES LSTM #######################################
#############################################################################################

def ESCALADOR(
    train,
    validation,
    test,
    variables_exogenas
):
    """
    Escala la variable objetivo y las variables exógenas utilizando
    MinMaxScaler con rango [0, 1].

    Los escaladores se ajustan únicamente sobre el conjunto de entrenamiento
    y posteriormente se aplican a entrenamiento, validación y test.

    Parámetros
    ----------
    train : pandas.DataFrame
        Conjunto de entrenamiento.

    validation : pandas.DataFrame
        Conjunto de validación.

    test : pandas.DataFrame
        Conjunto de test.

    variables_exogenas : list
        Lista con los nombres de las variables exógenas que se escalarán.

    Retorna
    -------
    train_lstm : pandas.DataFrame
        Conjunto de entrenamiento escalado.

    validation_lstm : pandas.DataFrame
        Conjunto de validación escalado.

    test_lstm : pandas.DataFrame
        Conjunto de test escalado.

    scaler_y : MinMaxScaler
        Escalador ajustado sobre la variable objetivo.

    scaler_X : MinMaxScaler
        Escalador ajustado sobre las variables exógenas.
    """

    # ==========================================================================
    # COPIAS DE LOS CONJUNTOS DE DATOS
    # ==========================================================================

    train_lstm = train.copy()
    validation_lstm = validation.copy()
    test_lstm = test.copy()

    # ==========================================================================
    # CONVERSIÓN Y ORDENACIÓN DE LAS FECHAS
    # ==========================================================================

    for df_lstm in [
        train_lstm,
        validation_lstm,
        test_lstm
    ]:

        df_lstm["ds"] = pd.to_datetime(
            df_lstm["ds"]
        )

        df_lstm.sort_values(
            "ds",
            inplace=True
        )

        df_lstm.reset_index(
            drop=True,
            inplace=True
        )

    # ==========================================================================
    # ESCALADORES
    # ==========================================================================

    scaler_y = MinMaxScaler(
        feature_range=(0, 1)
    )

    scaler_X = MinMaxScaler(
        feature_range=(0, 1)
    )

    # ==========================================================================
    # AJUSTE DE LOS ESCALADORES ÚNICAMENTE SOBRE ENTRENAMIENTO
    # ==========================================================================

    scaler_y.fit(
        train_lstm[
            [
                "y"
            ]
        ]
    )

    scaler_X.fit(
        train_lstm[
            variables_exogenas
        ]
    )

    # ==========================================================================
    # TRANSFORMACIÓN DE LOS TRES CONJUNTOS
    # ==========================================================================

    for df_lstm in [
        train_lstm,
        validation_lstm,
        test_lstm
    ]:

        df_lstm["y"] = scaler_y.transform(
            df_lstm[
                [
                    "y"
                ]
            ]
        ).ravel()

        df_lstm[
            variables_exogenas
        ] = scaler_X.transform(
            df_lstm[
                variables_exogenas
            ]
        )

    return (
        train_lstm,
        validation_lstm,
        test_lstm,
        scaler_y,
        scaler_X
    )



def CREAR_DATASET_LSTM(df, input_size, variables_exogenas):
    """
    Crea secuencias temporales para entrenar una red LSTM.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con la variable objetivo 'y' y las variables exógenas.

    input_size : int
        Número de instantes temporales anteriores utilizados
        para predecir el siguiente valor.

    variables_exogenas : list
        Lista de variables exógenas utilizadas como predictores.

    Retorna
    -------
    X : np.ndarray
        Matriz tridimensional con forma:
        (n_muestras, input_size, n_variables).

    y : np.ndarray
        Valores de la variable objetivo correspondientes
        al instante inmediatamente posterior a cada secuencia.
    """

    # Variables utilizadas como entrada
    columnas_entrada = ["y"] + variables_exogenas

    valores_X = df[columnas_entrada].values.astype(np.float32)
    valores_y = df["y"].values.astype(np.float32)

    X = []
    y = []

    for i in range(input_size, len(df)):

        # Ventana con los input_size instantes anteriores
        X.append(
            valores_X[i - input_size:i]
        )

        # Valor de y en el instante siguiente
        y.append(
            valores_y[i]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )



def CREAR_DATASET_LSTM_VALIDATION(
    train,
    validation,
    input_size,
    variables_exogenas
):
    """
    Crea las secuencias de validación incorporando las últimas
    observaciones del conjunto de entrenamiento como contexto.
    """

    # Últimas observaciones del entrenamiento necesarias
    contexto = train.tail(input_size)

    # Las concatenamos con validación
    df_completo = pd.concat(
        [contexto, validation],
        axis=0,
        ignore_index=True
    )

    columnas_entrada = ["y"] + variables_exogenas

    valores_X = df_completo[
        columnas_entrada
    ].values.astype(np.float32)

    valores_y = df_completo[
        "y"
    ].values.astype(np.float32)

    X = []
    y = []

    # Comenzamos justo donde empieza validación
    for i in range(input_size, len(df_completo)):

        X.append(
            valores_X[i - input_size:i]
        )

        y.append(
            valores_y[i]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )



def CONSTRUIR_MODELO_LSTM(
    input_shape,
    units,
    num_layers,
    dropout,
    optimizer
):
    """
    Construye y compila una red LSTM.

    Parámetros
    ----------
    input_shape : tuple
        Dimensiones de cada secuencia:
        (input_size, número de variables).

    units : int
        Número de unidades de cada capa LSTM.

    num_layers : int
        Número de capas LSTM.

    dropout : float
        Proporción de dropout aplicada en las capas LSTM.

    optimizer : str
        Optimizador empleado durante el entrenamiento.

    Retorna
    -------
    model : tf.keras.Model
        Modelo LSTM compilado.
    """

    model = Sequential()

    # --------------------------------------------------------------------------
    # Primera capa LSTM
    # --------------------------------------------------------------------------

    if num_layers == 1:

        model.add(
            LSTM(
                units=units,
                dropout=dropout,
                return_sequences=False,
                input_shape=input_shape
            )
        )

    else:

        model.add(
            LSTM(
                units=units,
                dropout=dropout,
                return_sequences=True,
                input_shape=input_shape
            )
        )

        # ----------------------------------------------------------------------
        # Capas LSTM adicionales
        # ----------------------------------------------------------------------

        for i in range(num_layers - 1):

            ultima_capa = (i == num_layers - 2)

            model.add(
                LSTM(
                    units=units,
                    dropout=dropout,
                    return_sequences=not ultima_capa
                )
            )

    # --------------------------------------------------------------------------
    # Capa de salida
    # --------------------------------------------------------------------------

    model.add(
        Dense(1)
    )

    # --------------------------------------------------------------------------
    # Compilación
    # --------------------------------------------------------------------------

    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error"
    )

    return model



def BUSQUEDA_CONFIGURACIONES_LSMT(
    train,
    validation,
    variables_exogenas,
    param_grid,
    scaler_y,
    seed=42
):
    """
    Realiza una búsqueda de hiperparámetros para un modelo LSTM
    utilizando ParameterGrid.

    La selección se realiza según el MSE obtenido sobre
    el conjunto de validación.

    Parámetros
    ----------
    train : pd.DataFrame
        Conjunto de entrenamiento escalado.

    validation : pd.DataFrame
        Conjunto de validación escalado.

    variables_exogenas : list
        Lista de variables exógenas utilizadas.

    param_grid : dict
        Diccionario con los hiperparámetros a evaluar.

    scaler_y : MinMaxScaler
        Escalador empleado para la variable objetivo.

    seed : int
        Semilla para garantizar reproducibilidad.

    Retorna
    -------
    resultados : pd.DataFrame
        Resultados obtenidos para todas las configuraciones.

    mejor_configuracion : dict
        Configuración que obtiene el menor MSE de validación.
    """

    # ==========================================================================
    # Generación de configuraciones
    # ==========================================================================

    configuraciones = list(ParameterGrid(param_grid))

    resultados = []

    mejor_mse = np.inf
    mejor_configuracion = None

    # ==========================================================================
    # Evaluación de cada configuración
    # ==========================================================================

    for params in configuraciones:

        # ----------------------------------------------------------------------
        # Semillas
        # ----------------------------------------------------------------------

        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

        # Limpiar modelos anteriores de Keras
        tf.keras.backend.clear_session()

        # ----------------------------------------------------------------------
        # Creación de secuencias
        # ----------------------------------------------------------------------

        X_train, y_train = CREAR_DATASET_LSTM(
            train,
            input_size=params["input_size"],
            variables_exogenas=variables_exogenas
        )

        X_validation, y_validation = CREAR_DATASET_LSTM_VALIDATION(
            train=train,
            validation=validation,
            input_size=params["input_size"],
            variables_exogenas=variables_exogenas
        )

        # ----------------------------------------------------------------------
        # Construcción del modelo
        # ----------------------------------------------------------------------

        model = CONSTRUIR_MODELO_LSTM(
            input_shape=(
                X_train.shape[1],
                X_train.shape[2]
            ),
            units=params["units"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            optimizer=params["optimizer"]
        )

        # ----------------------------------------------------------------------
        # Entrenamiento
        # ----------------------------------------------------------------------

        model.fit(
            X_train,
            y_train,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=0,
            shuffle=False
        )

        # ----------------------------------------------------------------------
        # Predicción sobre validación
        # ----------------------------------------------------------------------

        pred_validation = model.predict(
            X_validation,
            verbose=0
        )

        # ----------------------------------------------------------------------
        # Desescalado
        # ----------------------------------------------------------------------

        pred_validation_inv = scaler_y.inverse_transform(
            pred_validation
        ).ravel()

        y_validation_inv = scaler_y.inverse_transform(
            y_validation.reshape(-1, 1)
        ).ravel()

        # ----------------------------------------------------------------------
        # MSE de validación
        # ----------------------------------------------------------------------

        mse_validation = mean_squared_error(
            y_validation_inv,
            pred_validation_inv
        )

        # ----------------------------------------------------------------------
        # Guardar resultados
        # ----------------------------------------------------------------------

        resultado = params.copy()
        resultado["MSE_validacion"] = mse_validation

        resultados.append(resultado)

        # ----------------------------------------------------------------------
        # Actualización de la mejor configuración
        # ----------------------------------------------------------------------

        if mse_validation < mejor_mse:

            mejor_mse = mse_validation
            mejor_configuracion = params.copy()

        # ----------------------------------------------------------------------
        # Liberación de memoria
        # ----------------------------------------------------------------------

        del model
        del X_train
        del y_train
        del X_validation
        del y_validation

        tf.keras.backend.clear_session()

    # ==========================================================================
    # DataFrame final de resultados
    # ==========================================================================

    resultados = pd.DataFrame(resultados)

    resultados = resultados.sort_values(
        by="MSE_validacion",
        ascending=True
    ).reset_index(drop=True)

    # ==========================================================================
    # Impresión de la mejor configuración
    # ==========================================================================

    print("Mejores parámetros:\n")

    for parametro, valor in mejor_configuracion.items():
        print(f"{parametro}: {valor}")

    print(
        f"\nMSE de validación: "
        f"{mejor_mse:.6f}"
    )

    return resultados, mejor_configuracion


def ENTRENAR_EVALUAR_LSTM(
    train,
    validation,
    test,
    variables_exogenas,
    mejores_parametros,
    scaler_y,
    seed=42
):
    """
    Entrena y evalúa el modelo LSTM final utilizando los mejores
    hiperparámetros obtenidos durante la búsqueda.

    El modelo final se entrena utilizando conjuntamente los conjuntos
    de entrenamiento y validación, mientras que el conjunto de prueba
    se reserva exclusivamente para la evaluación final.

    Parámetros
    ----------
    train : pd.DataFrame
        Conjunto de entrenamiento escalado.

    validation : pd.DataFrame
        Conjunto de validación escalado.

    test : pd.DataFrame
        Conjunto de prueba escalado.

    variables_exogenas : list
        Lista de variables exógenas utilizadas como predictores.

    mejores_parametros : dict
        Diccionario con los mejores hiperparámetros encontrados.

    scaler_y : MinMaxScaler
        Escalador utilizado para la variable objetivo.

    seed : int, default=42
        Semilla utilizada para garantizar reproducibilidad.

    Retorna
    -------
    modelo : tf.keras.Model
        Modelo LSTM final entrenado.

    metricas : dict
        Diccionario con las métricas obtenidas sobre test.

    y_test_real : np.ndarray
        Valores reales de la variable objetivo sin escalar.

    y_test_predicho : np.ndarray
        Predicciones del modelo sin escalar.

    fechas_test : pd.Series
        Fechas correspondientes a las predicciones realizadas.
    """

    # ==========================================================================
    # Semillas para reproducibilidad
    # ==========================================================================

    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    tf.keras.backend.clear_session()


    # ==========================================================================
    # Unión de entrenamiento y validación
    # ==========================================================================

    train_final = pd.concat(
        [train, validation],
        axis=0,
        ignore_index=True
    )

    train_final = train_final.sort_values(
        by="ds"
    ).reset_index(drop=True)


    # ==========================================================================
    # Extracción de los mejores hiperparámetros
    # ==========================================================================

    units = mejores_parametros["units"]
    num_layers = mejores_parametros["num_layers"]
    dropout = mejores_parametros["dropout"]
    optimizer = mejores_parametros["optimizer"]
    epochs = mejores_parametros["epochs"]
    batch_size = mejores_parametros["batch_size"]
    input_size = mejores_parametros["input_size"]


    # ==========================================================================
    # Creación de las secuencias de entrenamiento
    # ==========================================================================

    X_train, y_train = CREAR_DATASET_LSTM(
        df=train_final,
        input_size=input_size,
        variables_exogenas=variables_exogenas
    )


    # ==========================================================================
    # Creación de las secuencias de prueba
    # ==========================================================================

    X_test, y_test = CREAR_DATASET_LSTM_VALIDATION(
        train=train_final,
        validation=test,
        input_size=input_size,
        variables_exogenas=variables_exogenas
    )


    # ==========================================================================
    # Construcción del modelo final
    # ==========================================================================

    modelo = CONSTRUIR_MODELO_LSTM(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        ),
        units=units,
        num_layers=num_layers,
        dropout=dropout,
        optimizer=optimizer
    )


    # ==========================================================================
    # Entrenamiento del modelo
    # ==========================================================================

    modelo.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        shuffle=False
    )


    # ==========================================================================
    # Predicción sobre el conjunto de prueba
    # ==========================================================================

    y_test_predicho = modelo.predict(
        X_test,
        verbose=0
    )


    # ==========================================================================
    # Desescalado
    # ==========================================================================

    y_test_real = scaler_y.inverse_transform(
        y_test.reshape(-1, 1)
    ).ravel()

    y_test_predicho = scaler_y.inverse_transform(
        y_test_predicho
    ).ravel()


    # ==========================================================================
    # Evaluación del modelo
    # ==========================================================================

    metricas = EVALUAR_METRICAS(
        y_real=y_test_real,
        y_predicho=y_test_predicho,
        num_parametros=len(variables_exogenas)
    )


    # ==========================================================================
    # Fechas correspondientes a las predicciones
    # ==========================================================================

    fechas_test = test["ds"].reset_index(drop=True)


    # ==========================================================================
    # Retorno de resultados
    # ==========================================================================

    return (
        modelo,
        metricas,
        y_test_real,
        y_test_predicho,
        fechas_test
    )



#############################################################################################
##################################### FUNCIONES SARIMAX #####################################
#############################################################################################


def BUSQUEDA_CONFIGURACIONES_SARIMAX(
    train,
    validation,
    variables_exogenas,
    periodo_estacional=24
):
    """
    Busca automáticamente tres configuraciones SARIMAX/ARIMAX:
    - La que minimiza el AIC.
    - La que minimiza el AICc.
    - La que minimiza el BIC.

    Si periodo_estacional es None, el modelo se considera
    no estacional y se ajusta como ARIMAX.

    Las búsquedas se realizan exclusivamente sobre el conjunto
    de entrenamiento mediante auto_arima.

    Posteriormente, las tres configuraciones seleccionadas se
    ajustan sobre entrenamiento y se evalúan sobre validación.
    La configuración final será aquella que presente el menor
    MSE sobre el conjunto de validación.

    Parámetros
    ----------
    train : pd.DataFrame
        Conjunto de entrenamiento.

    validation : pd.DataFrame
        Conjunto de validación.

    variables_exogenas : list
        Variables exógenas empleadas por el modelo.

    periodo_estacional : int or None, default=24
        Periodicidad de la componente estacional.
        Si es None, se ajusta un modelo no estacional.

    Retorna
    -------
    resultados : pd.DataFrame
        Resultados de los modelos seleccionados mediante
        AIC, AICc y BIC.

    mejor_configuracion : dict
        Configuración con menor MSE sobre validación.

    mejor_modelo :
        Modelo SARIMAX/ARIMAX ajustado correspondiente a la
        mejor configuración.
    """

    # ==========================================================================
    # Variable objetivo
    # ==========================================================================

    y_train = train["y"]
    y_validation = validation["y"]

    # ==========================================================================
    # Variables exógenas
    # ==========================================================================

    X_train = train[variables_exogenas]
    X_validation = validation[variables_exogenas]

    # ==========================================================================
    # Configuración de la estacionalidad
    # ==========================================================================

    if periodo_estacional is None:

        seasonal = False
        m = 1

    else:

        seasonal = True
        m = periodo_estacional

    # ==========================================================================
    # Criterios de información
    # ==========================================================================

    criterios = ["aic", "aicc", "bic"]

    configuraciones = []

    # ==========================================================================
    # Búsqueda automática de configuraciones
    # ==========================================================================

    for criterio in criterios:

        modelo_auto = auto_arima(
            y=y_train,
            X=X_train,

            # ==============================================================
            # Componente estacional
            # ==============================================================

            seasonal=seasonal,
            m=m,

            # ==============================================================
            # Diferenciación
            # ==============================================================

            d=None,
            D=None if seasonal else 0,

            # ==============================================================
            # Búsqueda de órdenes
            # ==============================================================

            start_p=0,
            start_q=0,
            max_p=8,
            max_q=8,

            start_P=0,
            start_Q=0,
            max_P=8 if seasonal else 0,
            max_Q=8 if seasonal else 0,

            # ==============================================================
            # Criterio de selección
            # ==============================================================

            information_criterion=criterio,

            # ==============================================================
            # Configuración de la búsqueda
            # ==============================================================

            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False
        )

        # ======================================================================
        # Órdenes seleccionados
        # ======================================================================

        order = modelo_auto.order

        if seasonal:
            seasonal_order = modelo_auto.seasonal_order
        else:
            seasonal_order = (0, 0, 0, 0)

        configuraciones.append({
            "criterio": criterio.upper(),
            "order": order,
            "seasonal_order": seasonal_order
        })

    # ==========================================================================
    # Evaluación de las configuraciones sobre validación
    # ==========================================================================

    resultados = []

    mejor_mse = np.inf
    mejor_configuracion = None
    mejor_modelo = None

    for config in configuraciones:

        criterio = config["criterio"]
        order = config["order"]
        seasonal_order = config["seasonal_order"]

        try:

            # ==============================================================
            # Definición del modelo SARIMAX / ARIMAX
            # ==============================================================

            modelo = SARIMAX(
                endog=y_train,
                exog=X_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            # ==============================================================
            # Ajuste exclusivamente sobre entrenamiento
            # ==============================================================

            modelo_ajustado = modelo.fit(
                disp=False
            )

            # ==============================================================
            # Predicción sobre validación
            # ==============================================================

            prediccion = modelo_ajustado.get_forecast(
                steps=len(validation),
                exog=X_validation
            )

            y_pred = np.asarray(
                prediccion.predicted_mean
            )

            # ==============================================================
            # MSE de validación
            # ==============================================================

            mse_validacion = metrics.mean_squared_error(
                y_validation,
                y_pred
            )

            # ==============================================================
            # AICc
            # ==============================================================

            n = modelo_ajustado.nobs
            k = len(modelo_ajustado.params)

            if n - k - 1 > 0:

                aicc = (
                    modelo_ajustado.aic
                    +
                    (2 * k * (k + 1)) / (n - k - 1)
                )

            else:

                aicc = np.nan

            # ==============================================================
            # Almacenar resultados
            # ==============================================================

            resultados.append({
                "criterio_seleccion": criterio,

                "p": order[0],
                "d": order[1],
                "q": order[2],

                "P": seasonal_order[0],
                "D": seasonal_order[1],
                "Q": seasonal_order[2],
                "s": periodo_estacional,

                "AIC": modelo_ajustado.aic,
                "AICc": aicc,
                "BIC": modelo_ajustado.bic,

                "MSE_validacion": mse_validacion
            })

            # ==============================================================
            # Selección mediante MSE de validación
            # ==============================================================

            if mse_validacion < mejor_mse:

                mejor_mse = mse_validacion

                mejor_configuracion = {
                    "criterio_seleccion": criterio,

                    "p": order[0],
                    "d": order[1],
                    "q": order[2],

                    "P": seasonal_order[0],
                    "D": seasonal_order[1],
                    "Q": seasonal_order[2],

                    "s": periodo_estacional
                }

                mejor_modelo = modelo_ajustado

        except Exception:
            pass

    # ==========================================================================
    # DataFrame de resultados
    # ==========================================================================

    resultados = pd.DataFrame(resultados)

    resultados.sort_values(
        by="MSE_validacion",
        ascending=True,
        inplace=True
    )

    resultados.reset_index(
        drop=True,
        inplace=True
    )

    # ==========================================================================
    # Resultado final
    # ==========================================================================

    print("Mejores parámetros:")
    print()

    print(f"s = {mejor_configuracion['s']}")
    print(f"p = {mejor_configuracion['p']}")
    print(f"d = {mejor_configuracion['d']}")
    print(f"q = {mejor_configuracion['q']}")
    print(f"P = {mejor_configuracion['P']}")
    print(f"D = {mejor_configuracion['D']}")
    print(f"Q = {mejor_configuracion['Q']}")

    print()

    print(
        f"Criterio de Selección: "
        f"{mejor_configuracion['criterio_seleccion']}"
    )

    print(
        f"MSE de validación: {mejor_mse:.6f}"
    )

    return (
        resultados,
        mejor_configuracion,
        mejor_modelo
    )



def ENTRENAR_EVALUAR_SARIMAX(
    train,
    validation,
    test,
    variables_exogenas,
    mejores_parametros
):
    """
    Entrena y evalúa un modelo SARIMAX/ARIMAX utilizando los
    mejores hiperparámetros obtenidos previamente.

    Si s es None, se ajusta un modelo no estacional.

    El modelo se ajusta utilizando conjuntamente los conjuntos
    de entrenamiento y validación y se evalúa exclusivamente
    sobre el conjunto de prueba.

    Parámetros
    ----------
    train : pd.DataFrame
        Conjunto de entrenamiento.

    validation : pd.DataFrame
        Conjunto de validación.

    test : pd.DataFrame
        Conjunto de prueba.

    variables_exogenas : list
        Variables exógenas empleadas por el modelo.

    mejores_parametros : dict
        Diccionario con los mejores hiperparámetros:
        p, d, q, P, D, Q y s.

        Si s es None, se considera que no existe
        componente estacional.

    Retorna
    -------
    modelo_ajustado :
        Modelo SARIMAX/ARIMAX final ajustado.

    resultados : dict
        Diccionario con las métricas de evaluación.

    predicciones : np.ndarray
        Predicciones realizadas sobre el conjunto de prueba.
    """

    # ==========================================================================
    # Unión de entrenamiento y validación
    # ==========================================================================

    train_validation = pd.concat(
        [train, validation],
        ignore_index=True
    )

    # ==========================================================================
    # Conversión y ordenación temporal
    # ==========================================================================

    train_validation["ds"] = pd.to_datetime(
        train_validation["ds"]
    )

    test_sarimax = test.copy()

    test_sarimax["ds"] = pd.to_datetime(
        test_sarimax["ds"]
    )

    train_validation.sort_values(
        by="ds",
        inplace=True
    )

    test_sarimax.sort_values(
        by="ds",
        inplace=True
    )

    train_validation.reset_index(
        drop=True,
        inplace=True
    )

    test_sarimax.reset_index(
        drop=True,
        inplace=True
    )

    # ==========================================================================
    # Variable objetivo
    # ==========================================================================

    y_train_validation = train_validation["y"]
    y_test = test_sarimax["y"]

    # ==========================================================================
    # Variables exógenas
    # ==========================================================================

    X_train_validation = train_validation[
        variables_exogenas
    ]

    X_test = test_sarimax[
        variables_exogenas
    ]

    # ==========================================================================
    # Hiperparámetros
    # ==========================================================================

    p = mejores_parametros["p"]
    d = mejores_parametros["d"]
    q = mejores_parametros["q"]

    P = mejores_parametros["P"]
    D = mejores_parametros["D"]
    Q = mejores_parametros["Q"]

    s = mejores_parametros["s"]

    # ==========================================================================
    # Componente estacional
    # ==========================================================================

    if s is None:

        seasonal_order = (0, 0, 0, 0)

    else:

        seasonal_order = (
            P,
            D,
            Q,
            s
        )

    # ==========================================================================
    # Definición del modelo SARIMAX / ARIMAX
    # ==========================================================================

    modelo = SARIMAX(
        endog=y_train_validation,
        exog=X_train_validation,
        order=(p, d, q),
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # ==========================================================================
    # Entrenamiento del modelo
    # ==========================================================================

    modelo_ajustado = modelo.fit(
        disp=False
    )

    # ==========================================================================
    # Predicción sobre el conjunto de prueba
    # ==========================================================================

    prediccion = modelo_ajustado.get_forecast(
        steps=len(test_sarimax),
        exog=X_test
    )

    predicciones = np.asarray(
        prediccion.predicted_mean
    )

    # ==========================================================================
    # Número de parámetros del modelo
    # ==========================================================================

    num_parametros = len(
        modelo_ajustado.params
    )

    # ==========================================================================
    # Evaluación del modelo
    # ==========================================================================

    resultados = EVALUAR_METRICAS(
        y_real=y_test,
        y_predicho=predicciones,
        num_parametros=num_parametros
    )

    return (
        modelo_ajustado,
        resultados,
        predicciones
    )