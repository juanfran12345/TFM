import numpy as np
import pandas as pd
import itertools
from sklearn import metrics
from pathlib import Path

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error



import warnings
warnings.filterwarnings("ignore")

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

        print("MEJOR COMBINACIÓN DE HIPERPARÁMETROS:\n")
        print("Estacionalidades utilizadas:\n")

        if estacionalidades_normalizadas:
            for estacionalidad in estacionalidades_normalizadas:
                print(f"- {estacionalidad}")
        else:
            print("- Ninguna")

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