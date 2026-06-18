import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

# =====================================================
# Preparacion de Pagina 
# =====================================================

st.set_page_config(
    page_title="Trabajo 2 - Telco Customer Churn",
    layout="wide"
)

# =====================================================
# Aqui definimos la Clase POO, q sera usada por la applicacion
# =====================================================

class DataAnalyzer:

    def __init__(self, df):
        self.df = df

    def get_numeric_columns(self):
        return self.df.select_dtypes(include=np.number).columns.tolist()

    def get_categorical_columns(self):
        return self.df.select_dtypes(exclude=np.number).columns.tolist()

    def null_values(self):
        return self.df.isnull().sum()

    def descriptive_stats(self):
        return self.df.describe()

    def mode_values(self):
        return self.df.mode().iloc[0]

# =====================================================
# Contenido y Meunu HOME
# =====================================================

st.image("banner.png" )
st.sidebar.image("menu.jpg")

menu = st.sidebar.radio(
    "Menú Principal",
    ["1 Home", "2 Carga Dataset", "3 Analisis EDA", "4 Mis Conclusiones"]
)

# =====================================================
# HOME
# =====================================================

if menu == "1 Home":

    st.title("Analisis Exploratorio de Datos - Telco Customer Churn")
    st.markdown("Elaborado: Samuel Ladera Q.") 
    st.markdown("""
    ## Objetivo

    Analizar los datos históricos de clientes de una empresa de telecomunicaciones
    para identificar patrones asociados a la fuga de clientes (Churn).

    Este proyecto aplica conceptos de:

    - Python
    - Programación Orientada a Objetos (POO)
    - Pandas
    - NumPy
    - Estadística Descriptiva
    - Matplotlib
    - Seaborn
    - Streamlit
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **Autor:** SAMUEL LADERA QUINTO

        **Curso:** Especialización Python Potenciado con IA

        **Año:** 2026
        """)

    with col2:
        st.success("""
        Dataset: TelcoCustomerChurn.csv

        Contiene información sobre:
        - Clientes
        - Contratos
        - Servicios
        - Facturación
        - Churn
        """)

    st.subheader("Tecnologías utilizadas para este Trabajo 2")

    st.write("""
    Python, Pandas, NumPy, Streamlit,
    Matplotlib y Seaborn.
    """)

# =====================================================
# CARGA DATASET
# =====================================================

elif menu == "2 Carga Dataset":

    st.title("Carga del Dataset")

    uploaded_file = st.file_uploader(
        "Seleccione Archivo csv",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)
        #df = pd.read_csv(uploaded_file)
        
        # Convertir espacios vacíos en NaN
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        st.success("Archivo cargado correctamente.")

        st.subheader("Vista previa")

        st.dataframe(df.head())

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Filas", df.shape[0])

        with c2:
            st.metric("Columnas", df.shape[1])

        st.session_state["df"] = df

# =====================================================
# EDA
# =====================================================

elif menu == "3 Analisis EDA":

    st.title("Analisis de Data")

    if "df" not in st.session_state:
        st.warning("Primero debe cargar el dataset.")
        st.stop()

    df = st.session_state["df"]

    analyzer = DataAnalyzer(df)

    tabs = st.tabs([
        "1️ Info",
        "2️ Variables",
        "3 Estadisticas",
        "4 Nulos",
        "5 Numericas",
        "6 Categoricas",
        "7 Num vs Cat",
        "8 Cat vs Cat",
        "9 Dinamico",
        "10 Hallazgos"
    ])

    # =====================================================
    # informacion general
    # =====================================================

    with tabs[0]:

        st.header("Información General")

        buffer = StringIO()

        df.info(buf=buffer)

        st.text(buffer.getvalue())

        st.subheader("Tipos de Datos")
        st.dataframe(df.dtypes.astype(str))

        st.subheader("Valores Nulos")
        st.dataframe(df.isnull().sum())

    # =====================================================
    # clasificacon de variables
    # =====================================================

    with tabs[1]:

        st.header("Clasificación de Variables")

        numeric_cols = analyzer.get_numeric_columns()
        categorical_cols = analyzer.get_categorical_columns()

        c1, c2 = st.columns(2)

        with c1:
            st.success(f"Variables Numéricas: {len(numeric_cols)}")
            st.write(numeric_cols)

        with c2:
            st.info(f"Variables Categóricas: {len(categorical_cols)}")
            st.write(categorical_cols)

    # =====================================================
    # Estadisticas
    # =====================================================

    with tabs[2]:

        st.header("Estadisticas Descriptivas")

        st.dataframe(analyzer.descriptive_stats())

        st.subheader("Moda")

        st.dataframe(analyzer.mode_values())

    # =====================================================
    # Valores Faltantes
    # =====================================================

    with tabs[3]:
        missing = pd.DataFrame({
            "Columna": df.columns,
            "Nulos": df.isnull().sum().values
           # "% Nulos": round((df.isnull().sum().values / len(df)) * 100,2)
        })
        
        st.dataframe(missing)
                

    # =====================================================
    # Variables Numericas
    # =====================================================

    with tabs[4]:

        st.header("Distribucion Variables Numericas")

        num_cols = analyzer.get_numeric_columns()

        selected_num = st.selectbox(
            "Seleccione Variable",
            num_cols
        )

        fig, ax = plt.subplots(figsize=(8,4))

        sns.histplot(
            df[selected_num],
            kde=True,
            ax=ax
        )

        st.pyplot(fig)

    # =====================================================
    # Variables Categoricas
    # =====================================================

    with tabs[5]:

        st.header("Variables Categoricas")

        cat_cols = analyzer.get_categorical_columns()

        selected_cat = st.selectbox(
            "Variable Categórica",
            cat_cols
        )

        counts = df[selected_cat].value_counts()

        st.dataframe(counts)

        fig, ax = plt.subplots(figsize=(8,4))

        sns.countplot(
            data=df,
            x=selected_cat,
            ax=ax
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)

    # =====================================================
    # Numerico y Churn
    # =====================================================

    with tabs[6]:

        st.header("Analisis Numerico vs Churn")

        variable = st.selectbox(
            "Variable Numérica",
            analyzer.get_numeric_columns()
        )

        fig, ax = plt.subplots(figsize=(8,4))

        sns.boxplot(
            data=df,
            x="Churn",
            y=variable,
            ax=ax
        )

        st.pyplot(fig)

    # =====================================================
    # Categorico vs Churn
    # =====================================================

    with tabs[7]:

        st.header("Analisis Categorico vs Churn")

        cat_cols = analyzer.get_categorical_columns()

        variable = st.selectbox(
            "Seleccione",
            cat_cols
        )

        cross = pd.crosstab(
            df[variable],
            df["Churn"]
        )

        st.dataframe(cross)

        fig, ax = plt.subplots(figsize=(8,4))

        cross.plot(
            kind="bar",
            stacked=True,
            ax=ax
        )

        st.pyplot(fig)

    # =====================================================
    # Dinamico
    # =====================================================

    with tabs[8]:

        st.header("Analisis Dinamico")

        columnas = st.multiselect(
            "Seleccione Columnas",
            df.columns.tolist(),
            default=df.columns[:3]
        )

        if columnas:
            st.dataframe(df[columnas].head())

        filas = st.slider(
            "Cantidad de registros",
            5,
            100,
            10
        )

        st.dataframe(df[columnas].head(filas))

        mostrar = st.checkbox(
            "Mostrar Estadísticas"
        )

        if mostrar:
            st.dataframe(df.describe())

    # =====================================================
    # Hallazgos
    # =====================================================

    with tabs[9]:

        st.header("Hallazgos Clave")

        if "Churn" in df.columns:

            churn_rate = (
                df["Churn"]
                .value_counts(normalize=True)
                * 100
            )

            fig, ax = plt.subplots()

            churn_rate.plot(
                kind="pie",
                autopct="%1.1f%%",
                ax=ax
            )

            st.pyplot(fig)

            st.write(
                f"Tasa de fuga observada: "
                f"{churn_rate.get('Yes',0):.2f}%"
            )

# =====================================================
# mis conclusiones
# =====================================================

elif menu == "Conclusiones":

    st.title("Conclusiones Finales")

    st.markdown("""
    ### Conclusiones del análisis

    1. Los clientes con menor tiempo de permanencia presentan mayor tendencia a abandonar la empresa.

    2. Los contratos mensuales muestran una mayor proporción de churn frente a contratos de largo plazo.

    3. Los clientes con cargos mensuales más altos presentan tasas superiores de abandono.

    4. Los servicios complementarios como soporte técnico y seguridad en línea parecen asociarse con una menor fuga.

    5. Los métodos de pago electrónicos y contratos flexibles muestran patrones diferenciados de churn que deben ser monitoreados.

    ### Recomendaciones

    - Implementar campañas de fidelización temprana.
    - Incentivar contratos de largo plazo.
    - Ofrecer paquetes con servicios de valor agregado.
    - Analizar segmentos de alto riesgo para mejorar retención.
    - Continuar monitoreando indicadores de churn periódicamente.

    """)
