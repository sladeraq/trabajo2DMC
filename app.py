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
    page_title="Telco Customer Churn - EDA",
    layout="wide"
)

# =====================================================
# CLASE POO
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
# HOME
# =====================================================

menu = st.sidebar.radio(
    "📌 Menú Principal",
    ["🏠 Home", "📂 Carga Dataset", "📈 EDA", "📋 Conclusiones"]
)

# =====================================================
# HOME
# =====================================================

if menu == "🏠 Home":

    st.title("📊 Análisis Exploratorio de Datos - Telco Customer Churn")

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
        **Autor:** TU NOMBRE COMPLETO

        **Curso:** Especialización Python para Data Analytics

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

    st.subheader("🛠 Tecnologías utilizadas")

    st.write("""
    Python, Pandas, NumPy, Streamlit,
    Matplotlib y Seaborn.
    """)

# =====================================================
# CARGA DATASET
# =====================================================

elif menu == "📂 Carga Dataset":

    st.title("📂 Carga del Dataset")

    uploaded_file = st.file_uploader(
        "Seleccione TelcoCustomerChurn.csv",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

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

elif menu == "📈 EDA":

    st.title("📈 Exploratory Data Analysis")

    if "df" not in st.session_state:
        st.warning("Primero debe cargar el dataset.")
        st.stop()

    df = st.session_state["df"]

    analyzer = DataAnalyzer(df)

    tabs = st.tabs([
        "1️⃣ Info",
        "2️⃣ Variables",
        "3️⃣ Estadísticas",
        "4️⃣ Nulos",
        "5️⃣ Numéricas",
        "6️⃣ Categóricas",
        "7️⃣ Num vs Cat",
        "8️⃣ Cat vs Cat",
        "9️⃣ Dinámico",
        "🔟 Hallazgos"
    ])

    # =====================================================
    # ITEM 1
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
    # ITEM 2
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
    # ITEM 3
    # =====================================================

    with tabs[2]:

        st.header("Estadísticas Descriptivas")

        st.dataframe(analyzer.descriptive_stats())

        st.subheader("Moda")

        st.dataframe(analyzer.mode_values())

    # =====================================================
    # ITEM 4
    # =====================================================

    with tabs[3]:

        st.header("Análisis de Valores Faltantes")

        nulls = analyzer.null_values()

        st.dataframe(nulls)

        fig, ax = plt.subplots(figsize=(10,4))

        nulls.plot(kind="bar", ax=ax)

        plt.xticks(rotation=45)

        st.pyplot(fig)

    # =====================================================
    # ITEM 5
    # =====================================================

    with tabs[4]:

        st.header("Distribución Variables Numéricas")

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
    # ITEM 6
    # =====================================================

    with tabs[5]:

        st.header("Variables Categóricas")

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
    # ITEM 7
    # =====================================================

    with tabs[6]:

        st.header("Análisis Numérico vs Churn")

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
    # ITEM 8
    # =====================================================

    with tabs[7]:

        st.header("Análisis Categórico vs Churn")

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
    # ITEM 9
    # =====================================================

    with tabs[8]:

        st.header("Análisis Dinámico")

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
    # ITEM 10
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
# CONCLUSIONES
# =====================================================

elif menu == "📋 Conclusiones":

    st.title("📋 Conclusiones Finales")

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
