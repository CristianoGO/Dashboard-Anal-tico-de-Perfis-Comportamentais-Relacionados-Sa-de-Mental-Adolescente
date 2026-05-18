import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.cluster import KMeans, AgglomerativeClustering  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.metrics import silhouette_score  # type: ignore

st.set_page_config(
    page_title="Dashboard - Saúde Mental Adolescente",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #f1f5f9;
    }

    h1, h2, h3 {
        color: #38bdf8;
    }

    .stMetric {
        background-color: #111827;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #1e293b;
    }

    .stDataFrame {
        border-radius: 12px;
    }

    div[data-testid="stSidebar"] {
        background-color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Dashboard Analítico de Perfis Comportamentais Relacionados à Saúde Mental Adolescente")
st.markdown(
    "Análise comportamental utilizando técnicas de clustering e análise de dados."
)

uploaded_file = st.file_uploader(
    "Faça upload do dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Visualização Inicial do Dataset")
    st.dataframe(df.head())

    st.subheader("Informações Gerais")

    col1, col2, col3 = st.columns(3)

    col1.metric("Quantidade de Registros", df.shape[0])
    col2.metric("Quantidade de Variáveis", df.shape[1])
    col3.metric("Algoritmos", "K-Means / Agglomerative")

    df_model = df.copy()

    # Transformação da variável gender
    if 'gender' in df_model.columns:
        df_model['gender'] = df_model['gender'].map({
            'male': 0,
            'female': 1
        })

    # Transformação da interação social
    if 'social_interaction_level' in df_model.columns:
        df_model['social_interaction_level'] = df_model[
            'social_interaction_level'
        ].map({
            'low': 0,
            'medium': 1,
            'high': 2
        })

    # One-hot encoding para plataforma
    if 'platform_usage' in df_model.columns:
        df_model = pd.get_dummies(
            df_model,
            columns=['platform_usage'],
            drop_first=True
        )

    X = df_model.drop(columns=['depression_label'], errors='ignore')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("Método do Cotovelo")

    inertia = []
    K = range(1, 11)

    for k in K:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(K, inertia, marker='o')
    ax1.set_xlabel('Número de Clusters')
    ax1.set_ylabel('Inércia')
    ax1.set_title('Método do Cotovelo')

    st.pyplot(fig1)

    n_clusters = st.slider(
        "Escolha o número de clusters",
        min_value=2,
        max_value=6,
        value=3
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters_kmeans = kmeans.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, clusters_kmeans)

    st.subheader("Avaliação dos Clusters")

    st.metric(
        "Silhouette Score",
        round(silhouette, 3)
    )

    st.markdown(
        "Valores próximos de 1 indicam melhor separação entre os grupos."
    )

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': clusters_kmeans
    })

    st.subheader("Visualização dos Clusters via PCA")

    fig2, ax2 = plt.subplots(figsize=(5, 4))

    scatter = ax2.scatter(
        pca_df['PCA1'],
        pca_df['PCA2'],
        c=pca_df['Cluster']
    )

    ax2.set_xlabel('PCA 1')
    ax2.set_ylabel('PCA 2')
    ax2.set_title('Clusters K-Means após PCA')

    st.pyplot(fig2)

    agg = AgglomerativeClustering(n_clusters=n_clusters)
    clusters_agg = agg.fit_predict(X_scaled)

    silhouette_agg = silhouette_score(X_scaled, clusters_agg)

    st.subheader("Comparação entre Algoritmos")

    comparison_df = pd.DataFrame({
        'Algoritmo': ['K-Means', 'Agglomerative'],
        'Silhouette Score': [
            round(silhouette, 3),
            round(silhouette_agg, 3)
        ]
    })

    st.dataframe(comparison_df)

    st.subheader("Caracterização dos Clusters")

    df_clusters = X.copy()
    df_clusters['Cluster'] = clusters_kmeans

    cluster_means = df_clusters.groupby('Cluster').mean()

    fig3, ax3 = plt.subplots(figsize=(7, 4))

    sns.heatmap(
        cluster_means,
        cmap='coolwarm',
        annot=False,
        ax=ax3
    )

    ax3.set_title('Média das Variáveis por Cluster')

    st.pyplot(fig3)

    st.subheader("Matriz de Correlação")

    selected_columns = [
        'daily_social_media_hours',
        'sleep_hours',
        'stress_level',
        'anxiety_level',
        'addiction_level'
    ]

    available_columns = [
        col for col in selected_columns
        if col in df_model.columns
    ]

    correlation = df_model[available_columns].corr()  # type: ignore

    fig4, ax4 = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        correlation,
        annot=True,
        cmap='coolwarm',
        ax=ax4
    )

    ax4.set_title('Correlação entre Variáveis')

    st.pyplot(fig4)

    st.subheader("Discussão dos Resultados")

    st.markdown(
        f'''
        O valor do Silhouette Score obtido ({round(silhouette, 3)})
        indica baixa separação estrutural entre os clusters,
        sugerindo elevada sobreposição comportamental entre os
        indivíduos analisados.

        Os resultados demonstram que técnicas de clustering podem
        auxiliar na identificação de padrões comportamentais,
        embora apresentem limitações diante da complexidade do
        comportamento humano e da natureza multifatorial dos dados.
        '''
    )

else:
    st.info("Faça upload do dataset para iniciar a análise.")
