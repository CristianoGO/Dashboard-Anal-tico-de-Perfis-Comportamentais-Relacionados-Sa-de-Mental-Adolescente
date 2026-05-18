# Dashboard Analítico de Perfis Comportamentais Relacionados à Saúde Mental Adolescente

Este projeto consiste em um dashboard analítico desenvolvido em Python utilizando Streamlit para visualização e análise de dados relacionados à saúde mental adolescente por meio de técnicas de clustering e análise comportamental.

O dashboard foi desenvolvido como complemento da monografia de pós-graduação em Sistemas de Informação, permitindo explorar visualmente os resultados obtidos pelos algoritmos de agrupamento aplicados ao dataset.

---

# Objetivo

O objetivo do dashboard é facilitar a interpretação dos padrões comportamentais identificados nos dados, utilizando recursos de visualização interativa e métricas de avaliação de clusters.

---

# Funcionalidades

O dashboard permite:

- Upload do dataset em formato CSV;
- Visualização inicial dos dados;
- Pré-processamento automático;
- Transformação de variáveis categóricas;
- Padronização dos dados com StandardScaler;
- Aplicação do algoritmo K-Means;
- Aplicação do Agglomerative Clustering;
- Método do Cotovelo;
- Avaliação utilizando Silhouette Score;
- Redução de dimensionalidade com PCA;
- Visualização dos clusters;
- Heatmap dos agrupamentos;
- Matriz de correlação entre variáveis;
- Comparação entre algoritmos.

---

# Tecnologias Utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

# Estrutura do Projeto

```bash
dashboard_tcc/
│
├── app.py
├── dataset.csv
├── requirements.txt
└── README.md
