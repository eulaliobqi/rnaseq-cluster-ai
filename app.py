import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import sys
import os

# Adicionar o diretório atual ao path para importar o módulo
sys.path.append(os.getcwd())

from rnaseq_cluster_ai import RNAseqClusterAI

# Configuração da página
st.set_page_config(
    page_title="RNA-seq Cluster AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título e descrição
st.title("🧬 RNA-seq Cluster AI")
st.markdown("""
Ferramenta interativa para análise de agrupamento de dados de RNA-seq utilizando abordagens de IA.
Carregue seus dados do featureCounts e explore os agrupamentos de suas réplicas biológicas.
""")

# Sidebar para upload e configurações
with st.sidebar:
    st.header("📁 Entrada de Dados")
    
    # Upload do arquivo de contagens
    counts_file = st.file_uploader(
        "Arquivo de output do featureCounts",
        type=['txt', 'tsv'],
        help="Selecione o arquivo tabular gerado pelo featureCounts"
    )
    
    # Upload do arquivo de metadados (opcional)
    metadata_file = st.file_uploader(
        "Arquivo de metadados (opcional)",
        type=['csv', 'txt'],
        help="CSV com informações das amostras. Primeira coluna deve ser a condição experimental."
    )
    
    st.header("⚙️ Parâmetros de Análise")
    
    # Parâmetros de pré-processamento
    n_genes = st.slider(
        "Número de genes mais variáveis",
        min_value=1000,
        max_value=10000,
        value=5000,
        help="Seleciona os genes com maior variância para análise"
    )
    
    # Parâmetros de redução dimensional
    n_components = st.slider(
        "Componentes principais (PCA)",
        min_value=2,
        max_value=50,
        value=11,
        help="Número de componentes para redução dimensional"
    )
    
    # Parâmetros UMAP
    n_neighbors = st.slider(
        "Vizinhos (UMAP)",
        min_value=2,
        max_value=20,
        value=5,
        help="Controla o balanceamento entre estrutura local e global"
    )
    
    min_dist = st.slider(
        "Distância mínima (UMAP)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Distância mínima entre pontos no espaço embutido"
    )
    
    # Parâmetros de clusterização
    min_cluster_size = st.slider(
        "Tamanho mínimo de cluster",
        min_value=2,
        max_value=10,
        value=2,
        help="Parâmetro do HDBSCAN para o tamanho mínimo de clusters"
    )
    
    # Botão para executar análise
    run_analysis = st.button("▶️ Executar Análise", type="primary")

# Área principal do aplicativo
if counts_file is not None:
    # Carregar dados
    with st.spinner("Carregando dados..."):
        # Inicializar analisador
        cluster_ai = RNAseqClusterAI(
            n_components=n_components, 
            random_state=42
        )
        
        # Processar dados
        X_scaled = cluster_ai.load_featurecounts_data(
            counts_file, 
            metadata_file,
            n_genes=n_genes
        )
        
        # Mostrar preview dos dados
        st.header("📊 Visualização dos Dados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dados de Contagem (Primeiras 5 amostras)")
            st.dataframe(cluster_ai.X.head(), use_container_width=True)
            st.caption(f"Forma dos dados: {cluster_ai.X.shape}")
        
        # Processar metadados se fornecido
        if cluster_ai.metadata is not None:
            with col2:
                st.subheader("Metadados")
                st.dataframe(cluster_ai.metadata, use_container_width=True)
    
    # Executar análise quando o botão for pressionado
    if run_analysis:
        with st.status("Executando análise...", expanded=True) as status:
            # Redução dimensional com PCA
            st.write("Aplicando PCA...")
            X_pca, explained_variance = cluster_ai.apply_pca()
            
            # Redução dimensional com UMAP
            st.write("Aplicando UMAP...")
            X_umap = cluster_ai.apply_umap(
                n_neighbors=n_neighbors,
                min_dist=min_dist
            )
            
            # Treinar autoencoder
            st.write("Treinando autoencoder...")
            autoencoder, encoder = cluster_ai.build_autoencoder(encoding_dim=10)
            X_encoded, history = cluster_ai.train_autoencoder(epochs=50)
            
            # Clusterização
            st.write("Realizando clusterização...")
            representations = {
                'PCA': X_pca,
                'UMAP': X_umap,
                'Autoencoder': X_encoded
            }
            
            results = {}
            for name, X_rep in representations.items():
                clusters_kmeans = cluster_ai.cluster_kmeans(X_rep)
                results[f'{name}_Kmeans'] = cluster_ai.evaluate_clusters(clusters_kmeans)
            
            status.update(label="Análise concluída!", state="complete")
        
        # Mostrar resultados
        st.header("📈 Resultados da Análise")
        
        # Métricas de avaliação
        st.subheader("Métricas de Avaliação")
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)
        
        # Visualizações
        st.subheader("Visualizações")
        
        # Criar abas para diferentes visualizações
        tab1, tab2, tab3 = st.tabs(["PCA", "UMAP", "Autoencoder"])
        
        with tab1:
            clusters = cluster_ai.cluster_kmeans(X_pca)
            # Pegar a variância explicada dos dois primeiros componentes
            explained_variance_2d = (explained_variance[0], explained_variance[1])
            fig = cluster_ai.visualize_clusters(
                X_pca[:, :2], 
                clusters, 
                title="Clusters baseados em PCA",
                explained_variance=explained_variance_2d
            )
            st.pyplot(fig)
        
        with tab2:
            clusters = cluster_ai.cluster_kmeans(X_umap)
            fig = cluster_ai.visualize_clusters(
                X_umap, 
                clusters, 
                title="Clusters baseados em UMAP"
            )
            st.pyplot(fig)
        
        with tab3:
            clusters = cluster_ai.cluster_kmeans(X_encoded)
            fig = cluster_ai.visualize_clusters(
                X_encoded[:, :2], 
                clusters, 
                title="Clusters baseados em Autoencoder"
            )
            st.pyplot(fig)
        
        # Download dos resultados
        st.subheader("📥 Download dos Resultados")
        
        # Converter resultados para CSV
        csv = results_df.to_csv()
        st.download_button(
            label="Baixar resultados como CSV",
            data=csv,
            file_name="resultados_analise.csv",
            mime="text/csv"
        )
        
else:
    st.info("👈 Por favor, faça upload de um arquivo de output do featureCounts para iniciar a análise.")
    
    # Seção de exemplos (opcional)
    with st.expander("📋 Exemplo de formato dos dados"):
        st.markdown("""
        **Arquivo de contagens (featureCounts):**
        - Formato TSV com as primeiras colunas: Geneid, Chr, Start, End, Strand, Length
        - Colunas seguintes: amostras com contagens de reads
        
        **Arquivo de metadados (opcional):**
        - CSV com amostras na primeira coluna e condições experimentais na segunda coluna
        - Exemplo:
        ```
        Sample,Condition
        Sample1,Treatment
        Sample2,Control
        Sample3,Treatment
        Sample4,Control
        ```
        """)

# Rodapé
st.markdown("---")
st.caption("RNA-seq Cluster AI - Ferramenta de análise de agrupamento para dados de RNA-seq")