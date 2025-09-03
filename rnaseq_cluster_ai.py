import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo e tamanho de fonte para os gráficos
plt.rcParams.update({'font.size': 14})
sns.set_style("whitegrid")

class RNAseqClusterAI:
    def __init__(self, n_components=50, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def load_featurecounts_data(self, counts_file, metadata_file=None, n_genes=5000, skip_columns=6):
        """
        Carrega e pré-processa dados de output do featureCounts
        
        Args:
            counts_file: caminho para o arquivo ou objeto de arquivo do output do featureCounts
            metadata_file: caminho ou objeto de arquivo para o arquivo de metadados (opcional)
            n_genes: Número de genes mais variáveis a selecionar
            skip_columns: Número de colunas iniciais a pular (Geneid, Chr, Start, End, Strand, Length)
        """
        print("Carregando dados do featureCounts...")
        
        # Carregar dados do featureCounts (separado por tabulação)
        if isinstance(counts_file, str):
            self.data = pd.read_csv(counts_file, sep='\t', comment='#')
        else:
            counts_file.seek(0)
            self.data = pd.read_csv(counts_file, sep='\t', comment='#')
        
        # Extrair nomes das amostras (colunas após as colunas de metadados)
        self.sample_names = self.data.columns[skip_columns:].tolist()
        
        # Criar matriz de expressão (genes x amostras)
        self.count_matrix = self.data.iloc[:, skip_columns:]
        self.count_matrix.index = self.data.iloc[:, 0]  # Usar Geneid como índice
        
        # Carregar metadados se disponível
        if metadata_file is not None:
            if isinstance(metadata_file, str):
                self.metadata = pd.read_csv(metadata_file, index_col=0)
            else:
                metadata_file.seek(0)
                self.metadata = pd.read_csv(metadata_file, index_col=0)
            # Garantir que a ordem das amostras é a mesma
            self.metadata = self.metadata.loc[self.sample_names]
            self.conditions = self.metadata.iloc[:, 0].values  # Assume que a primeira coluna são as condições
        else:
            self.metadata = None
            self.conditions = None
            
        # Transpor para ter amostras nas linhas (como necessário para análise)
        self.X = self.count_matrix.T
        
        # Pré-processamento: selecionar genes mais variáveis
        print(f"Selecionando {n_genes} genes mais variáveis...")
        gene_vars = self.X.var(axis=0)
        top_genes = gene_vars.nlargest(min(n_genes, len(gene_vars))).index
        self.X_filtered = self.X[top_genes]
        
        # Normalizar os dados (transformação logarítmica + scaling)
        print("Normalizando dados...")
        # Adicionar pseudocount e aplicar log transform
        X_log = np.log2(self.X_filtered + 1)
        self.X_scaled = self.scaler.fit_transform(X_log)
        
        return self.X_scaled
    
    def apply_pca(self):
        """Aplica PCA para redução de dimensionalidade"""
        n_samples, n_features = self.X_scaled.shape
        max_components = min(n_samples, n_features)
        
        # Ajusta n_components se necessário
        if self.n_components > max_components:
            self.n_components = max_components
            print(f"Aviso: n_components ajustado para {max_components} devido ao tamanho dos dados")
        
        print("Aplicando PCA...")
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        # Calcular variância explicada
        self.explained_variance = self.pca.explained_variance_ratio_
        
        return self.X_pca, self.explained_variance
    
    def apply_umap(self, n_neighbors=5, min_dist=0.1):
        """Aplica UMAP para redução de dimensionalidade não linear"""
        print("Aplicando UMAP...")
        self.umap_reducer = umap.UMAP(
            n_components=2, 
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=self.random_state
        )
        self.X_umap = self.umap_reducer.fit_transform(self.X_scaled)
        return self.X_umap
    
    def build_autoencoder(self, encoding_dim=10):
        """Constrói um autoencoder para aprendizado de representação"""
        input_dim = self.X_scaled.shape[1]
        
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(512, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoded = Dense(256, activation='relu')(encoded)
        encoded = Dense(128, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(128, activation='relu')(encoded)
        decoded = Dense(256, activation='relu')(decoded)
        decoded = Dense(512, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        # Modelos
        self.autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        
        # Compilar
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        return self.autoencoder, self.encoder
    
    def train_autoencoder(self, epochs=100, batch_size=16):
        """Treina o autoencoder"""
        print("Treinando autoencoder...")
        history = self.autoencoder.fit(
            self.X_scaled, self.X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.2,
            verbose=0
        )
        
        # Extrair features
        self.X_encoded = self.encoder.predict(self.X_scaled)
        return self.X_encoded, history
    
    def cluster_hdbscan(self, X, min_cluster_size=2):
        """Aplica clustering HDBSCAN"""
        print("Aplicando HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        clusters = clusterer.fit_predict(X)
        return clusters
    
    def cluster_kmeans(self, X, n_clusters=None):
        """Aplica clustering K-means"""
        if n_clusters is None and self.conditions is not None:
            n_clusters = len(np.unique(self.conditions))
        elif n_clusters is None:
            # Usar método do cotovelo para determinar número de clusters
            inertias = []
            max_clusters = min(10, len(X))
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=self.random_state)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
            
            # Encontrar o ponto de "cotovelo"
            diffs = np.diff(inertias)
            n_clusters = np.argmin(diffs) + 2  # +2 porque diff reduz o tamanho em 1 e queremos o índice +1
        
        print(f"Aplicando K-means com {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        clusters = kmeans.fit_predict(X)
        return clusters
    
    def evaluate_clusters(self, clusters, true_labels=None):
        """Avalia a qualidade dos clusters"""
        if true_labels is None and self.conditions is not None:
            true_labels = self.conditions
        
        results = {}
        
        # Silhouette Score
        if len(np.unique(clusters)) > 1:
            results['silhouette'] = silhouette_score(self.X_scaled, clusters)
        else:
            results['silhouette'] = -1
        
        # Se temos labels verdadeiros, calcular métricas adicionais
        if true_labels is not None:
            # Converter labels verdadeiros para numéricos se necessário
            if isinstance(true_labels[0], str):
                le = LabelEncoder()
                true_labels_numeric = le.fit_transform(true_labels)
            else:
                true_labels_numeric = true_labels
                
            results['ari'] = adjusted_rand_score(true_labels_numeric, clusters)
            results['ami'] = adjusted_mutual_info_score(true_labels_numeric, clusters)
        
        return results
    
    def visualize_clusters(self, X_reduced, clusters, title="Clusters", explained_variance=None):
        """Visualiza os clusters em 2D e retorna a figura"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Se temos condições reais, usar cores diferentes para clusters e formas para condições
        if self.conditions is not None:
            unique_conditions = np.unique(self.conditions)
            unique_clusters = np.unique(clusters)
            
            # Mapear condições para formas
            condition_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'H']
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
            
            # Criar uma legenda combinada para condições e clusters
            legend_elements = []
            
            for i, condition in enumerate(unique_conditions):
                for j, cluster in enumerate(unique_clusters):
                    mask = (self.conditions == condition) & (clusters == cluster)
                    if np.any(mask):
                        scatter = ax.scatter(
                            X_reduced[mask, 0], X_reduced[mask, 1],
                            marker=condition_markers[i % len(condition_markers)],
                            c=[colors[cluster]],
                            s=150,
                            alpha=0.8,
                            edgecolors='w',
                            linewidth=1
                        )
                        # Adicionar à legenda apenas uma vez por combinação
                        if j == 0:
                            legend_elements.append(plt.Line2D([0], [0], 
                                                             marker=condition_markers[i % len(condition_markers)], 
                                                             color='w', 
                                                             markerfacecolor='gray',
                                                             markersize=12, 
                                                             label=condition))
            
            # Adicionar legenda para clusters
            for cluster in unique_clusters:
                legend_elements.append(plt.Line2D([0], [0], 
                                                 marker='o', 
                                                 color='w', 
                                                 markerfacecolor=colors[cluster],
                                                 markersize=12, 
                                                 label=f'Cluster {cluster}'))
            
            # Adicionar legenda fora do gráfico à direita
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        else:
            # Apenas colorir por cluster
            unique_clusters = np.unique(clusters)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
            
            for cluster in unique_clusters:
                mask = clusters == cluster
                ax.scatter(
                    X_reduced[mask, 0], X_reduced[mask, 1],
                    c=[colors[cluster]],
                    label=f'Cluster {cluster}',
                    s=150,
                    alpha=0.8,
                    edgecolors='w',
                    linewidth=1
                )
            
            # Adicionar legenda fora do gráfico à direita
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        
        # Configurar eixos com porcentagem para PCA
        if explained_variance is not None:
            ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=16)
            ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=16)
        else:
            ax.set_xlabel('Componente 1', fontsize=16)
            ax.set_ylabel('Componente 2', fontsize=16)
        
        ax.set_title(title, fontsize=18, pad=20)
        
        # Adicionar anotações com nomes das amostras
        for i, (x, y) in enumerate(X_reduced):
            ax.annotate(self.sample_names[i], (x, y), xytext=(5, 5), 
                         textcoords='offset points', fontsize=12, alpha=0.7)
        
        # Ajustar layout para acomodar a legenda
        plt.tight_layout()
        
        return fig