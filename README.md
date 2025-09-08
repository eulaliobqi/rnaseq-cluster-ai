RNA-seq Cluster AI 🧬

Ferramenta interativa de análise de agrupamento para dados de RNA-seq utilizando abordagens de IA. Esta aplicação permite carregar dados de output do featureCounts, visualizar agrupamentos de réplicas biológicas e aplicar múltiplos algoritmos de clusterização inteligente.



✨ Funcionalidades
Pré-processamento Inteligente: Seleção automática de genes mais variáveis e normalização de dados (transformação logarítmica e escalonamento).

Redução de Dimensionalidade: Suporte a PCA, UMAP e Autoencoders para visualização de dados.

Algoritmos de Clusterização: K-means e HDBSCAN para identificação de grupos.

Visualização Interativa: Gráficos high-quality com legendas personalizadas e porcentagem de variância explicada.

Interface Streamlit: Interface amigável para upload de dados e ajuste de parâmetros.

🧪 Princípios Estatísticos e Abordagens de IA
📊 Fundamentos Estatísticos
A análise de dados de RNA-seq requer abordagens estatísticas robustas para lidar com a natureza de contagem dos dados e a alta variabilidade técnica e biológica. Nossa ferramenta incorpora:

Normalização: Transformação logarítmica (log2(x+1)) e escalonamento (StandardScaler) para estabilizar a variância e remover viéses técnicos.

Seleção de Genes: Foco nos genes mais variáveis (padrão: 5000) para reduzir ruído e dimensionalidade.

Métricas de Avaliação:

Silhouette Score: Mede a qualidade dos clusters com base na coesão e separação.

Adjusted Rand Index (ARI) e Mutual Information (AMI): Comparam clusters com labels verdadeiros (quando disponíveis).

🤖 Abordagens de IA
PCA (Análise de Componentes Principais): Redução linear de dimensionalidade, com eixos representados como porcentagem de variância explicada.

UMAP (Uniform Manifold Approximation and Projection): Redução não-linear que preserva estruturas locais e globais.

Autoencoders: Redes neurais profundas para aprendizado de representações latentes dos dados, capturando padrões complexos não lineares.

Clusterização:

K-means: Particiona dados em k clusters, com número determinado pelo método do cotovelo.

HDBSCAN: Identifica clusters baseados em densidade, sem necessidade de especificar k.

📈 Resultados e Interpretação
A ferramenta gera:

Métricas de Avaliação: Tabela comparativa de Silhouette Score, ARI e AMI para cada método.

Visualizações: Gráficos de dispersão para PCA, UMAP e Autoencoder, com cores para clusters e formas para condições experimentais.

Download de Resultados: CSV com métricas de agrupamento para análise posterior.

Exemplo de Saída
<img width="1062" height="713" alt="image" src="https://github.com/user-attachments/assets/d5b60ef4-6794-4ecc-8ead-b9018f18634c" />


🚀 Como Usar
Pré-requisitos
Python 3.8+
Git instalado

