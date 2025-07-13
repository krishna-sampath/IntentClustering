# Sample configuration for advanced pipeline
from advanced_clustering_pipeline import ClusteringConfig

# Basic configuration for quick testing
basic_config = ClusteringConfig(
    remove_stopwords=True,
    lemmatize=True,
    use_ensemble_embeddings=False,
    embedding_models=['all-MiniLM-L6-v2'],
    clustering_methods=['agglomerative', 'hdbscan'],
    evaluation_metrics=['silhouette'],
    visualization_methods=['umap'],
    save_plots=True
)

# Advanced configuration for production
advanced_config = ClusteringConfig(
    remove_stopwords=True,
    lemmatize=True,
    use_ensemble_embeddings=True,
    embedding_models=['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
    clustering_methods=['agglomerative', 'spectral', 'hdbscan'],
    evaluation_metrics=['silhouette', 'calinski_harabasz', 'davies_bouldin'],
    visualization_methods=['umap', 'tsne'],
    save_plots=True
)
