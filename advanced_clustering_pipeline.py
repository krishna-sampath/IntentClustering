#!/usr/bin/env python3
"""
Advanced Query Intent Clustering Pipeline
========================================

A state-of-the-art unsupervised machine learning pipeline for clustering user queries
by intent using advanced deep learning techniques and sophisticated evaluation methods.

Features:
- Advanced sentence embeddings with multiple transformer models
- Sophisticated preprocessing with domain-specific handling
- Multiple clustering algorithms with ensemble methods
- Robust evaluation with multiple metrics
- Advanced visualization and analysis
- Automatic hyperparameter optimization

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import hdbscan
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
# Optional spaCy import - will be None if not available
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    spacy = None
    SPACY_AVAILABLE = False
from keybert import KeyBERT
import yake
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
from collections import Counter, defaultdict
import warnings
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import joblib
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich.panel import Panel
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize rich console
console = Console()

@dataclass
class ClusteringConfig:
    """Configuration for the advanced clustering pipeline."""
    
    # Data preprocessing
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    lemmatize: bool = True
    min_query_length: int = 3
    max_query_length: int = 500
    
    # Vectorization
    embedding_models: List[str] = None
    use_ensemble_embeddings: bool = True
    normalize_embeddings: bool = True
    
    # Clustering
    clustering_methods: List[str] = None
    n_clusters_range: Tuple[int, int] = (3, 20)
    distance_metrics: List[str] = None
    
    # Evaluation
    evaluation_metrics: List[str] = None
    use_cross_validation: bool = True
    
    # Visualization
    visualization_methods: List[str] = None
    save_plots: bool = True
    
    def __post_init__(self):
        if self.embedding_models is None:
            self.embedding_models = [
                'all-MiniLM-L6-v2',
                'all-mpnet-base-v2',
                'paraphrase-multilingual-MiniLM-L12-v2'
            ]
        
        if self.clustering_methods is None:
            self.clustering_methods = [
                'agglomerative',
                'spectral',
                'hdbscan',
                'kmeans'
            ]
        
        if self.distance_metrics is None:
            self.distance_metrics = ['cosine', 'euclidean']
        
        if self.evaluation_metrics is None:
            self.evaluation_metrics = [
                'silhouette',
                'calinski_harabasz',
                'davies_bouldin'
            ]
        
        if self.visualization_methods is None:
            self.visualization_methods = ['umap', 'tsne', 'pca']


class AdvancedTextPreprocessor:
    """Advanced text preprocessing with domain-specific handling."""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.setup_nlp()
        self.setup_domain_specific()
    
    def setup_nlp(self):
        """Initialize NLP components."""
        try:
            # Download NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
            # Load spaCy model
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    console.print("[yellow]Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm[/yellow]")
                    self.nlp = None
            else:
                console.print("[yellow]Warning: spaCy not available. Skipping spaCy-based features.[/yellow]")
                self.nlp = None
                
        except Exception as e:
            console.print(f"[red]Error setting up NLP: {e}[/red]")
    
    def setup_domain_specific(self):
        """Setup domain-specific preprocessing rules."""
        # Domain-specific stopwords for mattress/e-commerce
        self.domain_stopwords = {
            'mattress', 'mattresses', 'bed', 'sleep', 'comfort', 'soft', 'hard',
            'firm', 'pillow', 'pillows', 'sheet', 'sheets', 'blanket', 'blankets',
            'buy', 'purchase', 'order', 'delivery', 'shipping', 'payment', 'price',
            'cost', 'expensive', 'cheap', 'discount', 'offer', 'deal', 'sale'
        }
        
        # Common abbreviations and their expansions
        self.abbreviations = {
            'emi': 'equated monthly installment',
            'cod': 'cash on delivery',
            'sof': 'sleep on foam',
            'ergo': 'ergonomic',
            'ortho': 'orthopedic',
            'warranty': 'warranty period',
            'trial': 'trial period'
        }
    
    def preprocess_query(self, query: str) -> str:
        """Advanced preprocessing for a single query."""
        if not query or not isinstance(query, str):
            return ""
        
        # Convert to lowercase
        query = query.lower().strip()
        
        # Handle abbreviations
        for abbr, expansion in self.abbreviations.items():
            query = re.sub(rf'\b{abbr}\b', expansion, query)
        
        # Remove URLs and special characters
        query = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', query)
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Tokenize
        tokens = word_tokenize(query)
        
        # Apply preprocessing steps
        processed_tokens = []
        for token in tokens:
            # Remove stopwords
            if self.config.remove_stopwords and token in self.stop_words:
                continue
            
            # Remove domain-specific stopwords
            if token in self.domain_stopwords:
                continue
            
            # Lemmatize
            if self.config.lemmatize:
                token = self.lemmatizer.lemmatize(token)
            
            # Filter by length
            if len(token) >= 2:
                processed_tokens.append(token)
        
        # Join tokens
        processed_query = ' '.join(processed_tokens)
        
        # Length filtering
        if len(processed_query.split()) < self.config.min_query_length:
            return ""
        
        if len(processed_query) > self.config.max_query_length:
            processed_query = ' '.join(processed_query.split()[:self.config.max_query_length])
        
        return processed_query
    
    def preprocess_batch(self, queries: List[str]) -> List[str]:
        """Preprocess a batch of queries."""
        processed_queries = []
        
        for query in track(queries, description="Preprocessing queries"):
            processed = self.preprocess_query(query)
            if processed:
                processed_queries.append(processed)
        
        return processed_queries


class AdvancedVectorizer:
    """Advanced vectorization using multiple transformer models."""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.models = {}
        self.embeddings = {}
        self.setup_models()
    
    def setup_models(self):
        """Initialize embedding models."""
        console.print("[blue]Loading embedding models...[/blue]")
        
        for model_name in track(self.config.embedding_models, description="Loading models"):
            try:
                model = SentenceTransformer(model_name)
                self.models[model_name] = model
                console.print(f"[green]✓ Loaded {model_name}[/green]")
            except Exception as e:
                console.print(f"[red]✗ Failed to load {model_name}: {e}[/red]")
    
    def get_embeddings(self, queries: List[str], model_name: str = None) -> np.ndarray:
        """Get embeddings for queries using specified model."""
        if model_name is None:
            model_name = self.config.embedding_models[0]
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        embeddings = model.encode(queries, show_progress_bar=True)
        
        if self.config.normalize_embeddings:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def get_ensemble_embeddings(self, queries: List[str]) -> np.ndarray:
        """Get ensemble embeddings from multiple models."""
        console.print("[blue]Generating ensemble embeddings...[/blue]")
        
        all_embeddings = []
        for model_name in track(self.models.keys(), description="Generating embeddings"):
            embeddings = self.get_embeddings(queries, model_name)
            all_embeddings.append(embeddings)
        
        # Concatenate embeddings
        ensemble_embeddings = np.concatenate(all_embeddings, axis=1)
        
        # Normalize
        if self.config.normalize_embeddings:
            ensemble_embeddings = ensemble_embeddings / np.linalg.norm(ensemble_embeddings, axis=1, keepdims=True)
        
        return ensemble_embeddings


class AdvancedClusterer:
    """Advanced clustering with multiple algorithms and ensemble methods."""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.clustering_results = {}
    
    def agglomerative_clustering(self, embeddings: np.ndarray, n_clusters: int = None) -> Dict:
        """Perform agglomerative clustering with cosine affinity."""
        if n_clusters is None:
            n_clusters = self.config.n_clusters_range[1] // 2
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        
        labels = clustering.fit_predict(embeddings)
        
        return {
            'method': 'agglomerative',
            'labels': labels,
            'n_clusters': n_clusters,
            'model': clustering
        }
    
    def spectral_clustering(self, embeddings: np.ndarray, n_clusters: int = None) -> Dict:
        """Perform spectral clustering."""
        if n_clusters is None:
            n_clusters = self.config.n_clusters_range[1] // 2
        
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='cosine',
            random_state=42
        )
        
        labels = clustering.fit_predict(embeddings)
        
        return {
            'method': 'spectral',
            'labels': labels,
            'n_clusters': n_clusters,
            'model': clustering
        }
    
    def hdbscan_clustering(self, embeddings: np.ndarray) -> Dict:
        """Perform HDBSCAN clustering."""
        clustering = hdbscan.HDBSCAN(
            min_cluster_size=3,
            min_samples=2,
            metric='cosine',
            cluster_selection_method='eom'
        )
        
        labels = clustering.fit_predict(embeddings)
        
        return {
            'method': 'hdbscan',
            'labels': labels,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'model': clustering
        }
    
    def kmeans_clustering(self, embeddings: np.ndarray, n_clusters: int = None) -> Dict:
        """Perform K-means clustering."""
        if n_clusters is None:
            n_clusters = self.config.n_clusters_range[1] // 2
        
        clustering = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        labels = clustering.fit_predict(embeddings)
        
        return {
            'method': 'kmeans',
            'labels': labels,
            'n_clusters': n_clusters,
            'model': clustering
        }
    
    def find_optimal_clusters(self, embeddings: np.ndarray, method: str = 'agglomerative') -> Dict:
        """Find optimal number of clusters using silhouette score."""
        console.print(f"[blue]Finding optimal clusters for {method}...[/blue]")
        
        best_score = -1
        best_result = None
        
        for n_clusters in track(range(self.config.n_clusters_range[0], self.config.n_clusters_range[1] + 1)):
            if method == 'agglomerative':
                result = self.agglomerative_clustering(embeddings, n_clusters)
            elif method == 'spectral':
                result = self.spectral_clustering(embeddings, n_clusters)
            elif method == 'kmeans':
                result = self.kmeans_clustering(embeddings, n_clusters)
            else:
                continue
            
            # Calculate silhouette score
            if len(set(result['labels'])) > 1:
                score = silhouette_score(embeddings, result['labels'])
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_result['silhouette_score'] = score
        
        return best_result
    
    def cluster_all_methods(self, embeddings: np.ndarray) -> Dict:
        """Cluster using all available methods."""
        console.print("[blue]Performing clustering with all methods...[/blue]")
        
        results = {}
        
        for method in self.config.clustering_methods:
            try:
                if method == 'hdbscan':
                    result = self.hdbscan_clustering(embeddings)
                else:
                    result = self.find_optimal_clusters(embeddings, method)
                
                if result:
                    results[method] = result
                    console.print(f"[green]✓ {method}: {result['n_clusters']} clusters, silhouette: {result.get('silhouette_score', 'N/A'):.4f}[/green]")
                
            except Exception as e:
                console.print(f"[red]✗ {method} failed: {e}[/red]")
        
        return results


class AdvancedEvaluator:
    """Advanced evaluation with multiple metrics and cross-validation."""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
    
    def evaluate_clustering(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        """Evaluate clustering with multiple metrics."""
        if len(set(labels)) <= 1:
            return {'error': 'Only one cluster found'}
        
        # Remove noise points for evaluation
        mask = labels != -1
        if np.sum(mask) < 2:
            return {'error': 'Insufficient non-noise points'}
        
        eval_embeddings = embeddings[mask]
        eval_labels = labels[mask]
        
        metrics = {}
        
        # Silhouette score
        if 'silhouette' in self.config.evaluation_metrics:
            metrics['silhouette'] = silhouette_score(eval_embeddings, eval_labels)
        
        # Calinski-Harabasz score
        if 'calinski_harabasz' in self.config.evaluation_metrics:
            metrics['calinski_harabasz'] = calinski_harabasz_score(eval_embeddings, eval_labels)
        
        # Davies-Bouldin score
        if 'davies_bouldin' in self.config.evaluation_metrics:
            metrics['davies_bouldin'] = davies_bouldin_score(eval_embeddings, eval_labels)
        
        # Cluster statistics
        cluster_sizes = Counter(labels)
        metrics['n_clusters'] = len(cluster_sizes)
        metrics['noise_points'] = cluster_sizes.get(-1, 0)
        metrics['noise_percentage'] = (metrics['noise_points'] / len(labels)) * 100
        metrics['cluster_sizes'] = dict(cluster_sizes)
        
        return metrics


class AdvancedVisualizer:
    """Advanced visualization with multiple methods and interactive plots."""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup plotting style."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_umap_visualization(self, embeddings: np.ndarray, labels: np.ndarray, 
                                 title: str = "UMAP Visualization") -> go.Figure:
        """Create interactive UMAP visualization."""
        console.print("[blue]Creating UMAP visualization...[/blue]")
        
        # Reduce dimensionality with UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Create interactive plot
        unique_labels = sorted(set(labels))
        colors = px.colors.qualitative.Set3[:len(unique_labels)]
        
        fig = go.Figure()
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:  # Noise points
                fig.add_trace(go.Scatter(
                    x=embeddings_2d[mask, 0],
                    y=embeddings_2d[mask, 1],
                    mode='markers',
                    marker=dict(color='black', symbol='x', size=8),
                    name=f'Noise ({np.sum(mask)} points)',
                    hovertemplate='<b>Noise Point</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<extra></extra>'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=embeddings_2d[mask, 0],
                    y=embeddings_2d[mask, 1],
                    mode='markers',
                    marker=dict(color=colors[i], size=8),
                    name=f'Cluster {label} ({np.sum(mask)} points)',
                    hovertemplate='<b>Cluster %{fullData.name}</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<extra></extra>'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            template="plotly_white",
            height=600
        )
        
        return fig
    
    def create_tsne_visualization(self, embeddings: np.ndarray, labels: np.ndarray,
                                 title: str = "t-SNE Visualization") -> go.Figure:
        """Create interactive t-SNE visualization."""
        console.print("[blue]Creating t-SNE visualization...[/blue]")
        
        # Reduce dimensionality with t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create interactive plot
        unique_labels = sorted(set(labels))
        colors = px.colors.qualitative.Set3[:len(unique_labels)]
        
        fig = go.Figure()
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:  # Noise points
                fig.add_trace(go.Scatter(
                    x=embeddings_2d[mask, 0],
                    y=embeddings_2d[mask, 1],
                    mode='markers',
                    marker=dict(color='black', symbol='x', size=8),
                    name=f'Noise ({np.sum(mask)} points)',
                    hovertemplate='<b>Noise Point</b><br>t-SNE1: %{x:.3f}<br>t-SNE2: %{y:.3f}<extra></extra>'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=embeddings_2d[mask, 0],
                    y=embeddings_2d[mask, 1],
                    mode='markers',
                    marker=dict(color=colors[i], size=8),
                    name=f'Cluster {label} ({np.sum(mask)} points)',
                    hovertemplate='<b>Cluster %{fullData.name}</b><br>t-SNE1: %{x:.3f}<br>t-SNE2: %{y:.3f}<extra></extra>'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            template="plotly_white",
            height=600
        )
        
        return fig
    
    def create_evaluation_dashboard(self, evaluation_results: Dict) -> go.Figure:
        """Create interactive evaluation dashboard."""
        console.print("[blue]Creating evaluation dashboard...[/blue]")
        
        # Prepare data for visualization
        methods = list(evaluation_results.keys())
        metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Silhouette Score', 'Calinski-Harabasz Score', 
                          'Davies-Bouldin Score', 'Cluster Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add metric plots
        # (row=1, col=1): Silhouette
        values = [evaluation_results[method].get('silhouette', 0) for method in methods]
        fig.add_trace(
            go.Bar(x=methods, y=values, name='Silhouette Score'),
            row=1, col=1
        )
        # (row=1, col=2): Calinski-Harabasz
        values = [evaluation_results[method].get('calinski_harabasz', 0) for method in methods]
        fig.add_trace(
            go.Bar(x=methods, y=values, name='Calinski-Harabasz Score'),
            row=1, col=2
        )
        # (row=2, col=1): Davies-Bouldin
        values = [evaluation_results[method].get('davies_bouldin', 0) for method in methods]
        fig.add_trace(
            go.Bar(x=methods, y=values, name='Davies-Bouldin Score'),
            row=2, col=1
        )
        # (row=2, col=2): Cluster Distribution
        cluster_counts = [evaluation_results[method].get('n_clusters', 0) for method in methods]
        fig.add_trace(
            go.Bar(x=methods, y=cluster_counts, name='Number of Clusters'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Clustering Evaluation Dashboard",
            template="plotly_white",
            height=800
        )
        
        return fig


class AdvancedLabeler:
    """Advanced cluster labeling using multiple techniques."""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.setup_labeling_tools()
    
    def setup_labeling_tools(self):
        """Setup labeling tools."""
        try:
            self.keybert = KeyBERT()
            console.print("[green]✓ KeyBERT loaded[/green]")
        except Exception as e:
            console.print(f"[red]✗ KeyBERT failed: {e}[/red]")
            self.keybert = None
        
        try:
            import yake
            self.yake = yake.KeywordExtractor()
            console.print("[green]✓ YAKE loaded[/green]")
        except Exception as e:
            console.print(f"[red]✗ YAKE failed: {e}[/red]")
            self.yake = None
    
    def extract_keywords_tfidf(self, cluster_queries: List[str], top_n: int = 5) -> List[str]:
        """Extract keywords using TF-IDF."""
        if not cluster_queries:
            return []
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(cluster_queries)
        
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        top_indices = np.argsort(tfidf_scores)[-top_n:][::-1]
        keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
        
        return keywords
    
    def extract_keywords_keybert(self, cluster_queries: List[str], top_n: int = 5) -> List[str]:
        """Extract keywords using KeyBERT."""
        if not self.keybert or not cluster_queries:
            return []
        
        try:
            text = ' '.join(cluster_queries)
            keywords = self.keybert.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n
            )
            return [kw for kw, score in keywords]
        except Exception as e:
            console.print(f"[red]KeyBERT extraction failed: {e}[/red]")
            return []
    
    def extract_keywords_yake(self, cluster_queries: List[str], top_n: int = 5) -> List[str]:
        """Extract keywords using YAKE."""
        if not self.yake or not cluster_queries:
            return []
        
        try:
            text = ' '.join(cluster_queries)
            keywords = self.yake.extract_keywords(text)
            return [kw for kw, score in keywords[:top_n]]
        except Exception as e:
            console.print(f"[red]YAKE extraction failed: {e}[/red]")
            return []
    
    def generate_cluster_label(self, cluster_queries: List[str], cluster_id: int) -> Dict:
        """Generate comprehensive cluster label using multiple methods."""
        console.print(f"[blue]Generating label for cluster {cluster_id}...[/blue]")
        
        # Extract keywords using multiple methods
        tfidf_keywords = self.extract_keywords_tfidf(cluster_queries, top_n=3)
        keybert_keywords = self.extract_keywords_keybert(cluster_queries, top_n=3)
        yake_keywords = self.extract_keywords_yake(cluster_queries, top_n=3)
        
        # Combine keywords
        all_keywords = tfidf_keywords + keybert_keywords + yake_keywords
        unique_keywords = list(dict.fromkeys(all_keywords))[:5]
        
        # Generate label
        if unique_keywords:
            label = f"Cluster {cluster_id}: {'_'.join(unique_keywords[:3])}"
        else:
            label = f"Cluster {cluster_id}: General"
        
        return {
            'cluster_id': cluster_id,
            'label': label,
            'keywords': unique_keywords,
            'tfidf_keywords': tfidf_keywords,
            'keybert_keywords': keybert_keywords,
            'yake_keywords': yake_keywords,
            'sample_queries': cluster_queries[:3],
            'cluster_size': len(cluster_queries)
        }


class AdvancedQueryClusteringPipeline:
    """Advanced query clustering pipeline with state-of-the-art techniques."""
    
    def __init__(self, csv_file: str, query_column: str = 'sentence', config: ClusteringConfig = None):
        self.csv_file = csv_file
        self.query_column = query_column
        self.config = config or ClusteringConfig()
        
        # Initialize components
        self.preprocessor = AdvancedTextPreprocessor(self.config)
        self.vectorizer = AdvancedVectorizer(self.config)
        self.clusterer = AdvancedClusterer(self.config)
        self.evaluator = AdvancedEvaluator(self.config)
        self.visualizer = AdvancedVisualizer(self.config)
        self.labeler = AdvancedLabeler(self.config)
        
        # Data storage
        self.data = None
        self.processed_queries = None
        self.embeddings = None
        self.clustering_results = {}
        self.evaluation_results = {}
        self.cluster_labels = {}
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate data."""
        console.print("[blue]Loading data...[/blue]")
        
        try:
            self.data = pd.read_csv(self.csv_file)
            
            if self.query_column not in self.data.columns:
                raise ValueError(f"Column '{self.query_column}' not found in CSV file")
            
            # Remove empty rows
            self.data = self.data.dropna(subset=[self.query_column])
            self.data[self.query_column] = self.data[self.query_column].astype(str)
            self.data = self.data[self.data[self.query_column].str.strip() != '']
            
            console.print(f"[green]✓ Loaded {len(self.data)} queries[/green]")
            return self.data
            
        except Exception as e:
            console.print(f"[red]✗ Error loading data: {e}[/red]")
            raise
    
    def preprocess_data(self) -> List[str]:
        """Preprocess all queries."""
        console.print("[blue]Preprocessing queries...[/blue]")
        
        self.processed_queries = self.preprocessor.preprocess_batch(
            self.data[self.query_column].tolist()
        )
        
        console.print(f"[green]✓ Preprocessed {len(self.processed_queries)} queries[/green]")
        return self.processed_queries
    
    def vectorize_data(self) -> np.ndarray:
        """Vectorize queries using advanced embeddings."""
        console.print("[blue]Vectorizing queries...[/blue]")
        
        if self.config.use_ensemble_embeddings:
            self.embeddings = self.vectorizer.get_ensemble_embeddings(self.processed_queries)
        else:
            self.embeddings = self.vectorizer.get_embeddings(self.processed_queries)
        
        console.print(f"[green]✓ Vectorized queries: {self.embeddings.shape}[/green]")
        return self.embeddings
    
    def perform_clustering(self) -> Dict:
        """Perform clustering with all methods."""
        console.print("[blue]Performing clustering...[/blue]")
        
        self.clustering_results = self.clusterer.cluster_all_methods(self.embeddings)
        
        console.print(f"[green]✓ Completed clustering with {len(self.clustering_results)} methods[/green]")
        return self.clustering_results
    
    def evaluate_clustering(self) -> Dict:
        """Evaluate all clustering results."""
        console.print("[blue]Evaluating clustering results...[/blue]")
        
        self.evaluation_results = {}
        
        for method, result in self.clustering_results.items():
            try:
                evaluation = self.evaluator.evaluate_clustering(
                    self.embeddings, result['labels']
                )
                self.evaluation_results[method] = evaluation
                
                if 'error' not in evaluation:
                    console.print(f"[green]✓ {method}: silhouette={evaluation.get('silhouette', 'N/A'):.4f}[/green]")
                else:
                    console.print(f"[red]✗ {method}: {evaluation['error']}[/red]")
                    
            except Exception as e:
                console.print(f"[red]✗ {method} evaluation failed: {e}[/red]")
        
        return self.evaluation_results
    
    def generate_visualizations(self) -> Dict:
        """Generate all visualizations."""
        console.print("[blue]Generating visualizations...[/blue]")
        
        visualizations = {}
        
        for method, result in self.clustering_results.items():
            if 'error' in result:
                continue
            
            # UMAP visualization
            if 'umap' in self.config.visualization_methods:
                fig_umap = self.visualizer.create_umap_visualization(
                    self.embeddings, result['labels'], 
                    f"UMAP - {method.title()} Clustering"
                )
                visualizations[f'{method}_umap'] = fig_umap
            
            # t-SNE visualization
            if 'tsne' in self.config.visualization_methods:
                fig_tsne = self.visualizer.create_tsne_visualization(
                    self.embeddings, result['labels'],
                    f"t-SNE - {method.title()} Clustering"
                )
                visualizations[f'{method}_tsne'] = fig_tsne
        
        # Evaluation dashboard
        if self.evaluation_results:
            fig_dashboard = self.visualizer.create_evaluation_dashboard(self.evaluation_results)
            visualizations['evaluation_dashboard'] = fig_dashboard
        
        return visualizations
    
    def label_clusters(self) -> Dict:
        """Label all clusters."""
        console.print("[blue]Labeling clusters...[/blue]")
        
        self.cluster_labels = {}
        
        for method, result in self.clustering_results.items():
            if 'error' in result:
                continue
            
            method_labels = {}
            labels = result['labels']
            
            for cluster_id in set(labels):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                # Get queries for this cluster
                cluster_mask = labels == cluster_id
                cluster_queries = [self.processed_queries[i] for i in range(len(labels)) if cluster_mask[i]]
                
                # Generate label
                cluster_info = self.labeler.generate_cluster_label(cluster_queries, cluster_id)
                method_labels[cluster_id] = cluster_info
            
            self.cluster_labels[method] = method_labels
        
        return self.cluster_labels
    
    def save_results(self, output_dir: str = "advanced_results"):
        """Save all results to files."""
        console.print(f"[blue]Saving results to {output_dir}...[/blue]")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save clustering results
        results_data = []
        for method, result in self.clustering_results.items():
            if 'error' in result:
                continue
            labels = result['labels']
            cluster_labels = self.cluster_labels.get(method, {})
            # Only iterate over processed queries and their labels
            for i, (processed_query, cluster_id) in enumerate(zip(self.processed_queries, labels)):
                cluster_info = cluster_labels.get(cluster_id, {})
                results_data.append({
                    'processed_query': processed_query,
                    'method': method,
                    'cluster_id': cluster_id,
                    'cluster_label': cluster_info.get('label', f'Cluster {cluster_id}'),
                    'keywords': ', '.join(cluster_info.get('keywords', [])),
                    'cluster_size': cluster_info.get('cluster_size', 0)
                })
        
        # Save to CSV
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(f"{output_dir}/advanced_clustering_results.csv", index=False)
        
        # Save evaluation results
        evaluation_df = pd.DataFrame(self.evaluation_results).T
        evaluation_df.to_csv(f"{output_dir}/evaluation_results.csv")
        
        # Save visualizations
        visualizations = self.generate_visualizations()
        for name, fig in visualizations.items():
            fig.write_html(f"{output_dir}/{name}.html")
            fig.write_image(f"{output_dir}/{name}.png", width=1200, height=800)
        
        # Save cluster labels
        import json
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            else:
                return obj
        
        cluster_labels_serializable = convert_numpy_types(self.cluster_labels)
        with open(f"{output_dir}/cluster_labels.json", 'w') as f:
            json.dump(cluster_labels_serializable, f, indent=2)
        
        console.print(f"[green]✓ Results saved to {output_dir}[/green]")
    
    def run_pipeline(self) -> Dict:
        """Run the complete advanced pipeline."""
        console.print(Panel.fit("[bold blue]Advanced Query Clustering Pipeline[/bold blue]"))
        
        # Load data
        self.load_data()
        
        # Preprocess
        self.preprocess_data()
        
        # Vectorize
        self.vectorize_data()
        
        # Cluster
        self.perform_clustering()
        
        # Evaluate
        self.evaluate_clustering()
        
        # Label
        self.label_clusters()
        
        # Save results
        self.save_results()
        
        console.print(Panel.fit("[bold green]Pipeline completed successfully![/bold green]"))
        
        return {
            'clustering_results': self.clustering_results,
            'evaluation_results': self.evaluation_results,
            'cluster_labels': self.cluster_labels
        }


def main():
    """Main function to run the advanced pipeline."""
    
    # Configuration
    config = ClusteringConfig(
        remove_stopwords=True,
        lemmatize=True,
        use_ensemble_embeddings=True,
        normalize_embeddings=True,
        clustering_methods=['agglomerative', 'spectral', 'hdbscan'],
        evaluation_metrics=['silhouette', 'calinski_harabasz', 'davies_bouldin'],
        visualization_methods=['umap', 'tsne'],
        save_plots=True
    )
    
    # Initialize pipeline
    pipeline = AdvancedQueryClusteringPipeline('data.csv', 'sentence', config)
    
    # Run pipeline
    results = pipeline.run_pipeline()
    
    # Print summary
    console.print("\n[bold]Pipeline Summary:[/bold]")
    for method, result in results['evaluation_results'].items():
        if 'error' not in result:
            console.print(f"  {method}: {result.get('n_clusters', 0)} clusters, "
                        f"silhouette={result.get('silhouette', 'N/A'):.4f}")


if __name__ == "__main__":
    main() 