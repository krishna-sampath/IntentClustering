#!/usr/bin/env python3
"""
Setup Script for Advanced Query Clustering Pipeline
==================================================

This script helps set up the advanced pipeline environment and dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        # Install core requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Core dependencies installed")
        
        # Install spaCy model
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            print("✅ spaCy English model installed")
        except subprocess.CalledProcessError:
            print("⚠️  spaCy model installation failed (will use fallback)")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data."""
    print("📚 Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✅ NLTK data downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    required_modules = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn',
        'plotly', 'umap', 'sentence_transformers', 'torch',
        'keybert', 'nltk', 'hdbscan',
        'tqdm', 'rich'
    ]
    
    optional_modules = [
        'spacy', 'yake', 'textstat'
    ]
    
    failed_imports = []
    optional_failed = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✅ {module} (optional)")
        except Exception:
            print(f"⚠️  {module} (optional - not available or failed to import)")
            optional_failed.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import required modules: {', '.join(failed_imports)}")
        return False
    
    if optional_failed:
        print(f"\n⚠️  Optional modules not available: {', '.join(optional_failed)}")
        print("The pipeline will work without these modules. Some features may be disabled.")
    
    print("✅ All required imports successful")
    return True

def create_sample_config():
    """Create a sample configuration file."""
    config_content = '''# Sample configuration for advanced pipeline
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
'''
    
    with open('sample_config.py', 'w') as f:
        f.write(config_content)
    
    print("✅ Sample configuration created: sample_config.py")

def main():
    """Main setup function."""
    print("🚀 Advanced Query Clustering Pipeline Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Download NLTK data
    if not download_nltk_data():
        return False
    
    # Test imports
    if not test_imports():
        return False
    
    # Create sample configuration
    create_sample_config()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run: python test_advanced_pipeline.py")
    print("2. Or run: python advanced_clustering_pipeline.py")
    print("3. Check the 'advanced_results' folder for outputs")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 