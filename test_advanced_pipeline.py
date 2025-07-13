#!/usr/bin/env python3
"""
Test Script for Advanced Query Clustering Pipeline
=================================================

This script demonstrates the advanced pipeline with sample data and various configurations.
"""

import pandas as pd
import numpy as np
from advanced_clustering_pipeline import ClusteringConfig, AdvancedQueryClusteringPipeline
from rich.console import Console
from rich.panel import Panel

console = Console()

def create_sample_data():
    """Create sample data for testing."""
    sample_queries = [
        # EMI-related queries
        "You guys provide EMI option?",
        "Do you offer Zero Percent EMI payment options?",
        "0% EMI.",
        "EMI",
        "How to get in EMI",
        "I want to buy on EMI",
        "Is EMI available",
        "No cost EMI is available?",
        
        # COD-related queries
        "COD option is availble?",
        "Can I do COD?",
        "COD",
        "Is it possible to COD",
        "DO you have COD option",
        "Can it deliver by COD",
        "Is COD option available",
        "Can I get COD option?",
        
        # Mattress features
        "Features of Ortho mattress",
        "What are the key features of the SOF Ortho mattress",
        "SOF ortho",
        "Ortho mattress",
        "Tell me about SOF Ortho mattress",
        "What are the key features of the SOF Ergo mattress",
        "Features of Ergo mattress",
        "SOF ergo",
        "Ergo mattress",
        "Tell me about SOF Ergo mattress",
        
        # Warranty queries
        "What is the warranty period?",
        "Warranty",
        "Does mattress cover is included in warranty",
        "How long is the warranty you offer on your mattresses",
        "Do you offer warranty",
        "Tell me about the product warranty",
        "Share warranty information",
        "Need to know the warranty details",
        
        # Trial queries
        "How does the 100 night trial work",
        "What is the 100-night offer",
        "100 night trial",
        "Can I get free trial",
        "100 Nights trial version",
        "Can you give me 100 night trial",
        "100 free Nights",
        
        # Size queries
        "I want to change the size of the mattress.",
        "Custom size",
        "Customise size",
        "Can mattress size be customised?",
        "Mattress size change",
        "Can you help with the size?",
        "How do I know what size to order?",
        "What are the available sizes?",
        "Mattress size",
        "What size to order?",
        
        # Price queries
        "Price of mattress",
        "Mattress cost",
        "Cost of mattress",
        "How much does a SOF mattress cost",
        "Cost",
        "What does the mattress cost",
        "I need price",
        "Price Range",
        "Mattress price",
        "Price of Mattress",
        "Want to know the price",
        "How Much Cost",
        "Price",
        "Rate",
        "MRP",
        "Low price",
        "What will be the price",
        
        # Order status
        "Order Status",
        "What is my order status?",
        "Status of my order",
        "What about my order",
        "Where is my order",
        "Track order",
        "I want updates of my order",
        "I need my order status",
        "When will the order be delivered to me?",
        
        # Delivery queries
        "Do you deliver to my pincode",
        "Check pincode",
        "Is delivery possible on this pincode",
        "Will you be able to deliver here",
        "Can you make delivery on this pin code?",
        "Can you deliver on my pincode",
        "Can you please deliver on my pincode",
        "Need a delivery on this pincode",
        "Can I get delivery on this pincode",
        
        # Store location
        "Do you have any showrooms in Delhi state",
        "Do you have any distributors in Mumbai city",
        "Do you have any retailers in Pune city",
        "Where can I see the product before I buy",
        "Distributors",
        "Where is your showroom",
        "Do you have a showroom",
        "Can I visit SOF mattress showroom",
        "Need store",
        "Need Nearby Store",
        "Nearest shop",
        "Is there any offline stores",
        "Do you have store",
        "Where is the shop",
        "Is it available in shops",
        "You have any branch",
        "Store in",
        "We want dealer ship",
        "Any shop that I can visit",
        "Dealership",
        "Shop near by",
        "Need dealership",
        "Outlet",
        "Store near me"
    ]
    
    # Create DataFrame
    df = pd.DataFrame({'sentence': sample_queries})
    df.to_csv('sample_data.csv', index=False)
    
    console.print(f"[green]✓ Created sample data with {len(sample_queries)} queries[/green]")
    return 'sample_data.csv'

def test_basic_configuration():
    """Test the pipeline with basic configuration."""
    console.print(Panel.fit("[bold blue]Testing Basic Configuration[/bold blue]"))
    
    config = ClusteringConfig(
        remove_stopwords=True,
        lemmatize=True,
        use_ensemble_embeddings=False,  # Use single model for speed
        embedding_models=['all-MiniLM-L6-v2'],
        clustering_methods=['agglomerative', 'hdbscan'],
        evaluation_metrics=['silhouette'],
        visualization_methods=['umap'],
        save_plots=True
    )
    
    pipeline = AdvancedQueryClusteringPipeline('sample_data.csv', 'sentence', config)
    results = pipeline.run_pipeline()
    
    return results

def test_advanced_configuration():
    """Test the pipeline with advanced configuration."""
    console.print(Panel.fit("[bold blue]Testing Advanced Configuration[/bold blue]"))
    
    config = ClusteringConfig(
        remove_stopwords=True,
        lemmatize=True,
        use_ensemble_embeddings=True,
        embedding_models=['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
        clustering_methods=['agglomerative', 'spectral', 'hdbscan'],
        evaluation_metrics=['silhouette', 'calinski_harabasz', 'davies_bouldin'],
        visualization_methods=['umap', 'tsne'],
        save_plots=True
    )
    
    pipeline = AdvancedQueryClusteringPipeline('sample_data.csv', 'sentence', config)
    results = pipeline.run_pipeline()
    
    return results

def analyze_results(results):
    """Analyze and display results."""
    console.print(Panel.fit("[bold green]Results Analysis[/bold green]"))
    
    # Display clustering results
    for method, result in results['clustering_results'].items():
        if 'error' not in result:
            console.print(f"[bold]{method.title()}:[/bold]")
            console.print(f"  Clusters: {result['n_clusters']}")
            console.print(f"  Silhouette: {results['evaluation_results'].get(method, {}).get('silhouette', 'N/A'):.4f}")
            
            # Show some cluster labels
            cluster_labels = results['cluster_labels'].get(method, {})
            for cluster_id, info in list(cluster_labels.items())[:3]:
                console.print(f"  Cluster {cluster_id}: {info['label']}")
                console.print(f"    Keywords: {', '.join(info['keywords'][:3])}")
                console.print(f"    Size: {info['cluster_size']} queries")

def main():
    """Main test function."""
    console.print(Panel.fit("[bold yellow]Advanced Query Clustering Pipeline - Test Suite[/bold yellow]"))
    
    # Create sample data
    data_file = create_sample_data()
    
    # Test basic configuration
    console.print("\n[bold]1. Testing Basic Configuration[/bold]")
    basic_results = test_basic_configuration()
    analyze_results(basic_results)
    
    # Test advanced configuration
    console.print("\n[bold]2. Testing Advanced Configuration[/bold]")
    advanced_results = test_advanced_configuration()
    analyze_results(advanced_results)
    
    console.print(Panel.fit("[bold green]Test completed successfully![/bold green]"))
    console.print("Check the 'advanced_results' folder for detailed outputs.")

if __name__ == "__main__":
    main() 