"""
Script principal pour la classification non supervisée de fake news
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Importation des modules personnalisés
from preprocessing import load_and_preprocess_data, get_tfidf_features, get_bert_embeddings
from clustering import ClusteringModels, find_optimal_k
from evaluation import ClusteringEvaluator

# Chemins des datasets
ENGLISH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Fake_news')
FRENCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'French_Fake_News')

# Répertoires de sortie
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# Créer les répertoires s'ils n'existent pas
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_clustering_analysis(n_clusters=2, use_bert=False, max_features=5000):
    """
    Exécute l'analyse de clustering sur les datasets français et anglais
    
    Args:
        n_clusters (int): Nombre de clusters
        use_bert (bool): Utiliser BERT pour les embeddings (True) ou TF-IDF (False)
        max_features (int): Nombre maximum de features pour TF-IDF
    """
    print("="*80)
    print(f"ANALYSE DE CLUSTERING AVEC {n_clusters} CLUSTERS")
    print(f"Utilisation de {'BERT' if use_bert else 'TF-IDF'} pour les embeddings")
    print("="*80)
    
    # Charger et prétraiter les données
    print("\nChargement et prétraitement des données...")
    datasets = load_and_preprocess_data(ENGLISH_PATH, FRENCH_PATH)
    
    # Initialiser l'évaluateur
    evaluator = ClusteringEvaluator()
    
    # Traiter chaque dataset (anglais et français)
    for dataset_name, dataset in datasets.items():
        print(f"\n{'='*40}")
        print(f"TRAITEMENT DU DATASET {dataset_name.upper()}")
        print(f"{'='*40}")
        
        # Extraire les features
        print("\nExtraction des features...")
        if use_bert:
            # Utiliser BERT pour les embeddings
            train_texts = dataset['train']['clean_text'].tolist()
            test_texts = dataset['test']['clean_text'].tolist()
            
            print("Génération des embeddings BERT pour l'ensemble d'entraînement...")
            X_train = get_bert_embeddings(train_texts)
            
            print("Génération des embeddings BERT pour l'ensemble de test...")
            X_test = get_bert_embeddings(test_texts)
        else:
            # Utiliser TF-IDF pour les features
            train_texts = dataset['train']['clean_text'].tolist()
            test_texts = dataset['test']['clean_text'].tolist()
            
            print("Génération des features TF-IDF...")
            X_train, X_test, vectorizer = get_tfidf_features(train_texts, test_texts, max_features)
        
        # Initialiser le modèle de clustering
        clustering = ClusteringModels(n_clusters=n_clusters)
        
        # KMeans
        print("\nEntraînement et évaluation de KMeans...")
        kmeans_labels = clustering.fit_kmeans(X_train)
        kmeans_test_labels = clustering.predict_kmeans(X_test)
        
        # Ajouter les résultats à l'évaluateur
        evaluator.add_result('kmeans', dataset_name, 'train', kmeans_labels, X_train)
        evaluator.add_result('kmeans', dataset_name, 'test', kmeans_test_labels, X_test)
        
        # Visualiser les clusters KMeans
        clustering.visualize_clusters_2d(X_train, kmeans_labels, 
                                         title=f"KMeans Clusters - {dataset_name} (train)")
        plt.savefig(os.path.join(RESULTS_DIR, f"kmeans_{dataset_name}_train.png"))
        
        # Autoencodeur + KMeans
        print("\nEntraînement et évaluation de l'autoencodeur + KMeans...")
        autoencoder_labels = clustering.fit_autoencoder(X_train, epochs=30)
        autoencoder_test_labels = clustering.predict_autoencoder(X_test)
        
        # Ajouter les résultats à l'évaluateur
        evaluator.add_result('autoencoder', dataset_name, 'train', autoencoder_labels, X_train)
        evaluator.add_result('autoencoder', dataset_name, 'test', autoencoder_test_labels, X_test)
        
        # Visualiser les clusters de l'autoencodeur
        clustering.visualize_clusters_2d(X_train, autoencoder_labels, 
                                         title=f"Autoencoder Clusters - {dataset_name} (train)")
        plt.savefig(os.path.join(RESULTS_DIR, f"autoencoder_{dataset_name}_train.png"))
        
        # Clustering agglomératif
        print("\nEntraînement et évaluation du clustering agglomératif...")
        agglomerative_labels = clustering.fit_agglomerative(X_train)
        
        # Pour le test, nous utilisons un proxy KMeans car AgglomerativeClustering n'a pas de méthode predict
        kmeans_proxy = clustering.models['agglomerative']['kmeans_proxy']
        agglomerative_test_labels = kmeans_proxy.predict(X_test)
        
        # Ajouter les résultats à l'évaluateur
        evaluator.add_result('agglomerative', dataset_name, 'train', agglomerative_labels, X_train)
        evaluator.add_result('agglomerative', dataset_name, 'test', agglomerative_test_labels, X_test)
        
        # Visualiser les clusters agglomératifs
        clustering.visualize_clusters_2d(X_train, agglomerative_labels, 
                                         title=f"Agglomerative Clusters - {dataset_name} (train)")
        plt.savefig(os.path.join(RESULTS_DIR, f"agglomerative_{dataset_name}_train.png"))
        
        # Clustering hiérarchique
        print("\nEntraînement et évaluation du clustering hiérarchique...")
        hierarchical_labels = clustering.fit_hierarchical(X_train)
        
        # Pour le test, nous utilisons un proxy KMeans
        kmeans_proxy = clustering.models['hierarchical']['kmeans_proxy']
        hierarchical_test_labels = kmeans_proxy.predict(X_test)
        
        # Ajouter les résultats à l'évaluateur
        evaluator.add_result('hierarchical', dataset_name, 'train', hierarchical_labels, X_train)
        evaluator.add_result('hierarchical', dataset_name, 'test', hierarchical_test_labels, X_test)
        
        # Visualiser les clusters hiérarchiques
        clustering.visualize_clusters_2d(X_train, hierarchical_labels, 
                                         title=f"Hierarchical Clusters - {dataset_name} (train)")
        plt.savefig(os.path.join(RESULTS_DIR, f"hierarchical_{dataset_name}_train.png"))
        
        # Tracer le dendrogramme pour le clustering hiérarchique
        clustering.plot_dendrogram(title=f"Dendrogramme - {dataset_name}")
        plt.savefig(os.path.join(RESULTS_DIR, f"dendrogram_{dataset_name}.png"))
        
        # Sauvegarder les modèles
        print("\nSauvegarde des modèles...")
        model_dir = os.path.join(MODELS_DIR, dataset_name)
        os.makedirs(model_dir, exist_ok=True)
        clustering.save_models(model_dir)
    
    # Générer le rapport d'évaluation
    print("\nGénération du rapport d'évaluation...")
    evaluator.generate_report(RESULTS_DIR)
    
    # Afficher le tableau comparatif
    comparison_table = evaluator.get_comparison_table()
    print("\nTableau comparatif des performances :")
    print(comparison_table)
    
    return evaluator, comparison_table

if __name__ == "__main__":
    # Exécuter l'analyse avec 2 clusters (fake/real)
    evaluator, comparison_table = run_clustering_analysis(n_clusters=2, use_bert=False)
    
    # Sauvegarder le tableau comparatif
    comparison_table.to_csv(os.path.join(RESULTS_DIR, 'comparison_table.csv'), index=False)
    
    print("\nAnalyse terminée. Les résultats sont disponibles dans le répertoire 'results'.")
