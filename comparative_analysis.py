"""
Module d'analyse comparative des algorithmes de clustering pour la détection de fake news
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
import time
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from preprocess import TextPreprocessor, load_and_preprocess_data
from clustering import ClusteringModels, find_optimal_k

def run_comparative_analysis(train_data, test_data, preprocessor, n_clusters=2, feature_type='bert'):
    """
    Exécute une analyse comparative des algorithmes de clustering
    
    Args:
        train_data: DataFrame des données d'entraînement
        test_data: DataFrame des données de test
        preprocessor: Instance de TextPreprocessor
        n_clusters: Nombre de clusters
        feature_type: Type de caractéristiques ('tfidf' ou 'bert')
    
    Returns:
        DataFrame des résultats comparatifs
    """
    print(f"Analyse comparative avec {n_clusters} clusters et caractéristiques {feature_type}...")
    
    # Extraction des caractéristiques
    if feature_type == 'tfidf':
        print("Extraction des caractéristiques TF-IDF...")
        train_features, vectorizer = preprocessor.get_tfidf_vectors(train_data['processed_text'])
        test_features = vectorizer.transform(test_data['processed_text'])
        
        # Conversion en array dense pour certains algorithmes
        train_features_dense = train_features.toarray()
        test_features_dense = test_features.toarray()
    else:  # bert
        print("Extraction des embeddings BERT...")
        train_features = preprocessor.get_bert_embeddings_batch(train_data['processed_text'].values)
        test_features = preprocessor.get_bert_embeddings_batch(test_data['processed_text'].values)
        
        # Les embeddings BERT sont déjà en format dense
        train_features_dense = train_features
        test_features_dense = test_features
    
    # Initialisation du modèle de clustering
    clustering_models = ClusteringModels(n_clusters=n_clusters)
    
    # Dictionnaire pour stocker les résultats
    results = {
        'Algorithme': [],
        'Temps d\'entraînement (s)': [],
        'Score de silhouette (train)': [],
        'Score de silhouette (test)': [],
        'Indice Davies-Bouldin (train)': [],
        'Indice Davies-Bouldin (test)': []
    }
    
    # 1. KMeans
    print("\n--- KMeans ---")
    start_time = time.time()
    kmeans_labels_train = clustering_models.fit_kmeans(train_features_dense)
    kmeans_time = time.time() - start_time
    
    kmeans_labels_test = clustering_models.predict_kmeans(test_features_dense)
    
    kmeans_eval_train = clustering_models.evaluate_clustering(train_features_dense, kmeans_labels_train)
    kmeans_eval_test = clustering_models.evaluate_clustering(test_features_dense, kmeans_labels_test)
    
    results['Algorithme'].append('KMeans')
    results['Temps d\'entraînement (s)'].append(kmeans_time)
    results['Score de silhouette (train)'].append(kmeans_eval_train['silhouette_score'])
    results['Score de silhouette (test)'].append(kmeans_eval_test['silhouette_score'])
    results['Indice Davies-Bouldin (train)'].append(kmeans_eval_train['davies_bouldin_score'])
    results['Indice Davies-Bouldin (test)'].append(kmeans_eval_test['davies_bouldin_score'])
    
    # Visualisation des clusters KMeans
    kmeans_viz = clustering_models.visualize_clusters_2d(
        train_features_dense, kmeans_labels_train, 
        title="Visualisation des clusters KMeans"
    )
    kmeans_viz.savefig("results/kmeans_clusters.png")
    plt.close()
    
    # 2. Autoencodeur + KMeans
    print("\n--- Autoencodeur + KMeans ---")
    start_time = time.time()
    autoencoder_labels_train, history, encoded_data = clustering_models.fit_autoencoder(
        train_features_dense, epochs=30, batch_size=64
    )
    autoencoder_time = time.time() - start_time
    
    autoencoder_labels_test = clustering_models.predict_autoencoder(test_features_dense)
    
    autoencoder_eval_train = clustering_models.evaluate_clustering(train_features_dense, autoencoder_labels_train)
    autoencoder_eval_test = clustering_models.evaluate_clustering(test_features_dense, autoencoder_labels_test)
    
    results['Algorithme'].append('Autoencodeur + KMeans')
    results['Temps d\'entraînement (s)'].append(autoencoder_time)
    results['Score de silhouette (train)'].append(autoencoder_eval_train['silhouette_score'])
    results['Score de silhouette (test)'].append(autoencoder_eval_test['silhouette_score'])
    results['Indice Davies-Bouldin (train)'].append(autoencoder_eval_train['davies_bouldin_score'])
    results['Indice Davies-Bouldin (test)'].append(autoencoder_eval_test['davies_bouldin_score'])
    
    # Visualisation des clusters Autoencodeur + KMeans
    autoencoder_viz = clustering_models.visualize_clusters_2d(
        encoded_data, autoencoder_labels_train, 
        title="Visualisation des clusters Autoencodeur + KMeans"
    )
    autoencoder_viz.savefig("results/autoencoder_clusters.png")
    plt.close()
    
    # Visualisation de la perte de l'autoencodeur
    loss_viz = clustering_models.plot_autoencoder_loss(history)
    loss_viz.savefig("results/autoencoder_loss.png")
    plt.close()
    
    # 3. Clustering agglomératif
    print("\n--- Clustering agglomératif ---")
    start_time = time.time()
    agglomerative_labels_train = clustering_models.fit_agglomerative(train_features_dense)
    agglomerative_time = time.time() - start_time
    
    # Pour le clustering agglomératif, nous utilisons fit_predict pour les données de test
    agglomerative_labels_test = AgglomerativeClustering(
        n_clusters=n_clusters, linkage='ward'
    ).fit_predict(test_features_dense)
    
    agglomerative_eval_train = clustering_models.evaluate_clustering(train_features_dense, agglomerative_labels_train)
    agglomerative_eval_test = clustering_models.evaluate_clustering(test_features_dense, agglomerative_labels_test)
    
    results['Algorithme'].append('Clustering agglomératif')
    results['Temps d\'entraînement (s)'].append(agglomerative_time)
    results['Score de silhouette (train)'].append(agglomerative_eval_train['silhouette_score'])
    results['Score de silhouette (test)'].append(agglomerative_eval_test['silhouette_score'])
    results['Indice Davies-Bouldin (train)'].append(agglomerative_eval_train['davies_bouldin_score'])
    results['Indice Davies-Bouldin (test)'].append(agglomerative_eval_test['davies_bouldin_score'])
    
    # Visualisation des clusters Agglomératif
    agglomerative_viz = clustering_models.visualize_clusters_2d(
        train_features_dense, agglomerative_labels_train, 
        title="Visualisation des clusters Agglomératif"
    )
    agglomerative_viz.savefig("results/agglomerative_clusters.png")
    plt.close()
    
    # 4. Clustering hiérarchique
    print("\n--- Clustering hiérarchique ---")
    # Pour le clustering hiérarchique, nous utilisons un échantillon des données
    # si les données sont trop volumineuses
    sample_size = min(1000, train_features_dense.shape[0])
    
    start_time = time.time()
    hierarchical_labels_train = clustering_models.fit_hierarchical(train_features_dense, sample_size=sample_size)
    hierarchical_time = time.time() - start_time
    
    # Pour le clustering hiérarchique, nous utilisons AgglomerativeClustering pour les données de test
    hierarchical_labels_test = AgglomerativeClustering(
        n_clusters=n_clusters, linkage='ward'
    ).fit_predict(test_features_dense)
    
    hierarchical_eval_train = clustering_models.evaluate_clustering(train_features_dense, hierarchical_labels_train)
    hierarchical_eval_test = clustering_models.evaluate_clustering(test_features_dense, hierarchical_labels_test)
    
    results['Algorithme'].append('Clustering hiérarchique')
    results['Temps d\'entraînement (s)'].append(hierarchical_time)
    results['Score de silhouette (train)'].append(hierarchical_eval_train['silhouette_score'])
    results['Score de silhouette (test)'].append(hierarchical_eval_test['silhouette_score'])
    results['Indice Davies-Bouldin (train)'].append(hierarchical_eval_train['davies_bouldin_score'])
    results['Indice Davies-Bouldin (test)'].append(hierarchical_eval_test['davies_bouldin_score'])
    
    # Visualisation du dendrogramme
    dendrogram_viz = clustering_models.plot_dendrogram()
    dendrogram_viz.savefig("results/hierarchical_dendrogram.png")
    plt.close()
    
    # Visualisation des clusters Hiérarchique
    hierarchical_viz = clustering_models.visualize_clusters_2d(
        train_features_dense, hierarchical_labels_train, 
        title="Visualisation des clusters Hiérarchique"
    )
    hierarchical_viz.savefig("results/hierarchical_clusters.png")
    plt.close()
    
    # Sauvegarde des modèles
    clustering_models.save_models()
    
    # Création du DataFrame des résultats
    results_df = pd.DataFrame(results)
    
    # Sauvegarde des résultats
    results_df.to_csv("results/comparative_results.csv", index=False)
    
    # Visualisation interactive des résultats
    create_interactive_comparison(results_df)
    
    return results_df

def create_interactive_comparison(results_df):
    """
    Crée une visualisation interactive des résultats comparatifs
    """
    # Création du subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Score de silhouette", "Indice Davies-Bouldin"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Ajout des barres pour le score de silhouette
    fig.add_trace(
        go.Bar(
            x=results_df['Algorithme'],
            y=results_df['Score de silhouette (train)'],
            name='Entraînement',
            marker_color='blue'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=results_df['Algorithme'],
            y=results_df['Score de silhouette (test)'],
            name='Test',
            marker_color='green'
        ),
        row=1, col=1
    )
    
    # Ajout des barres pour l'indice Davies-Bouldin
    fig.add_trace(
        go.Bar(
            x=results_df['Algorithme'],
            y=results_df['Indice Davies-Bouldin (train)'],
            name='Entraînement',
            marker_color='blue',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=results_df['Algorithme'],
            y=results_df['Indice Davies-Bouldin (test)'],
            name='Test',
            marker_color='green',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Mise à jour de la mise en page
    fig.update_layout(
        title="Comparaison des performances des algorithmes de clustering",
        barmode='group',
        height=600,
        width=1200
    )
    
    # Sauvegarde de la figure
    fig.write_html("results/comparative_results.html")

def analyze_clusters_by_language(train_data, labels_dict):
    """
    Analyse la distribution des langues dans chaque cluster
    
    Args:
        train_data: DataFrame des données d'entraînement
        labels_dict: Dictionnaire des labels de cluster pour chaque algorithme
    """
    # Création du DataFrame pour l'analyse
    analysis_df = train_data[['language']].copy()
    
    # Ajout des labels de cluster pour chaque algorithme
    for algo, labels in labels_dict.items():
        analysis_df[f'{algo}_cluster'] = labels
    
    # Analyse de la distribution des langues par cluster pour chaque algorithme
    results = {}
    
    for algo in labels_dict.keys():
        cluster_col = f'{algo}_cluster'
        
        # Comptage des langues par cluster
        lang_dist = pd.crosstab(
            analysis_df[cluster_col], 
            analysis_df['language'],
            normalize='index'
        ) * 100
        
        results[algo] = lang_dist
    
    return results

def visualize_language_distribution(lang_dist_dict):
    """
    Visualise la distribution des langues dans chaque cluster
    
    Args:
        lang_dist_dict: Dictionnaire des distributions de langues par algorithme
    """
    n_algos = len(lang_dist_dict)
    
    fig, axes = plt.subplots(1, n_algos, figsize=(n_algos * 6, 5))
    
    for i, (algo, lang_dist) in enumerate(lang_dist_dict.items()):
        ax = axes[i] if n_algos > 1 else axes
        
        lang_dist.plot(
            kind='bar',
            stacked=True,
            colormap='viridis',
            ax=ax
        )
        
        ax.set_title(f"Distribution des langues par cluster ({algo})")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Pourcentage (%)")
        ax.legend(title="Langue")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("results/language_distribution.png")
    plt.close()

def main():
    """
    Fonction principale
    """
    # Création du répertoire pour les résultats
    os.makedirs("results", exist_ok=True)
    
    # Chemins des données
    english_train_path = "Fake_news/train.csv"
    english_test_path = "Fake_news/test (1).csv"
    french_train_path = "French_Fake_News/train.csv"
    french_test_path = "French_Fake_News/test.csv"
    
    # Chargement des données prétraitées si elles existent, sinon prétraitement
    if os.path.exists("preprocessed_train_data.csv") and os.path.exists("preprocessed_test_data.csv"):
        print("Chargement des données prétraitées...")
        train_data = pd.read_csv("preprocessed_train_data.csv")
        test_data = pd.read_csv("preprocessed_test_data.csv")
        preprocessor = TextPreprocessor()
    else:
        print("Prétraitement des données...")
        train_data, test_data, preprocessor = load_and_preprocess_data(
            english_train_path, english_test_path, french_train_path, french_test_path
        )
    
    # Extraction des caractéristiques pour trouver le nombre optimal de clusters
    print("Extraction des embeddings BERT pour l'analyse du nombre optimal de clusters...")
    sample_size = min(5000, len(train_data))
    sample_indices = np.random.choice(len(train_data), sample_size, replace=False)
    
    sample_features = preprocessor.get_bert_embeddings_batch(
        train_data.iloc[sample_indices]['processed_text'].values
    )
    
    # Recherche du nombre optimal de clusters
    k_optimal, k_plot = find_optimal_k(sample_features, k_range=range(2, 6))
    k_plot.savefig("results/optimal_k.png")
    plt.close()
    
    print(f"Nombre optimal de clusters: {k_optimal}")
    
    # Exécution de l'analyse comparative avec le nombre optimal de clusters
    results_df = run_comparative_analysis(train_data, test_data, preprocessor, n_clusters=k_optimal, feature_type='bert')
    
    # Affichage des résultats
    print("\nRésultats comparatifs:")
    print(results_df)
    
    # Création d'un tableau formaté pour le rapport
    print("\nTableau pour le rapport:")
    print(results_df.to_markdown(index=False))

if __name__ == "__main__":
    # Import nécessaire pour l'analyse des clusters
    from sklearn.cluster import AgglomerativeClustering
    
    main()
