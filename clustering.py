"""
Module d'algorithmes de clustering pour la détection de fake news
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import time

class ClusteringModels:
    """
    Classe pour les modèles de clustering non supervisé
    """
    def __init__(self, n_clusters=2, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_model = None
        self.autoencoder_model = None
        self.encoder_model = None
        self.agglomerative_model = None
        self.hierarchical_linkage = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=random_state)
        
    def fit_kmeans(self, X):
        """Entraîne un modèle KMeans"""
        print("Entraînement du modèle KMeans...")
        start_time = time.time()
        
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans_model.fit(X)
        
        end_time = time.time()
        print(f"Temps d'entraînement KMeans: {end_time - start_time:.2f} secondes")
        
        return self.kmeans_model.labels_
    
    def predict_kmeans(self, X):
        """Prédit les clusters avec KMeans"""
        if self.kmeans_model is None:
            raise ValueError("Le modèle KMeans n'a pas été entraîné")
        
        return self.kmeans_model.predict(X)
    
    def build_autoencoder(self, input_dim, encoding_dim=32):
        """Construit un modèle d'autoencodeur"""
        # Encodeur
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(256, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(128, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        # Décodeur
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(256, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        # Modèle complet
        autoencoder = Model(input_layer, decoded)
        
        # Modèle d'encodeur uniquement
        encoder = Model(input_layer, encoded)
        
        return autoencoder, encoder
    
    def fit_autoencoder(self, X, epochs=50, batch_size=64, validation_split=0.1):
        """Entraîne un autoencodeur et applique KMeans sur les représentations encodées"""
        print("Entraînement de l'autoencodeur...")
        start_time = time.time()
        
        # Normalisation des données
        X_scaled = self.scaler.fit_transform(X)
        
        # Construction de l'autoencodeur
        input_dim = X_scaled.shape[1]
        self.autoencoder_model, self.encoder_model = self.build_autoencoder(input_dim)
        
        # Compilation de l'autoencodeur
        self.autoencoder_model.compile(optimizer='adam', loss='mse')
        
        # Entraînement de l'autoencodeur
        history = self.autoencoder_model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            verbose=1
        )
        
        # Obtention des représentations encodées
        encoded_data = self.encoder_model.predict(X_scaled)
        
        # Application de KMeans sur les représentations encodées
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(encoded_data)
        
        end_time = time.time()
        print(f"Temps d'entraînement Autoencodeur + KMeans: {end_time - start_time:.2f} secondes")
        
        return labels, history, encoded_data
    
    def predict_autoencoder(self, X):
        """Prédit les clusters avec l'autoencodeur + KMeans"""
        if self.encoder_model is None:
            raise ValueError("L'autoencodeur n'a pas été entraîné")
        
        # Normalisation des données
        X_scaled = self.scaler.transform(X)
        
        # Obtention des représentations encodées
        encoded_data = self.encoder_model.predict(X_scaled)
        
        # Application de KMeans sur les représentations encodées
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        return kmeans.fit_predict(encoded_data)
    
    def fit_agglomerative(self, X):
        """Entraîne un modèle de clustering agglomératif"""
        print("Entraînement du modèle de clustering agglomératif...")
        start_time = time.time()
        
        self.agglomerative_model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage='ward'
        )
        labels = self.agglomerative_model.fit_predict(X)
        
        end_time = time.time()
        print(f"Temps d'entraînement clustering agglomératif: {end_time - start_time:.2f} secondes")
        
        return labels
    
    def fit_hierarchical(self, X, sample_size=1000):
        """Entraîne un modèle de clustering hiérarchique"""
        print("Entraînement du modèle de clustering hiérarchique...")
        start_time = time.time()
        
        # Si les données sont trop volumineuses, on prend un échantillon
        if X.shape[0] > sample_size:
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
            
        # Calcul de la matrice de liaison
        self.hierarchical_linkage = linkage(X_sample, method='ward')
        
        # Prédiction des clusters
        labels = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # Trouver le point le plus proche dans l'échantillon
            if X.shape[0] > sample_size and i not in indices:
                distances = np.linalg.norm(X_sample - X[i].reshape(1, -1), axis=1)
                closest_idx = np.argmin(distances)
                labels[i] = labels[indices[closest_idx]]
            else:
                # Utiliser AgglomerativeClustering pour prédire les clusters
                if i < sample_size:
                    labels[i] = AgglomerativeClustering(
                        n_clusters=self.n_clusters, 
                        linkage='ward'
                    ).fit_predict(X_sample)[i]
        
        end_time = time.time()
        print(f"Temps d'entraînement clustering hiérarchique: {end_time - start_time:.2f} secondes")
        
        return labels
    
    def evaluate_clustering(self, X, labels):
        """Évalue la qualité du clustering"""
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        return {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin
        }
    
    def visualize_clusters_2d(self, X, labels, title="Visualisation des clusters"):
        """Visualise les clusters en 2D avec PCA"""
        # Réduction de dimension avec PCA
        X_pca = self.pca.fit_transform(X)
        
        # Création du graphique
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        plt.title(title)
        plt.xlabel('Composante principale 1')
        plt.ylabel('Composante principale 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        return plt
    
    def visualize_clusters_interactive(self, X, labels, title="Visualisation interactive des clusters"):
        """Visualise les clusters en 2D avec Plotly"""
        # Réduction de dimension avec PCA
        X_pca = self.pca.fit_transform(X)
        
        # Création du DataFrame pour Plotly
        df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': labels.astype(str)
        })
        
        # Création du graphique interactif
        fig = px.scatter(
            df, x='PC1', y='PC2', color='Cluster',
            title=title,
            labels={'PC1': 'Composante principale 1', 'PC2': 'Composante principale 2'},
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7), selector=dict(mode='markers'))
        fig.update_layout(
            plot_bgcolor='white',
            legend_title_text='Cluster',
            width=900,
            height=700
        )
        
        return fig
    
    def plot_dendrogram(self, title="Dendrogramme du clustering hiérarchique"):
        """Trace le dendrogramme du clustering hiérarchique"""
        if self.hierarchical_linkage is None:
            raise ValueError("Le modèle de clustering hiérarchique n'a pas été entraîné")
        
        plt.figure(figsize=(12, 8))
        dendrogram(
            self.hierarchical_linkage,
            truncate_mode='level',
            p=5,
            leaf_font_size=10,
            leaf_rotation=90,
            color_threshold=0.7 * max(self.hierarchical_linkage[:, 2])
        )
        plt.title(title)
        plt.xlabel('Échantillons')
        plt.ylabel('Distance')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        return plt
    
    def plot_autoencoder_loss(self, history, title="Évolution de la perte de l'autoencodeur"):
        """Trace l'évolution de la perte de l'autoencodeur"""
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Perte (entraînement)')
        plt.plot(history.history['val_loss'], label='Perte (validation)')
        plt.title(title)
        plt.xlabel('Époque')
        plt.ylabel('Perte (MSE)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        return plt
    
    def save_models(self, output_dir='models'):
        """Sauvegarde les modèles entraînés"""
        # Création du répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarde du modèle KMeans
        if self.kmeans_model is not None:
            joblib.dump(self.kmeans_model, os.path.join(output_dir, 'kmeans_model.pkl'))
        
        # Sauvegarde de l'autoencodeur
        if self.autoencoder_model is not None:
            self.autoencoder_model.save(os.path.join(output_dir, 'autoencoder_model'))
            self.encoder_model.save(os.path.join(output_dir, 'encoder_model'))
        
        # Sauvegarde du modèle de clustering agglomératif
        if self.agglomerative_model is not None:
            joblib.dump(self.agglomerative_model, os.path.join(output_dir, 'agglomerative_model.pkl'))
        
        # Sauvegarde de la matrice de liaison hiérarchique
        if self.hierarchical_linkage is not None:
            np.save(os.path.join(output_dir, 'hierarchical_linkage.npy'), self.hierarchical_linkage)
        
        # Sauvegarde du scaler et du PCA
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(self.pca, os.path.join(output_dir, 'pca.pkl'))
        
        print(f"Modèles sauvegardés dans le répertoire {output_dir}")
    
    def load_models(self, input_dir='models'):
        """Charge les modèles entraînés"""
        # Chargement du modèle KMeans
        kmeans_path = os.path.join(input_dir, 'kmeans_model.pkl')
        if os.path.exists(kmeans_path):
            self.kmeans_model = joblib.load(kmeans_path)
        
        # Chargement de l'autoencodeur
        autoencoder_path = os.path.join(input_dir, 'autoencoder_model')
        encoder_path = os.path.join(input_dir, 'encoder_model')
        if os.path.exists(autoencoder_path) and os.path.exists(encoder_path):
            self.autoencoder_model = tf.keras.models.load_model(autoencoder_path)
            self.encoder_model = tf.keras.models.load_model(encoder_path)
        
        # Chargement du modèle de clustering agglomératif
        agglomerative_path = os.path.join(input_dir, 'agglomerative_model.pkl')
        if os.path.exists(agglomerative_path):
            self.agglomerative_model = joblib.load(agglomerative_path)
        
        # Chargement de la matrice de liaison hiérarchique
        hierarchical_path = os.path.join(input_dir, 'hierarchical_linkage.npy')
        if os.path.exists(hierarchical_path):
            self.hierarchical_linkage = np.load(hierarchical_path)
        
        # Chargement du scaler et du PCA
        scaler_path = os.path.join(input_dir, 'scaler.pkl')
        pca_path = os.path.join(input_dir, 'pca.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        if os.path.exists(pca_path):
            self.pca = joblib.load(pca_path)
        
        print(f"Modèles chargés depuis le répertoire {input_dir}")

def find_optimal_k(X, k_range=range(2, 11)):
    """
    Trouve le nombre optimal de clusters en utilisant la méthode du coude
    et le score de silhouette
    """
    inertia_values = []
    silhouette_values = []
    
    for k in k_range:
        print(f"Essai avec k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertia_values.append(kmeans.inertia_)
        
        # Calcul du score de silhouette (seulement si k > 1)
        if k > 1:
            silhouette_values.append(silhouette_score(X, labels))
        else:
            silhouette_values.append(0)
    
    # Tracé de la méthode du coude
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia_values, 'o-', color='blue')
    plt.title('Méthode du coude')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Inertie')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_values, 'o-', color='green')
    plt.title('Score de silhouette')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Score de silhouette')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Détermination du k optimal
    # Pour la méthode du coude, on cherche le "coude" dans la courbe d'inertie
    k_optimal_inertia = k_range[np.argmax(np.diff(np.diff(inertia_values))) + 1]
    
    # Pour le score de silhouette, on cherche le k qui maximise le score
    k_optimal_silhouette = k_range[np.argmax(silhouette_values)]
    
    print(f"k optimal selon la méthode du coude: {k_optimal_inertia}")
    print(f"k optimal selon le score de silhouette: {k_optimal_silhouette}")
    
    return k_optimal_silhouette, plt
