"""
Script pour l'entraînement des modèles de détection de fake news
"""
import pandas as pd
import numpy as np
import os
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Chemins des datasets
ENGLISH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Fake_news')
FRENCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'French_Fake_News')

# Répertoire de sortie pour les modèles entraînés
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Listes de mots clés pour la détection de fake news
FAKE_NEWS_WORDS = [
    'shocking', 'secret', 'they don\'t want you to know', 'conspiracy', 'hoax', 'scam', 
    'fake', 'miracle', 'shocking truth', 'government lies', 'exposed',
    'scandal', 'breaking', 'bombshell', 'urgent', 'alarming', 'censored', 'banned',
    'choquant', 'secret', 'complot', 'canular', 'arnaque', 'faux', 'miracle', 
    'vérité choquante', 'mensonges', 'complot', 'exposé',
    'scandale', 'urgent', 'alarmant', 'censuré', 'interdit'
]

REAL_NEWS_WORDS = [
    'report', 'according to', 'study finds', 'research', 'announced', 'published',
    'confirmed', 'stated', 'evidence', 'data', 'source', 'official', 'expert',
    'analysis', 'investigation', 'rapport', 'selon', 'étude', 'recherche', 'annoncé',
    'publié', 'confirmé', 'déclaré', 'preuve', 'données', 'source', 'officiel',
    'expert', 'analyse', 'enquête'
]

def clean_text(text):
    """Nettoie le texte"""
    if not isinstance(text, str):
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Suppression des caractères spéciaux et des chiffres
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_keyword_features(text):
    """Calcule des caractéristiques basées sur les mots-clés"""
    fake_words_count = sum(1 for word in FAKE_NEWS_WORDS if word.lower() in text.lower())
    real_words_count = sum(1 for word in REAL_NEWS_WORDS if word.lower() in text.lower())
    
    total_words = len(text.split())
    fake_ratio = fake_words_count / max(1, total_words)
    real_ratio = real_words_count / max(1, total_words)
    
    return fake_ratio, real_ratio

def load_and_preprocess_english_data():
    """Charge et prétraite les données anglaises"""
    print("\nChargement des données anglaises...")
    
    # Charger les données
    train_path = os.path.join(ENGLISH_PATH, 'train.csv')
    test_path = os.path.join(ENGLISH_PATH, 'test (1).csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Erreur: Fichiers de données anglaises non trouvés dans {ENGLISH_PATH}")
        return None, None
    
    train_data = pd.read_csv(train_path, delimiter=';')
    test_data = pd.read_csv(test_path, delimiter=';')
    
    # Identifier les colonnes
    text_col = 'text' if 'text' in train_data.columns else train_data.columns[2]
    title_col = 'title' if 'title' in train_data.columns else train_data.columns[1]
    
    # Prétraiter les données
    print("Prétraitement des données anglaises...")
    
    # Combiner titre et texte pour une meilleure analyse
    train_data['combined_text'] = train_data[title_col].fillna('') + ' ' + train_data[text_col].fillna('')
    test_data['combined_text'] = test_data[title_col].fillna('') + ' ' + test_data[text_col].fillna('')
    
    # Nettoyer les textes
    train_data['clean_text'] = train_data['combined_text'].apply(clean_text)
    test_data['clean_text'] = test_data['combined_text'].apply(clean_text)
    
    # Ajouter une colonne pour la langue
    train_data['language'] = 'en'
    test_data['language'] = 'en'
    
    # Ajouter des caractéristiques basées sur les mots-clés
    train_fake_ratio, train_real_ratio = zip(*train_data['clean_text'].apply(calculate_keyword_features))
    train_data['fake_ratio'] = train_fake_ratio
    train_data['real_ratio'] = train_real_ratio
    
    test_fake_ratio, test_real_ratio = zip(*test_data['clean_text'].apply(calculate_keyword_features))
    test_data['fake_ratio'] = test_fake_ratio
    test_data['real_ratio'] = test_real_ratio
    
    return train_data, test_data

def load_and_preprocess_french_data():
    """Charge et prétraite les données françaises"""
    print("\nChargement des données françaises...")
    
    # Charger les données
    train_path = os.path.join(FRENCH_PATH, 'train.csv')
    test_path = os.path.join(FRENCH_PATH, 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Erreur: Fichiers de données françaises non trouvés dans {FRENCH_PATH}")
        return None, None
    
    train_data = pd.read_csv(train_path, delimiter=';')
    test_data = pd.read_csv(test_path, delimiter=';')
    
    # Identifier les colonnes
    text_col = 'post' if 'post' in train_data.columns else train_data.columns[1]
    
    # Prétraiter les données
    print("Prétraitement des données françaises...")
    
    # Nettoyer les textes
    train_data['clean_text'] = train_data[text_col].apply(clean_text)
    test_data['clean_text'] = test_data[text_col].apply(clean_text)
    
    # Ajouter une colonne pour la langue
    train_data['language'] = 'fr'
    test_data['language'] = 'fr'
    
    # Ajouter des caractéristiques basées sur les mots-clés
    train_fake_ratio, train_real_ratio = zip(*train_data['clean_text'].apply(calculate_keyword_features))
    train_data['fake_ratio'] = train_fake_ratio
    train_data['real_ratio'] = train_real_ratio
    
    test_fake_ratio, test_real_ratio = zip(*test_data['clean_text'].apply(calculate_keyword_features))
    test_data['fake_ratio'] = test_fake_ratio
    test_data['real_ratio'] = test_real_ratio
    
    return train_data, test_data

def extract_features(train_data, test_data, max_features=2000):
    """Extrait les caractéristiques TF-IDF et des méta-caractéristiques"""
    print("\nExtraction des caractéristiques...")
    
    # TF-IDF sur les textes nettoyés - utiliser moins de features
    print(f"Utilisation de {max_features} caractéristiques max pour TF-IDF")
    vectorizer = TfidfVectorizer(max_features=max_features)
    train_tfidf = vectorizer.fit_transform(train_data['clean_text'])
    test_tfidf = vectorizer.transform(test_data['clean_text'])
    
    # Extraire les méta-caractéristiques
    meta_features_train = np.column_stack((
        train_data['fake_ratio'].values,
        train_data['real_ratio'].values
    ))
    
    meta_features_test = np.column_stack((
        test_data['fake_ratio'].values,
        test_data['real_ratio'].values
    ))
    
    # Normaliser les méta-caractéristiques séparément
    meta_scaler = StandardScaler()
    meta_features_train_scaled = meta_scaler.fit_transform(meta_features_train)
    meta_features_test_scaled = meta_scaler.transform(meta_features_test)
    
    # Nous n'avons pas besoin d'un scaler pour les features TF-IDF car elles sont déjà normalisées
    
    return train_tfidf, test_tfidf, meta_features_train_scaled, meta_features_test_scaled, vectorizer, meta_scaler

def train_kmeans(train_tfidf, test_tfidf, meta_features_train, meta_features_test, n_clusters=2):
    """Entraîne un modèle KMeans sur des matrices creuses"""
    print(f"\nEntraînement du modèle KMeans avec {n_clusters} clusters...")
    start_time = time.time()
    
    # MiniBatchKMeans est plus économe en mémoire pour les grands datasets
    from sklearn.cluster import MiniBatchKMeans
    
    # Échantillonnage pour accélérer l'entraînement si nécessaire
    sample_size = min(10000, train_tfidf.shape[0])
    print(f"Utilisation d'un échantillon de {sample_size} documents pour l'entraînement initial")
    
    # Sélectionner un échantillon aléatoire pour l'initialisation
    from sklearn.utils import resample
    indices = resample(np.arange(train_tfidf.shape[0]), n_samples=sample_size, random_state=42)
    train_tfidf_sample = train_tfidf[indices]
    meta_features_train_sample = meta_features_train[indices]
    
    # Extraction des caractéristiques pour l'échantillon dans la mémoire
    # Pour l'échantillon, nous pouvons nous permettre de convertir en dense
    train_tfidf_sample_array = train_tfidf_sample.toarray()
    
    # Ajouter les méta-caractéristiques à l'échantillon dense
    from scipy.sparse import hstack, csr_matrix
    
    # Initialiser le modèle avec l'échantillon
    print("Initialisation du modèle avec l'échantillon...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3, batch_size=1000)
    
    # Pour l'initialisation, nous utilisons l'échantillon converti en dense avec les méta-caractéristiques
    init_features = np.hstack((train_tfidf_sample_array, meta_features_train_sample))
    kmeans.fit(init_features)
    
    # Ensuite, continuer l'entraînement sur l'ensemble complet par lots
    print("Continuation de l'entraînement sur l'ensemble complet par lots...")
    batch_size = 1000
    n_batches = (train_tfidf.shape[0] + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, train_tfidf.shape[0])
        
        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(f"Traitement du lot {i+1}/{n_batches}")
        
        # Extraire le lot
        batch_tfidf = train_tfidf[start_idx:end_idx].toarray()
        batch_meta = meta_features_train[start_idx:end_idx]
        
        # Combiner TF-IDF et méta-caractéristiques pour ce lot
        batch_features = np.hstack((batch_tfidf, batch_meta))
        
        # Mettre à jour le modèle avec ce lot
        kmeans.partial_fit(batch_features)
    
    # Prédictions - également par lots pour économiser la mémoire
    print("Génération des prédictions par lots...")
    train_labels = np.zeros(train_tfidf.shape[0], dtype=int)
    test_labels = np.zeros(test_tfidf.shape[0], dtype=int)
    
    # Prédictions sur l'ensemble d'entraînement
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, train_tfidf.shape[0])
        
        batch_tfidf = train_tfidf[start_idx:end_idx].toarray()
        batch_meta = meta_features_train[start_idx:end_idx]
        batch_features = np.hstack((batch_tfidf, batch_meta))
        
        train_labels[start_idx:end_idx] = kmeans.predict(batch_features)
    
    # Prédictions sur l'ensemble de test
    n_test_batches = (test_tfidf.shape[0] + batch_size - 1) // batch_size
    for i in range(n_test_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, test_tfidf.shape[0])
        
        batch_tfidf = test_tfidf[start_idx:end_idx].toarray()
        batch_meta = meta_features_test[start_idx:end_idx]
        batch_features = np.hstack((batch_tfidf, batch_meta))
        
        test_labels[start_idx:end_idx] = kmeans.predict(batch_features)
    
    # Évaluation - calcul par lots pour les scores
    print("Évaluation du modèle...")
    from sklearn.metrics import silhouette_samples
    
    # Calcul du score de silhouette par échantillonnage
    sample_size_eval = min(5000, train_tfidf.shape[0])
    eval_indices = resample(np.arange(train_tfidf.shape[0]), n_samples=sample_size_eval, random_state=43)
    
    train_tfidf_eval = train_tfidf[eval_indices].toarray()
    train_meta_eval = meta_features_train[eval_indices]
    train_features_eval = np.hstack((train_tfidf_eval, train_meta_eval))
    train_labels_eval = train_labels[eval_indices]
    
    silhouette_train = silhouette_score(train_features_eval, train_labels_eval)
    davies_bouldin_train = davies_bouldin_score(train_features_eval, train_labels_eval)
    
    # Même chose pour les données de test
    sample_size_test = min(5000, test_tfidf.shape[0])
    eval_indices_test = resample(np.arange(test_tfidf.shape[0]), n_samples=sample_size_test, random_state=44)
    
    test_tfidf_eval = test_tfidf[eval_indices_test].toarray()
    test_meta_eval = meta_features_test[eval_indices_test]
    test_features_eval = np.hstack((test_tfidf_eval, test_meta_eval))
    test_labels_eval = test_labels[eval_indices_test]
    
    silhouette_test = silhouette_score(test_features_eval, test_labels_eval)
    davies_bouldin_test = davies_bouldin_score(test_features_eval, test_labels_eval)
    
    end_time = time.time()
    
    print(f"Entraînement terminé en {end_time - start_time:.2f} secondes")
    print(f"Score de silhouette (train): {silhouette_train:.4f}")
    print(f"Score de silhouette (test): {silhouette_test:.4f}")
    print(f"Indice de Davies-Bouldin (train): {davies_bouldin_train:.4f}")
    print(f"Indice de Davies-Bouldin (test): {davies_bouldin_test:.4f}")
    
    # Déterminer quel cluster correspond aux fake news
    # Nous devons faire cette analyse par lots également
    print("Analyse des clusters pour identification des fake news...")
    fake_ratio_sum = [0, 0]
    count = [0, 0]
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, train_tfidf.shape[0])
        batch_labels = train_labels[start_idx:end_idx]
        batch_meta = meta_features_train[start_idx:end_idx]
        
        # La première colonne est le fake_ratio
        for cluster in [0, 1]:
            mask = (batch_labels == cluster)
            if np.any(mask):
                fake_ratio_sum[cluster] += np.sum(batch_meta[mask, 0])
                count[cluster] += np.sum(mask)
    
    avg_fake_ratio_cluster_0 = fake_ratio_sum[0] / max(count[0], 1)
    avg_fake_ratio_cluster_1 = fake_ratio_sum[1] / max(count[1], 1)
    
    fake_cluster = 0 if avg_fake_ratio_cluster_0 > avg_fake_ratio_cluster_1 else 1
    print(f"Cluster identifié pour les fake news: {fake_cluster}")
    print(f"Ratio moyen de mots-clés fake dans cluster 0: {avg_fake_ratio_cluster_0:.4f}")
    print(f"Ratio moyen de mots-clés fake dans cluster 1: {avg_fake_ratio_cluster_1:.4f}")
    
    # Ajouter cette information au modèle
    kmeans.fake_cluster = fake_cluster
    
    return kmeans, silhouette_test, davies_bouldin_test, train_labels, test_labels

def visualize_clusters(tfidf_matrix, meta_features, labels, title="Visualisation des clusters"):
    """Visualise les clusters en 2D en utilisant un échantillon pour économiser la mémoire"""
    print(f"\nCréation de la visualisation: {title}...")
    
    # Prendre un échantillon pour la visualisation
    max_sample = 2000
    if tfidf_matrix.shape[0] > max_sample:
        print(f"Échantillonnage de {max_sample} points pour la visualisation")
        from sklearn.utils import resample
        indices = resample(np.arange(tfidf_matrix.shape[0]), n_samples=max_sample, random_state=42)
        tfidf_sample = tfidf_matrix[indices].toarray()
        meta_sample = meta_features[indices]
        labels_sample = labels[indices]
    else:
        tfidf_sample = tfidf_matrix.toarray()
        meta_sample = meta_features
        labels_sample = labels
    
    # Combiner les features pour la visualisation
    features_sample = np.hstack((tfidf_sample, meta_sample))
    
    # Réduction de dimension pour la visualisation
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=2, random_state=42)
    features_2d = svd.fit_transform(features_sample)
    
    # Création du graphique
    plt.figure(figsize=(10, 8))
    colors = ['#3498db', '#e74c3c']  # Bleu pour vraies news, Rouge pour fake news
    
    # Déterminer l'ordre des couleurs selon le fake_cluster
    try:
        # Si on a accès à kmeans.fake_cluster
        if 'kmeans' in globals() and hasattr(kmeans, 'fake_cluster'):
            if kmeans.fake_cluster == 1:
                colors = colors[::-1]  # Inverser les couleurs
    except:
        pass
    
    # Créer un scatterplot pour chaque cluster avec sa propre couleur
    for i, color in enumerate(colors):
        mask = (labels_sample == i)
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=color, label=f'Cluster {i}', alpha=0.6)
    
    plt.title(title)
    plt.xlabel('Composante 1')
    plt.ylabel('Composante 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Sauvegarder la figure
    output_path = os.path.join(MODELS_DIR, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(output_path)
    print(f"Visualisation sauvegardée dans {output_path}")
    
    plt.close()

def train_and_save_models():
    """Entraîne et sauvegarde les modèles avec une approche économe en mémoire"""
    # Créer le répertoire des modèles s'il n'existe pas
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Charger et prétraiter les données
    en_train, en_test = load_and_preprocess_english_data()
    fr_train, fr_test = load_and_preprocess_french_data()
    
    if en_train is None or fr_train is None:
        print("Erreur lors du chargement des données. Vérifiez les chemins des fichiers.")
        return
    
    # Combiner les données pour un modèle multilingue
    print("\nCréation d'un modèle multilingue...")
    train_data = pd.concat([en_train, fr_train], ignore_index=True)
    test_data = pd.concat([en_test, fr_test], ignore_index=True)
    
    # Extraire les caractéristiques avec l'approche optimisée
    train_tfidf, test_tfidf, meta_train, meta_test, vectorizer, meta_scaler = extract_features(train_data, test_data, max_features=2000)
    
    # Entraîner et évaluer le modèle KMeans avec l'approche par lots
    kmeans_model, silhouette_score_val, davies_bouldin_score_val, train_labels, test_labels = train_kmeans(
        train_tfidf, test_tfidf, meta_train, meta_test)
    
    # Visualiser les clusters avec l'approche économe en mémoire
    visualize_clusters(train_tfidf, meta_train, train_labels, "Clusters sur données d'entraînement")
    visualize_clusters(test_tfidf, meta_test, test_labels, "Clusters sur données de test")
    
    # Sauvegarder les modèles et transformateurs
    print("\nSauvegarde des modèles...")
    joblib.dump(kmeans_model, os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(meta_scaler, os.path.join(MODELS_DIR, 'meta_scaler.pkl'))
    
    # Sauvegarder les performances et paramètres
    performance = {
        'silhouette_score': silhouette_score_val,
        'davies_bouldin_score': davies_bouldin_score_val,
        'fake_cluster': kmeans_model.fake_cluster,
        'tfidf_max_features': vectorizer.max_features,
        'n_clusters': kmeans_model.n_clusters
    }
    
    with open(os.path.join(MODELS_DIR, 'performance.txt'), 'w') as f:
        for metric, value in performance.items():
            f.write(f"{metric}: {value}\n")
    
    # Sauvegarder aussi les mots-clés pour qu'ils soient disponibles lors du chargement
    keywords = {
        'fake_news_words': FAKE_NEWS_WORDS,
        'real_news_words': REAL_NEWS_WORDS
    }
    joblib.dump(keywords, os.path.join(MODELS_DIR, 'keywords.pkl'))
    
    print(f"\nEntraînement terminé avec succès! Modèles sauvegardés dans le répertoire '{MODELS_DIR}'")
    print(f"Score de silhouette: {silhouette_score_val:.4f}")
    print(f"Indice de Davies-Bouldin: {davies_bouldin_score_val:.4f}")
    print(f"Cluster identifié pour les fake news: {kmeans_model.fake_cluster}")
    
    return kmeans_model, vectorizer, meta_scaler

if __name__ == "__main__":
    print("="*80)
    print("ENTRAÎNEMENT DES MODÈLES DE DÉTECTION DE FAKE NEWS")
    print("="*80)
    
    train_and_save_models()
