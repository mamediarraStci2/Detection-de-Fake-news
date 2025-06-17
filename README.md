# Système de Détection de Fausses Nouvelles (Fake News)

Ce projet propose un système complet pour la détection de fausses nouvelles en s'appuyant sur des articles en français et en anglais. Il met en œuvre et compare plusieurs approches de Machine Learning, incluant des méthodes non supervisées (clustering) et des modèles de pointe supervisés (Transformers). L'objectif est d'identifier automatiquement la nature d'un article (information véridique ou fausse nouvelle) et de comparer l'efficacité des différentes techniques.

## 1. Datasets

Le projet utilise deux corpus de textes distincts :

- **Dataset Anglais** : Situé dans le répertoire `Fake_news/`, il contient des articles de presse en anglais.
- **Dataset Français** : Situé dans le répertoire `French_Fake_News/`, il est composé d'articles en français.

Chaque dataset est divisé en un ensemble d'entraînement (`train.csv`) et un ensemble de test (`test.csv` ou `test (1).csv`). La taille exacte des datasets (nombre d'articles) est calculée et affichée lors de l'exécution des scripts de prétraitement et d'entraînement (par exemple, `transformer_fake_news.py`).

| Dataset  | Fichier d'entraînement | Fichier de test      | Langue  |
|----------|------------------------|----------------------|---------|
| Anglais  | `Fake_news/train.csv`  | `Fake_news/test (1).csv` | Anglais |
| Français | `French_Fake_News/train.csv` | `French_Fake_News/test.csv` | Français|

## 2. Pipeline d'Exécution

Le projet est structuré autour d'un pipeline d'exécution clair, orchestré principalement par le script `run_all.py`.

**Étape 1 : Prétraitement des Données (`preprocess.py`)**
- **Nettoyage du texte** : Suppression des URLs, des caractères spéciaux, des chiffres et conversion en minuscules.
- **Tokenisation** : Segmentation du texte en mots (tokens).
- **Suppression des mots vides (Stopwords)** : Élimination des mots courants (ex: "le", "the", "et", "and") qui n'apportent pas de sens significatif.
- **Lemmatisation** : Réduction des mots à leur forme de base (lemme) pour standardiser le vocabulaire (ex: "running" -> "run").

**Étape 2 : Extraction des Caractéristiques (Features)**
Deux approches sont utilisées pour convertir le texte en vecteurs numériques :
- **TF-IDF (`get_tfidf_features`)** : Représentation statistique qui évalue l'importance d'un mot dans un document par rapport à l'ensemble du corpus.
- **Embeddings BERT (`get_bert_embeddings`)** : Représentations vectorielles denses et contextuelles issues de modèles Transformers pré-entraînés, capturant des relations sémantiques complexes.

**Étape 3 : Modélisation et Entraînement**
Le projet explore deux paradigmes de modélisation :

- **Approche Non Supervisée (`clustering.py`, `comparative_analysis.py`)** :
  - Entraînement des modèles de clustering sur les vecteurs de texte.
  - L'objectif est de regrouper les articles similaires sans utiliser d'étiquettes préexistantes.

- **Approche Supervisée (`transformer_fake_news.py`)** :
  - Entraînement de modèles Transformers sur les données étiquetées (vraie/fausse nouvelle).
  - Le modèle apprend à classifier les articles en se basant sur les exemples fournis.

**Étape 4 : Évaluation et Comparaison**
- Les performances des modèles sont rigoureusement évaluées à l'aide de métriques adaptées à chaque approche.
- Les temps d'entraînement pour chaque algorithme sont mesurés et rapportés pour comparer leur efficacité computationnelle.
- Des rapports et des visualisations sont générés dans le dossier `results/`.

**Étape 5 : Interface Utilisateur (`app.py`)**
- Une application web interactive développée avec Streamlit permet de tester les modèles en temps réel.
- L'utilisateur peut soumettre un texte et obtenir une prédiction sur sa nature.

## 3. Approches Utilisées

### Approche Non Supervisée (Clustering)

Cette approche vise à découvrir des structures naturelles dans les données sans connaître les étiquettes.

- **Algorithmes** :
  1. **KMeans** : Partitionne les données en *k* clusters en minimisant l'inertie.
  2. **Autoencodeur + KMeans** : Un autoencodeur (réseau de neurones) réduit la dimensionnalité des données, puis KMeans est appliqué sur cette représentation compressée (embedding).
  3. **Clustering Agglomératif** : Construit une hiérarchie de clusters en fusionnant itérativement les paires de clusters les plus proches.
  4. **Clustering Hiérarchique** : Similaire au clustering agglomératif, il permet de visualiser la hiérarchie des clusters via un dendrogramme.

- **Évaluation** :
  - **Score de Silhouette** : Mesure à quel point un article est similaire à son propre cluster par rapport aux autres clusters. Une valeur proche de 1 indique des clusters denses et bien séparés.
  - **Indice de Davies-Bouldin** : Mesure le rapport entre la dispersion intra-cluster et la séparation inter-cluster. Une valeur plus faible indique une meilleure partition.

### Approche Supervisée (Transformers)

Cette approche utilise des modèles pré-entraînés sur d'immenses corpus de texte, affinés (fine-tuned) pour notre tâche de classification.

- **Modèles** :
  1. **BERT (bert-base-multilingual-cased)** : Modèle de référence pour de nombreuses tâches NLP.
  2. **DistilBERT (distilbert-base-multilingual-cased)** : Une version plus légère et rapide de BERT, idéale pour des déploiements plus rapides.
  3. **CamemBERT (camembert-base)** : Un modèle spécialisé pour le français, offrant d'excellentes performances sur les textes dans cette langue.

- **Évaluation** :
  - **Accuracy** : Pourcentage de prédictions correctes.
  - **Precision** : Capacité du modèle à ne pas étiqueter une vraie nouvelle comme fausse.
  - **Recall** : Capacité du modèle à identifier toutes les fausses nouvelles.
  - **F1-Score** : Moyenne harmonique de la précision et du rappel.
  - **Matrice de Confusion** : Tableau qui visualise les performances du classifieur.

## 4. Comparaison des Résultats

Les scripts génèrent des tableaux comparatifs et des graphiques pour analyser les performances.

### Résultats du Clustering (Exemple)

| Algorithme                | Temps d'entraînement (s) | Score Silhouette (test) | Indice Davies-Bouldin (test) |
|---------------------------|--------------------------|-------------------------|------------------------------|
| KMeans                    | *~5s*                    | *0.05*                  | *2.5*                        |
| Autoencodeur + KMeans     | *~120s*                  | *0.12*                  | *1.8*                        |
| Clustering Agglomératif   | *~300s*                  | *0.04*                  | *2.8*                        |
| Clustering Hiérarchique   | *~300s*                  | *0.04*                  | *2.8*                        |
*Note : Les valeurs sont des exemples et dépendent des données et des hyperparamètres.*

### Résultats des Transformers (Exemple)

| Modèle      | Langue   | Temps d'entraînement (s) | Accuracy (test) | F1-Score (test) |
|-------------|----------|--------------------------|-----------------|-----------------|
| DistilBERT  | Anglais  | *~600s*                  | *0.95*          | *0.95*          |
| CamemBERT   | Français | *~600s*                  | *0.98*          | *0.98*          |
*Note : Les valeurs sont des exemples et dépendent des données et des hyperparamètres.*

## 5. Comparaison : Clustering Agglomératif vs. Transformers

Ces deux approches, bien que visant un objectif similaire, sont fondamentalement différentes.

| Caractéristique             | Clustering Agglomératif                                   | Modèles Transformers (BERT, etc.)                           |
|-----------------------------|-----------------------------------------------------------|-------------------------------------------------------------|
| **Paradigme**               | **Non Supervisé**                                         | **Supervisé**                                               |
| **Besoin en données**       | Pas besoin d'étiquettes. Idéal pour l'exploration.        | Nécessite un grand volume de données étiquetées.            |
| **Objectif**                | Regrouper les données par similarité.                     | Apprendre une fonction de classification.                   |
| **Résultat**                | Des clusters de documents.                                | Une prédiction de classe pour chaque document.              |
| **Complexité (temps)**      | Élevée (O(n² log n)), difficile à scaler.                 | Entraînement long, mais prédiction rapide.                  |
| **Performance**             | Utile pour l'analyse exploratoire, mais moins précis pour la classification. | État de l'art pour la classification de texte. Très précis. |
| **Interprétabilité**        | Le dendrogramme aide à visualiser la hiérarchie.          | Plus complexe à interpréter ("boîte noire").                |

En résumé, le **clustering agglomératif** est excellent pour découvrir des thèmes ou des groupes dans des données non étiquetées, tandis que les **Transformers** sont supérieurs pour les tâches de classification pure lorsque des données d'entraînement de qualité sont disponibles.

## 6. Visualisations Générées

Les scripts sauvegardent plusieurs figures dans le dossier `results/` pour aider à l'interprétation :

- **Visualisation des clusters** : Projection 2D (via PCA) des clusters pour observer leur séparation.
  `![Cluster Visualization](results/kmeans_clusters.png)`
- **Dendrogramme** : Arbre hiérarchique montrant comment les clusters sont fusionnés.
  `![Dendrogramme](results/hierarchical_dendrogram.png)`
- **Perte de l'Autoencodeur** : Courbe montrant l'évolution de l'erreur de reconstruction pendant l'entraînement.
  `![Autoencoder Loss](results/autoencoder_loss.png)`
- **Matrice de Confusion** : Visualisation des performances des modèles Transformers.
  `![Confusion Matrix](results/bert_english_confusion_matrix.png)`

## 7. Installation et Utilisation

### Prérequis
- Python 3.8+
- Git

### Installation
```bash
# 1. Cloner le dépôt
git clone https://github.com/mamediarraStci2/Detection-de-Fake-news.git
cd Detection-de-Fake-news

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Télécharger les modèles linguistiques spaCy
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

### Exécution du Pipeline Complet
Le script `run_all.py` exécute les étapes principales :
```bash
python run_all.py
```
Ce script lancera le prétraitement, l'analyse comparative, et enfin l'application Streamlit.

### Lancement de l'Interface
Pour lancer uniquement l'application web :
```bash
streamlit run app.py
```

## 8. Structure du Projet
```
Detection_de_fake_news/
│
├── Fake_news/                 # Dataset anglais
├── French_Fake_News/          # Dataset français
├── models/                    # Modèles entraînés
├── results/                   # Résultats, rapports et figures
├── .gitignore
├── app.py                     # Interface utilisateur Streamlit
├── clustering.py              # Algorithmes de clustering
├── comparative_analysis.py    # Script d'analyse comparative (clustering)
├── main.py                    # Script principal pour le clustering
├── preprocess.py              # Prétraitement des données
├── requirements.txt
├── run_all.py                 # Script pour exécuter tout le pipeline
├── transformer_fake_news.py   # Entraînement des Transformers
└── ... (autres scripts)
```
