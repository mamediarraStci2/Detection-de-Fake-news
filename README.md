# Système de Détection de Fake News

Ce projet implémente un système complet de détection de fake news utilisant à la fois des techniques supervisées (transformers) et non supervisées (clustering). Il permet d'analyser des articles en français et en anglais, et de les classifier en utilisant différents algorithmes.

## Algorithmes implémentés

### Approches supervisées (Transformers)

1. **DistilBERT** - Version allégée de BERT utilisée pour l'analyse des textes en anglais
2. **CamemBERT** - Modèle de type BERT spécifiquement pré-entraîné sur des textes en français

### Approches non supervisées (Clustering)

1. **KMeans** - Algorithme de partitionnement qui divise les données en K clusters distincts
2. **Deep Learning Non Supervisé (Autoencodeur + KMeans)** - Utilise un réseau de neurones pour apprendre une représentation compressée des données, puis applique KMeans sur ces représentations
3. **Clustering Agglomératif** - Algorithme hiérarchique ascendant qui fusionne progressivement les clusters
4. **Clustering Hiérarchique** - Algorithme qui construit une hiérarchie de clusters, visualisée par un dendrogramme

## Structure du Projet

```
Detection_de_fake_news/
│
├── Fake_news/                 # Dataset anglais
│   ├── train.csv
│   └── test (1).csv
│
├── French_Fake_News/          # Dataset français
│   ├── train.csv
│   └── test.csv
│
├── models/                    # Répertoire des modèles entraînés
│   ├── pytorch/                # Modèles PyTorch (Transformers)
│   └── results/                # Résultats d'évaluation
│
├── preprocess.py              # Module de prétraitement des données
├── clustering.py              # Implémentation des algorithmes de clustering
├── transformer_fake_news.py    # Implémentation des transformers avec TensorFlow
├── transformer_pytorch_version.py # Implémentation des transformers avec PyTorch
├── transformer_prediction.py   # Module de prédiction avec les transformers
├── compare_models.py           # Comparaison des performances entre modèles
├── comparative_analysis.py     # Analyse comparative des algorithmes de clustering
├── app.py                      # Interface utilisateur Streamlit
├── streamlit_app.py            # Point d'entrée pour le déploiement Streamlit Cloud
├── model_comparison_report.md  # Rapport de comparaison des performances
├── main.py                     # Script principal d'exécution
└── requirements.txt            # Dépendances du projet
```

## Pipeline de Réalisation

1. **Prétraitement des données**
   - Nettoyage des textes (suppression de la ponctuation, des chiffres, etc.)
   - Tokenisation et suppression des mots vides (stopwords)
   - Lemmatisation des textes
   - Détection automatique de la langue (français ou anglais)

2. **Extraction des caractéristiques**
   - Utilisation d'embeddings BERT multilingues pour représenter les textes
   - Modèles de tokenisation spécifiques à chaque langue (DistilBERT, CamemBERT)
   - Alternative: représentation TF-IDF pour les textes

3. **Modélisation**
   - **Approche supervisée**: Entraînement de modèles transformers (DistilBERT/CamemBERT)
   - **Approche non supervisée**: Application des algorithmes de clustering sur les représentations vectorielles
   - Détermination automatique du nombre optimal de clusters pour les approches non supervisées

4. **Évaluation et comparaison**
   - Calcul de métriques d'évaluation supervisée (accuracy, F1-score, precision, recall)
   - Calcul de métriques d'évaluation non supervisée (score de silhouette, indice de Davies-Bouldin)
   - Comparaison des performances entre les approches supervisées et non supervisées

## Déploiement

L'application est déployée sur Streamlit Cloud et accessible via le lien suivant :

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mamediarrastci2-detection-de-fake-news.streamlit.app/)

Pour exécuter l'application localement :

```bash
# Cloner le dépôt
git clone https://github.com/mamediarraStci2/Detection-de-Fake-news.git
cd Detection-de-Fake-news

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run streamlit_app.py
```
   - Visualisation des clusters et des résultats

5. **Interface utilisateur**
   - Application web Streamlit permettant de détecter les fake news en temps réel
   - Support multilingue (français et anglais)

## Installation et Utilisation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation

1. Clonez ce dépôt ou téléchargez les fichiers.
2. Installez les dépendances requises :

```bash
pip install -r requirements.txt
```

3. Téléchargez les modèles linguistiques spaCy :

```bash
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

### Exécution

1. **Prétraitement et analyse des données** :

```bash
python preprocess.py
```

2. **Analyse comparative des algorithmes** :

```bash
python comparative_analysis.py
```

3. **Lancement de l'interface utilisateur** :

```bash
streamlit run app.py
```

## Résultats Comparatifs

Les performances des algorithmes sont évaluées selon les métriques suivantes :
- **Score de silhouette** : Mesure la cohésion et la séparation des clusters (valeur entre -1 et 1, plus c'est élevé, mieux c'est)
- **Indice de Davies-Bouldin** : Mesure la séparation des clusters (valeur plus faible = meilleurs clusters)

Exemple de tableau comparatif :

| Algorithme | Score de silhouette (train) | Score de silhouette (test) | Indice Davies-Bouldin (train) | Indice Davies-Bouldin (test) |
|------------|------------------------------|----------------------------|--------------------------------|-------------------------------|
| KMeans | 0.65 | 0.62 | 0.85 | 0.88 |
| Autoencodeur + KMeans | 0.70 | 0.68 | 0.78 | 0.80 |
| Clustering agglomératif | 0.60 | 0.58 | 0.90 | 0.92 |
| Clustering hiérarchique | 0.63 | 0.61 | 0.87 | 0.89 |

## Interface Utilisateur

L'interface utilisateur permet de :
- Saisir un texte en français ou en anglais
- Choisir l'algorithme de clustering à utiliser pour la détection
- Obtenir une prédiction sur la nature du texte (fake news ou vraie news)
- Visualiser les probabilités d'appartenance aux différents clusters

## Auteurs

- Mame Diarra Bousso DIOP
- Mame Penda DIAW

## Licence

Ce projet est sous licence MIT.
