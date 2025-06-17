
"""
Module d'évaluation pour la comparaison des algorithmes de clustering
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class ClusteringEvaluator:
    """
    Classe pour l'évaluation et la comparaison des algorithmes de clustering
    """
    def __init__(self):
        self.results = {}
        self.metrics = ['silhouette_score', 'davies_bouldin_score']
        self.datasets = ['english', 'french']
        self.algorithms = ['kmeans', 'autoencoder', 'agglomerative', 'hierarchical']
        
    def add_result(self, algorithm, dataset, split, labels, data):
        """
        Ajoute un résultat d'évaluation
        
        Args:
            algorithm (str): Nom de l'algorithme
            dataset (str): Nom du dataset (english ou french)
            split (str): Type de split (train ou test)
            labels (array): Labels prédits
            data (array): Données utilisées pour l'évaluation
        """
        if algorithm not in self.results:
            self.results[algorithm] = {}
        
        if dataset not in self.results[algorithm]:
            self.results[algorithm][dataset] = {}
        
        # Calculer les métriques
        silhouette = silhouette_score(data, labels) if len(np.unique(labels)) > 1 else 0
        davies_bouldin = davies_bouldin_score(data, labels) if len(np.unique(labels)) > 1 else float('inf')
        
        self.results[algorithm][dataset][split] = {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'labels': labels
        }
    
    def get_comparison_table(self):
        """
        Génère un tableau comparatif des performances des algorithmes
        
        Returns:
            pd.DataFrame: Tableau comparatif
        """
        # Initialiser les colonnes du tableau
        columns = ['Algorithme', 'Dataset']
        for split in ['train', 'test']:
            for metric in self.metrics:
                columns.append(f"{metric} ({split})")
        
        # Initialiser les données du tableau
        data = []
        
        # Remplir les données
        for algorithm in self.results:
            for dataset in self.results[algorithm]:
                row = [algorithm, dataset]
                
                for split in ['train', 'test']:
                    if split in self.results[algorithm][dataset]:
                        for metric in self.metrics:
                            value = self.results[algorithm][dataset][split].get(metric, None)
                            row.append(round(value, 3) if value is not None else None)
                    else:
                        for metric in self.metrics:
                            row.append(None)
                
                data.append(row)
        
        # Créer le DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        return df
    
    def plot_comparison(self, metric='silhouette_score', output_dir='results'):
        """
        Trace un graphique comparatif des performances des algorithmes
        
        Args:
            metric (str): Métrique à comparer ('silhouette_score' ou 'davies_bouldin_score')
            output_dir (str): Répertoire de sortie pour les graphiques
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Créer un DataFrame pour le graphique
        data = []
        
        for algorithm in self.results:
            for dataset in self.results[algorithm]:
                for split in ['train', 'test']:
                    if split in self.results[algorithm][dataset]:
                        value = self.results[algorithm][dataset][split].get(metric, None)
                        if value is not None:
                            data.append({
                                'Algorithme': algorithm,
                                'Dataset': dataset,
                                'Split': split,
                                'Valeur': value
                            })
        
        df = pd.DataFrame(data)
        
        # Créer le graphique
        plt.figure(figsize=(12, 8))
        
        # Utiliser seaborn pour un graphique plus esthétique
        ax = sns.barplot(x='Algorithme', y='Valeur', hue='Split', data=df)
        
        # Ajouter les titres et les étiquettes
        metric_name = 'Score de silhouette' if metric == 'silhouette_score' else 'Indice de Davies-Bouldin'
        plt.title(f'Comparaison des algorithmes - {metric_name}')
        plt.xlabel('Algorithme')
        plt.ylabel(metric_name)
        
        # Ajouter une légende
        plt.legend(title='Split')
        
        # Ajouter une grille
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Sauvegarder le graphique
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'))
        
        # Créer un graphique interactif avec Plotly
        fig = px.bar(
            df, 
            x='Algorithme', 
            y='Valeur', 
            color='Split',
            barmode='group',
            facet_col='Dataset',
            title=f'Comparaison des algorithmes - {metric_name}',
            labels={'Valeur': metric_name}
        )
        
        # Améliorer le layout
        fig.update_layout(
            xaxis_title='Algorithme',
            legend_title='Split',
            height=600,
            width=1000
        )
        
        # Sauvegarder le graphique interactif
        fig.write_html(os.path.join(output_dir, f'comparison_{metric}_interactive.html'))
        
        return fig
    
    def plot_all_comparisons(self, output_dir='results'):
        """
        Trace tous les graphiques comparatifs
        
        Args:
            output_dir (str): Répertoire de sortie pour les graphiques
        """
        for metric in self.metrics:
            self.plot_comparison(metric, output_dir)
    
    def generate_report(self, output_dir='results'):
        """
        Génère un rapport complet d'évaluation
        
        Args:
            output_dir (str): Répertoire de sortie pour le rapport
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Générer le tableau comparatif
        df = self.get_comparison_table()
        
        # Sauvegarder le tableau en CSV
        df.to_csv(os.path.join(output_dir, 'comparison_table.csv'), index=False)
        
        # Générer les graphiques
        self.plot_all_comparisons(output_dir)
        
        # Créer un rapport HTML
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport d'évaluation des algorithmes de clustering</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                }
                h1, h2, h3 {
                    color: #333;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .container {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                }
                .chart {
                    width: 48%;
                    margin-bottom: 20px;
                }
                .chart img {
                    width: 100%;
                }
                .full-width {
                    width: 100%;
                }
            </style>
        </head>
        <body>
            <h1>Rapport d'évaluation des algorithmes de clustering</h1>
            
            <h2>Tableau comparatif</h2>
        """
        
        # Ajouter le tableau
        html_content += df.to_html(index=False)
        
        # Ajouter les graphiques
        html_content += """
            <h2>Graphiques comparatifs</h2>
            <div class="container">
                <div class="chart">
                    <h3>Score de silhouette</h3>
                    <img src="comparison_silhouette_score.png" alt="Comparaison des scores de silhouette">
                </div>
                <div class="chart">
                    <h3>Indice de Davies-Bouldin</h3>
                    <img src="comparison_davies_bouldin_score.png" alt="Comparaison des indices de Davies-Bouldin">
                </div>
            </div>
            
            <h2>Graphiques interactifs</h2>
            <p>Les graphiques interactifs sont disponibles dans les fichiers suivants :</p>
            <ul>
                <li><a href="comparison_silhouette_score_interactive.html">Comparaison des scores de silhouette (interactif)</a></li>
                <li><a href="comparison_davies_bouldin_score_interactive.html">Comparaison des indices de Davies-Bouldin (interactif)</a></li>
            </ul>
            
            <h2>Conclusion</h2>
            <p>
                Cette étude comparative a permis d'évaluer les performances de différents algorithmes de clustering
                pour la détection de fake news. Les métriques utilisées sont le score de silhouette (plus il est élevé, meilleur est le clustering)
                et l'indice de Davies-Bouldin (plus il est bas, meilleur est le clustering).
            </p>
        </body>
        </html>
        """
        
        # Sauvegarder le rapport HTML
        with open(os.path.join(output_dir, 'evaluation_report.html'), 'w') as f:
            f.write(html_content)
        
        print(f"Rapport d'évaluation généré dans le répertoire {output_dir}")
