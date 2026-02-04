
# Projet de tarification auto en python avec comparaison ML et implémentation de Fairness

Ce projet propose une approche moderne de la modélisation de la prime pure en utilisant le dataset freMTPL2, l'objectif de ce projet est de comparer les outils classiques en actuariat (GLM Poisson/ Gamma) aux outils de machine learning actuels et de regarder les enjeux éthiques de ces modèles performants.

## Explication

La pipeline de mon projet se décompose en une première phase de préparation des données à travers l'EDA de la Fréquence puis de la Sévérité, une seconde phase de comparaison de modèles et d'entraînement de modèles, le choix du modèle fréquence final et du modèle sévérité finazl se basera sur différentes métriques telles que le Mean Square Error et la déviance Poisson pour la modélisation de la fréquence et le MAE et la déviance Gamma pour la modélisation de la sévérité , et pour finir une dernière phase de vérification éthique sur les modèles gagnants du comparatif.

Le projet est structuré de manière séquentielle, de l'acquisition des données à l'audit éthique final :

![Structure de la pipeline](https://github.com/user-attachments/assets/d1d4cdd8-ec5a-4fed-b77d-c527df19efa0)

Tout d'abord, voici ce que représente nos données:

![Structure de la pipeline](https://github.com/user-attachments/assets/a2c009e1-d1f3-41f0-92b8-e689a30ed450)

Pour commencer, j'ai supprimer toutes les valeurs qui sont aberrantes, c'est des données qui sont très certainement des erreurs de saisies et doivent être supprimés, je ne les ai pas normalisé ou windsorisé car dans ce dataset, très peyu de valeurs sont aberrantes, les supprimer ne nuit donc pas à nos modèles.

Par la suite, j'ai split mes données en train/test, je les ai splitté aussi tôt dans le projet pour éviter au maximum les fuites de données, c'est pour cette raison que mon analyse exploratoire de données se portent uniquement sur le dataframe train,  

Pour estimer la prime pure, on découpe le projet en 2 parties, une première qui a pour objectif d'estimer la fréquence qu'un client rencontre un accident et un second pour calculer le montant de la réclamation si il y en a une, 

Les données du dataset freMTPL2 présentent de nombreux challenges, premièrement, il est très difficile d'estimer sur ce dataset `ClaimAmount` car on dispose de très peu d'informations susceptibles d'obtenir une bonne estimation de cette variable,














































![Structure de la pipeline](https://github.com/user-attachments/assets/a0dce2db-2c1a-4f90-a57a-d9e782c22417)

Gemini:

🚗 Projet de Tarification Assurance AutoCe projet a pour objectif de modéliser la Prime Pure ($Prime\ Pure = Fréquence \times Sévérité$) en utilisant les jeux de données de référence en actuariat : freMTPL2freq (fréquence des sinistres) et freMTPL2sev (coût moyen des sinistres).


🎯 Objectifs

Le projet s'articule autour de trois axes principaux :

Performance : Comparer la précision des modèles d'ensemble (XGBoost, LightGBM) par rapport aux modèles linéaires généralisés (GLM).
Interprétabilité : Analyser le compromis entre la puissance prédictive des modèles "boîte noire" et la transparence nécessaire en assurance.
Équité : Auditer le modèle final pour détecter d'éventuels biais discriminatoires envers certaines catégories d'assurés (audit via Fairlearn).

🛠️ Méthodologie

L'approche technique est divisée en trois phases :Data Engineering : Nettoyage, traitement des valeurs aberrantes (ex: expositions négatives, montants extrêmes) et fusion des données fréquence/sévérité.Modélisation :Baseline (GLM) : Utilisation de lois de Poisson (Fréquence) et Gamma (Coût moyen).Machine Learning : Implémentation de Random Forest, XGBoost et LightGBM.Évaluation : Mesure de la performance via le RMSE et la déviance de Poisson.



📊 Structure du NotebookChargement des données : Importation des bibliothèques (scikit-learn, xgboost, lightgbm, fairlearn, shap) et des datasets.Préparation & Feature Engineering : Création de variables synthétiques et gestion des données brutes.Exploration de Données (EDA) : Analyse statistique univariée et détection des valeurs atypiques.Entraînement des Modèles : (Détail de la construction des pipelines de transformation et des modèles de régression).Audit d'Équité : Analyse de l'impact disparate selon l'âge du conducteur ou d'autres variables sensibles.📦 Bibliothèques Utiliséespandas, numpy : Manipulation des données.matplotlib, seaborn : Visualisation.statsmodels, scikit-learn : Modélisation statistique et ML.xgboost, lightgbm : Modèles de gradient boosting.shap : Interprétabilité locale et globale.fairlearn : Analyse de l'équité algorithmique.



🚀 Résultats Clés(Note : À compléter selon vos conclusions finales dans le notebook)Les modèles de Boosting surpassent généralement les GLM en termes de déviance.L'analyse SHAP révèle que les variables comme le BonusMalus et la Density sont des prédicteurs majeurs.L'audit d'équité permet d'ajuster les tarifs pour éviter une sur-pénalisation injustifiée de certains segments
