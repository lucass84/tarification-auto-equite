
# Projet de tarification auto en python avec comparaison ML et implémentation de Fairness

Ce projet propose une approche moderne de la modélisation de la prime pure en utilisant le dataset freMTPL2, l'objectif de ce projet est de comparer les outils classiques en actuariat (GLM Poisson/ Gamma) aux outils de machine learning actuels et de regarder les enjeux éthiques de ces modèles performants.

## Explication

La pipeline de mon projet se décompose en une première phase de préparation des données à travers l'EDA de la Fréquence puis de la Sévérité, une seconde phase de comparaison de modèles et d'entraînement de modèles, le choix du modèle fréquence final et du modèle sévérité finazl se basera sur différentes métriques telles que le Mean Square Error et la déviance Poisson pour la modélisation de la fréquence et le MAE et la déviance Gamma pour la modélisation de la sévérité , et pour finir une dernière phase de vérification éthique sur les modèles gagnants du comparatif.

Le projet est structuré de manière séquentielle, de l'acquisition des données à l'audit éthique final :

![Structure de la pipeline](https://github.com/user-attachments/assets/a0dce2db-2c1a-4f90-a57a-d9e782c22417)