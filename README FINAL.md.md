
#  Tarification Auto & IA : Modélisation de la Prime Pure, Benchmark ML vs GLM et Audit d'Équité

Ce projet propose une approche moderne de la modélisation de la prime pure en utilisant le dataset freMTPL2, l'objectif de ce projet est de comparer les outils classiques en actuariat (GLM Poisson/ Gamma) aux outils de machine learning actuels et de regarder les enjeux éthiques de ces modèles performants.

## 📌 Sommaire
1. [Présentation du Projet](#-présentation-du-projet)
2. [Analyse Exploratoire des Données (EDA)](#-analyse-exploratoire-des-données-eda)
3. [Méthodologie & Modélisation](#-méthodologie--modélisation)
4. [Résultats & Performance](#-résultats--performance)
5. [Structure du Projet](#-structure-du-projet)

---

##  Présentation du Projet

La tarification en assurance automobile repose sur la modélisation séparée de la **fréquence** et de la **sévérité** des sinistres. Ce projet confronte la rigueur des standards actuariels aux nouvelles capacités du Machine Learning.

**Structure de la modélisation:**

-   **L'Approche Traditionnelle (GLM) :** Mise en œuvre de Modèles Linéaires Généralisés, utilisant une loi de Poisson pour modéliser le nombre de sinistres (Fréquence) et une loi Gamma pour le coût moyen (Sévérité). Ces modèles vont constituer notre base pour l'interprétabilité.
    
-   **L'Approche Machine Learning :** Utilisation d'algorithmes de boosting pour capturer des interactions complexes et non-linéaires entre les variables (âge, puissance, zone) sans hypothèse de distribution préalable.
    
-   **L'Enjeu Éthique :** Le gain de précision du Machine Learning justifie-t-il sa complexité ? Nous auditons ces modèles pour vérifier si leur performance ne se fait pas au détriment de l'équité entre les assurés.

---

##  Analyse Exploratoire des Données (EDA)

Mon EDA s'effectue en 2 tenmps, une première partie dédiée à la fréquence, une seconde partie à la sévérité et enfin une partie comparative.

### EDA fréquence
Le graphique ci-dessous nous montre comment sont distrbuées nos différentes variables.
Courbe de Gini:
![Gini](https://github.com/user-attachments/assets/e4fcd176-8fb0-4555-991c-5766bc1e6090)

Lift Chart:

![Lift Chart](https://github.com/user-attachments/assets/52dffab3-ff53-4ce7-be5d-24732b54fbe3)

On voit très bien ici que la grande majorité des clients n'ont fait aucune réclamation. Cette distribution nous laisse penser que la survenue d'un sinistre suit un processus de Poisson.

![Gini](https://github.com/user-attachments/assets/52dffab3-ff53-4ce7-be5d-24732b54fbe3)

Et notre ratio Variance/moyenne vient nous le confirmer, comme la loi de Poisson admet une espérance et une Variance identique, $$X \sim \mathcal{P}(\lambda) \implies \mathbb{E}[X] = \text{Var}(X) = \lambda$$

Alors ce ratio proche de 1 vient nous le confirmer.


### EDA sévérité

---

## Préprocessing et modélisation




##  Résultats & Performance

### Résultats bruts
Le recalibrage sur données cappées permet de valider la performance prédictive du modèle sur la sinistralité de 'masse'. Il prouve que la segmentation est robuste et que le modèle a correctement appris le comportement moyen du risque sans être pollué par le bruit des événements extrêmes.

### Résultats 
La courbe de Gini mesure la capacité à modéliser les bons des mauvais risques
![Distribution de la fréquence](https://github.com/user-attachments/assets/f94c527a-2176-46f2-b5e1-9b9cf0616e2c)
##  Structure du Projet
- `notebooks/` : Contient l'EDA détaillée et le tuning des hyperparamètres.
- `src/` : Scripts de prétraitement, d'entraînement et pipeline de scoring.
- `data/` : (Optionnel) Échantillon de données ou lien vers la source (ex: OpenData).
- `models/` : Modèles sérialisés (.pkl).

---

## 🛠 Installation
```bash
git clone [https://github.com/votre-nom/tarification-auto.git](https://github.com/votre-nom/tarification-auto.git)
pip install -r requirements.txt













