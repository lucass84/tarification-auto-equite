
#  Tarification Auto & IA : Modélisation de la Prime Pure, Benchmark ML vs GLM et Audit d'Équité

Ce projet propose une approche moderne de la modélisation de la prime pure en utilisant le dataset freMTPL2, l'objectif de ce projet est de comparer les outils classiques en actuariat (GLM Poisson/ Gamma) aux outils de machine learning actuels et de regarder les enjeux éthiques de ces modèles performants.

## 📌 Sommaire
1. [Présentation du Projet](#-présentation-du-projet)
2. [Dataset](#-Dataset)
3. [Préparation données](#-préparation-donnes)
4. [Analyse Exploratoire des Données (EDA)](#-analyse-exploratoire-des-données-eda)
5. [Méthodologie & Modélisation](#-méthodologie--modélisation)
6. [Résultats & Performance](#-résultats--performance)
7. [Structure du Projet](#-structure-du-projet)

---

##  Présentation du Projet

La tarification en assurance automobile repose sur la modélisation séparée de la **fréquence** et de la **sévérité** des sinistres. Ce projet confronte la rigueur des standards actuariels aux nouvelles capacités du Machine Learning.

**Structure de la modélisation:**

-   **L'Approche Traditionnelle (GLM) :** Mise en œuvre de Modèles Linéaires Généralisés, utilisant une loi de Poisson pour modéliser le nombre de sinistres (Fréquence) et une loi Gamma pour le coût moyen (Sévérité). Ces modèles vont constituer notre base pour l'interprétabilité.
    
-   **L'Approche Machine Learning :** Utilisation d'algorithmes de boosting pour capturer des interactions complexes et non-linéaires entre les variables (âge, puissance, zone) sans hypothèse de distribution préalable.
    
-   **L'Enjeu Éthique :** Le gain de précision du Machine Learning justifie-t-il sa complexité ? Nous auditons ces modèles pour vérifier si leur performance ne se fait pas au détriment de l'équité entre les assurés.

---

## Dataset 

Ce projet s'appuie sur le dataset [freMTPL2](https://www.kaggle.com/datasets/karansarpal/fremtpl2-french-motor-tpl-insurance-claims/data)  qui est une référence en tarification non-vie, les données sont traitées en deux parties distinctes, la fréquence des sinistres, modélisée par la loi de Poisson et la sévérité, modélisée par la loi Gamma.


Pour comprendre le dataset , regardons nos différentes variables et ce qu'elles représentent

![Liste_var](https://github.com/user-attachments/assets/c6b61ff3-ad38-41eb-91f7-a60555642451)

Pour avoir un meilleur aperçu de ce que représente notre variable (`Region`), on va l'appliquer sur une carte de la France en fonction du nombre pou obtenir un meilleur aperçu.

## Préparation données

c


##  Analyse Exploratoire des Données (EDA)

Mon EDA s'effectue en 3 temps, une première partie dédiée à la fréquence, une seconde dédiée à la sévérité et enfin une partie comparative. Pour éviter toute fuite de donnée et des choix basés sur notre dataset 

### EDA fréquence

L'objectif de cet EDA dédié à la fréquence est d'étudier les différents facteurs qui influent sur la sinistralité. Dans ce dataset, une des principales difficultés est la grande proportion d'observations sans sinistres (~95%), et cette grande proportion d'assurés sans sinistres rend la modélisation de la fréquence délicate.

#### Gestion des valeurs aberrantes
Dans un premier temps, il est essentiel de s'assurer de l'intégrité des données. On commence donc par effectuer une analyse plus approfondie sur la distribution des valeurs sur notre dataset dédié à la fréquence, l'objectif ici est de s'assurer que notre modèle soit stable et ne sur-apprend pas sur des cas isolés.

Le graphique ci-dessous illustre parfaitement cette distribution et met en évidence le grand nombre  de valeurs nulles, ainsi que la présence de points atypiques:

![Plot_freq](https://github.com/user-attachments/assets/e4fcd176-8fb0-4555-991c-5766bc1e6090)

On voit très bien ici que la grande majorité des assurés n'ont fait aucune réclamation. Cette distribution nous laisse penser que la survenue d'un sinistre suit un processus de Poisson.

Et notre ratio Variance/moyenne (test de surdispersion) vient nous le confirmer, comme la loi de Poisson admet une espérance et une Variance identique, $$X \sim \mathcal{P}(\lambda) \implies \mathbb{E}[X] = \text{Var}(X) = \lambda$$.

Alors un ratio de 1 signifie que notre variable suit bien une loi de Poisson. Dans notre cas, ce ratio est de 1.077 (faible surdispersion), ce qui nous indique que la variable `ClaimNb` suit une loi proche de celle de Poisson.



Analysons désormais comment se distribue nos autres variables, commençons par les varibles numériques:

![Plot_var_num_df_train_freq](https://github.com/user-attachments/assets/f2a2cfcd-8f5f-4ea5-828b-30f019c05251)

Le tableau met en avant plusieurs points importants :
 * **Sinistralité et fréquence :** Le nombre de sinistres monte jusqu'à 11, ce qui est une valeur extreme pour un contrat annuel, de plus, certaines fréquences vont au-delà de 700, ces valeurs aberrantes sont probablement dû à une période d'exposition très courte, nous supprimons donc ces sinistres car ils risquent de biaiser notre modèle.
 
 *  **Age du conducteur :** On constate que la distribution de l'âge se concezntre majoritairement entre 30 et 60 ans, de plus, très peu d'assurés ont moins de 20 ans, cela peut s'expliquer par le fait qu'ils n'ont pas encore leur permis ou leur propre voiture.
 
 * **Bonus-Malus :** Ce pic à 50 (bonus maximal) nous indique que les assurés sont majoritairement des bons conducteurs et n'ont pas eu d'accidents responsables au cours des 13 dernières années, c'est donc cohérent avec le faible taux de sinistralité vu précédemment.
 
 * **Exposition :** Dans la distribition de la variable `Exposure`, on observe deux pics, un au point 1 et un second au point 0, le pic au point 1 est prévisible et correspond aux assurés qui sont protégés toute l'année, quant aux assurés avec une exposition proche de 0, cela peut correspondre à des personnes qui se sont assurés à la fin de l'année, à des personnes qui ont juste essayé l'assurance ou des personnes qui ont eu un accident peu de temps après leur souscription et ont vendu leur véhicule par la suite. C'est par ailleurs ce groupe qui provoque ces fréquences extrêmes.

En approfondissant l'analyse sur notre variable cible `ClaimNb`, elle révèle un petit nombre d'observations avec des fréquences de sinistralité très élevé, comme on peut le voir dans le tableau ci-dessous:

![Contrats 4+ sinistres](https://github.com/user-attachments/assets/2baed577-eb0a-4477-ad23-d897a0a4db47)

Parmi ces contrats, il y a des cas extrêmes qui sont plausibles, et d'autres totalement aberrants (ex: 11 sinistres en 3 semaines), j'ai fait le choix de supprimer les sinistres ci-dessus avec une fréquence annuelle supérieur ou égal à 40. Car ces données sont probablement des erreurs de saisie et risque de nuire à la stabilité de notre modèle.



Analysons désormais la distribution des variables catégorielles:

![Var_cat_df_train_freq](https://github.com/user-attachments/assets/f1c5d06e-6f7a-4b75-a0e0-10ff6d6b601d)

Encore une fois, ce tableau nous donne plusieurs points importants:
* **Zone :** Les zones C,D et E de la variable `Area` représentent plus de 70% de nos données et représentent les zones moyennement dense, à l'inverse la zone F représente que 2,65% des assurés malgré le fait qu'elle représente une forte densité de population. Ce faible pourcentage peut créer une plus forte volatilité sur cette zone. Ici, la variable `Area` possède une définition proche de celle de `Density`, nous regarderons le lien entre ces variables pour déterminer les choix effectués sur cette variable.

* **Motorisation :** On observe un équilibre quasi-parfait entre les deux différents types de motorisaton (51% essence/ 49% diesel), dans notre cas, le one-hot encoder semble le plus adapté.

* **Marques de véhicule :** On constate que notre dataset est dominé par trois grandes catégories, B12, B1 et B2, ces 3 catégories regroupent près de 72% des contrats. Ici, on va effectuer un binning des marques les moins présentes qu'on va appeler autre, puis on effectue de nouveau le one-hot encoder.

* **Région :** Par définition même de la variable `Region`, qui représente les différentes régions en France(découpage administratif avant 2016) et la définition de la variable `Area`, qui représente la densité de population, alors des dépendances apparaissent naturellement, par exemple la région R11 (Ile-de-France) est fortement corrélé aux zones E et F. La variable `Region`possède donc une distribution inégale qui reflète la géographie de la France. Cette variable sera donc testée pour capter des facteurs géographiques que la variable `Area` ne parviendrait pas à expliquer. Nous allons donc effectuer un target encoding sur la variable `Region` pour capturer du mieux possible les risques liés à la région.

Intéressons nous désormais aux liens entre nos différentes variables et notre variable cible `Frequency`

![Plot_biv_freq_cat](https://github.com/user-attachments/assets/73305604-54cc-42c1-a560-15c004469ab7)
### EDA sévérité

Pour analyser notre dataset df_train_sev, étudions la distribution de la variable (`ClaimAmount`) qui représente le coût de chaque sinistre, son étude est essentielle pour déterminer la manière dont on va modéliser la sévérité.

![ClaimAmount](https://github.com/user-attachments/assets/2c1f5e70-5c8c-4f56-9230-8fdffae01b18)

On remarque que le graphique est illisible, et ce résultat est prévisible car de nombreux sinistres sont sans suite,l'affichage est donc plombé par les sinistres ayant un montant de réclamation de 0.

![ClaimAmount_log](https://github.com/user-attachments/assets/9bc23776-e693-4da6-9a5f-0dfb1fa4a409)

Cette distribution en cloche au passage au log est la preuve que les valeurs de ClaimAmount suivent une loi Gamma

La variable `ClaimAmount` représente le coût de chaque sinistre. Son étude est cruciale car elle présente une distribution typique de l'assurance IARD : une très forte asymétrie avec une "queue épaisse" (long tail).

-   **Distribution brute :** Le premier graphique montre que la majorité des sinistres ont des montants faibles, mais quelques événements extrêmes écrasent l'échelle. Nous observons également un grand nombre de dossiers à 0 € (sinistres sans suite), que nous filtrons pour l'étude de la sévérité pure.
    
-   **Transformation Logarithmique :** Le passage à l'échelle log révèle une distribution proche de la loi Log-Normale. Cette forme confirme la pertinence d'utiliser une **loi Gamma** (via LightGBM ou XGBoost) pour modéliser le coût moyen, car elle gère naturellement la variance proportionnelle au carré de la moyenne.
    
-   **Stratégie de Capping :** Compte tenu de la présence de valeurs extrêmes pouvant déstabiliser l'apprentissage, nous avons fixé un seuil d'écrêtage (capping) à **20 000 €**. Cela permet au modèle de se concentrer sur la sinistralité de "masse" tout en traitant les grands sinistres via un facteur de redressement global.

---

## Préprocessing et modélisation

Pour consolider notre modèle de sévérité, on écrette les montants de (`ClaimAmount`) jusqu'à hauteur de 99,5% car les modèles de sévérité sont très sensibles aux 	très grandes valeurs




##  Résultats & Performance

### Performance technique
Le recalibrage sur données cappées permet de valider la performance prédictive du modèle sur la sinistralité de 'masse'. Il prouve que la segmentation est robuste et que le modèle a correctement appris le comportement moyen du risque sans être pollué par le bruit des événements extrêmes.

### Validation business
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








```



[Retour bonus-Malus](https://www.meilleurtaux.com/comparateur-assurance/assurance-auto/guide-assurance-auto/bonus-malus/recuperer-bonus-auto.html#:~:text=M%C3%AAme%20apr%C3%A8s%20un%20malus,%20le,progresser%20votre%20bonus%20de%205%25.)
