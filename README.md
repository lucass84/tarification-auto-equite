
#  Tarification Auto & IA : Modélisation de la Prime Pure, Benchmark ML vs GLM et Audit d'Équité

Ce projet propose une approche moderne de la modélisation de la prime pure en utilisant le dataset freMTPL2, l'objectif de ce projet est de comparer les outils classiques en actuariat (GLM Poisson/ Gamma) aux outils de machine learning actuels et de regarder les enjeux éthiques de ces modèles performants.

## 📌 Sommaire
1. [Présentation du Projet](#-Présentation-du-Projet)
2. [Jeu de données](#-Jeu-de-données)
3. [Préparation des données](#-Préparation-des-donnes)
4. [Analyse Exploratoire des Données (EDA)](#-analyse-exploratoire-des-données-eda)
5. [Méthodologie & Modélisation](#-méthodologie--modélisation)
6. [Résultats & Performance](#-résultats--performance)
7. [Equité algorithmique](#-Equité-algorithmique)
8. [Structure du Projet](#-structure-du-projet)

---

##  Présentation du Projet

La tarification en assurance automobile repose sur la modélisation séparée de la **fréquence** et de la **sévérité** des sinistres. Ce projet confronte la rigueur des standards actuariels aux nouvelles capacités du Machine Learning.

**Structure de la modélisation:**

-   **L'Approche Traditionnelle (GLM) :** Mise en œuvre de Modèles Linéaires Généralisés, utilisant une loi de Poisson pour modéliser le nombre de sinistres (Fréquence) et une loi Gamma pour le coût moyen (Sévérité). Ces modèles vont constituer notre base pour l'interprétabilité.
    
-   **L'Approche Machine Learning :** Utilisation d'algorithmes de boosting pour capturer des interactions complexes et non-linéaires entre les variables (âge, puissance, zone) sans hypothèse de distribution préalable.
    
-   **L'Enjeu Éthique :** Le gain de précision du Machine Learning justifie-t-il sa complexité ? Nous auditons ces modèles pour vérifier si leur performance ne se fait pas au détriment de l'équité entre les assurés.

---

## Jeu de données

Ce projet s'appuie sur le jeu de données [freMTPL2](https://www.kaggle.com/datasets/karansarpal/fremtpl2-french-motor-tpl-insurance-claims/data)  qui est une référence en tarification non-vie, les données sont traitées en deux parties distinctes, la fréquence des sinistres, modélisée par la loi de Poisson et la sévérité, modélisée par la loi Gamma.


Pour comprendre le jeu de données , regardons nos différentes variables et ce qu'elles représentent :

![Liste_var](https://github.com/user-attachments/assets/c6b61ff3-ad38-41eb-91f7-a60555642451)

**Précision sur le dataset :**
* Les données que nous analyserons représentent une année d'assurance française. 
* Ce dataset présente 676 780 lignes dans la partie fréquence et 36 639 lignes dans la partie sévérité.
* Les données du dataset sont anonymisées mais proviennent de réelles polices d'assurance du marché français, ce qui permet de confronter les modèles à la complexité concrète du risque.

## Préparation des données

Cette étape préliminaire garantit l'intégrité et la fiabilité de la pipeline. Le nettoyage des données est effectué avant l'EDA car l'EDA est réalisée exclusivement sur le jeu de données d'entraînement (_Train Set_). Cela permet d'éviter toute fuite d'information et de garantir que les décisions de feature engineering ne sont pas biaisées par la connaissance du jeu de test. Pour commencer, regardons comment sont distribuées nos valeurs:

![Stats_descriptives](https://github.com/user-attachments/assets/7bfeca45-5776-47ec-ac93-2f9029aba923)

Premièrement, le 99ème percentile de `ClaimAmount` est à **1250€**, mais le montant maximal est à plus de **4M€**, ces énormes valeurs sont problématiques car elles peuvent biaiser notre modèle alors qu'elles relèvent de l'aleatoire, dans l'EDA sévérité, nous nous pencherons dessus de nouveau pour savoir ce qu'on fait.

On observe également une anomalie sur la variable `exposure` (exposition), dont la valeur maximale atteint **2.01**. Or, l'exposition représente la fraction d'année durant laquelle l'assuré est couvert ; elle devrait donc être comprise strictement entre **0 et 1**. Nous supprimons donc les valeurs de `exposure` qui ne sont pas dans l'intervalle ]0,1].

La variable `ClaimNb` présente des valeurs extrêmes allant jusqu'à 16 sinistres par an. Afin de garantir la robustesse de notre modèle de fréquence, nous avons choisi de limiter l'étude aux assurés ayant déclaré au maximum 4 sinistres.

Nous réalisons un split précoce du jeu de données en ensembles d'entraînement (**80%**) et de test (**20%**). Cette étape, effectuée avant l'EDA, garantit l'absence de tout risque de **Data Leakage** (fuite de données).

##  Analyse Exploratoire des Données (EDA)

Afin d'assurer la robustesse de nos futurs modèles et d'éviter tout phénomène de fuite de données, l'analyse a été structurée en trois phases distinctes :

**1. Analyse de la Fréquence (`ClaimNb`):**

Cette partie se concentre sur la probabilité qu'un sinistre ait lieu.

-   **Objectif** : Comprendre quels facteurs (âge, zone, puissance du véhicule) influencent le nombre d'accidents.
    
-   **Traitement** : Les données sont analysées en tenant compte de l'exposition pour calculer des fréquences annualisées cohérentes.
    

**2. Analyse de la Sévérité (`ClaimAmount`):**

Ici, nous isolons le coût moyen des sinistres lorsqu'ils surviennent.

-   **Objectif** : Identifier si certaines variables impactent le montant des dommages (ex: les accidents en zone rurale sont-ils plus coûteux ?).
    
-   **Rigueur** : Cette analyse est menée indépendamment de la fréquence pour éviter que la volatilité des montants extrêmes ne biaise la prédiction du nombre de sinistres.
    

### EDA fréquence


L'objectif de cet EDA dédié à la fréquence est d'étudier les différents facteurs qui influent sur la sinistralité. Dans ce dataset, une des principales difficultés est la grande proportion d'observations sans sinistres (~95%), et cette grande proportion d'assurés sans sinistres rend la modélisation de la fréquence délicate.

#### Contrôle qualité des données

Dans un premier temps, il est essentiel de s'assurer de l'intégrité des données. On commence donc par effectuer une analyse plus approfondie sur la distribution des valeurs sur notre dataset dédié à la fréquence, l'objectif ici est de s'assurer que notre modèle soit stable et ne sur-apprend pas sur des cas isolés.

Le graphique ci-dessous illustre parfaitement cette distribution et met en évidence le grand nombre  de valeurs nulles, ainsi que la présence de points atypiques:

![Plot_freq](https://github.com/user-attachments/assets/d492ee5c-d1a1-43d2-ad27-93db663e0973)

On voit très bien ici que la grande majorité des assurés n'ont fait aucune réclamation. Cette distribution nous laisse penser que la survenue d'un sinistre suit un processus de Poisson.

Et notre ratio Variance/moyenne (test de surdispersion) vient nous le confirmer, comme la loi de Poisson admet une espérance et une Variance identique, $$X \sim \mathcal{P}(\lambda) \implies \mathbb{E}[X] = \text{Var}(X) = \lambda$$.

Alors un ratio de 1 signifie que notre variable suit bien une loi de Poisson. Dans notre cas, ce ratio est de 1.061 (faible surdispersion), ce qui nous indique que la variable `ClaimNb` suit une loi proche de celle de Poisson.

#### Analyse univariée

Analysons désormais comment se distribue nos autres variables, commençons par les varibles numériques:

![Plot_var_num_df_train_freq](https://github.com/user-attachments/assets/6f188c59-9b22-4674-84df-d86c1bdcdc4d)

Le tableau met en avant plusieurs points importants :
 * **Sinistralité et fréquence :** Le nombre de sinistres monte jusqu'à 11, ce qui est une valeur extreme pour un contrat annuel, de plus, certaines fréquences vont au-delà de 700, ces valeurs aberrantes sont probablement dû à une période d'exposition très courte, nous supprimons donc ces sinistres car ils risquent de biaiser notre modèle.
 
 *  **Âge du conducteur :** On constate que la distribution de l'âge se concentre majoritairement entre 30 et 60 ans, de plus, très peu d'assurés ont moins de 20 ans, cela peut s'expliquer par le fait qu'ils n'ont pas encore leur permis ou leur propre voiture.
 
 * **Bonus-Malus :** Ce pic à 50 (bonus maximal) nous indique que les assurés sont majoritairement des bons conducteurs et n'ont pas eu d'accidents responsables au cours des 13 dernières années, c'est donc cohérent avec le faible taux de sinistralité vu précédemment.
 
 * **Exposition :** Dans la distribition de la variable `Exposure`, on observe deux pics, un au point 1 et un second au point 0, le pic au point 1 est prévisible et correspond aux assurés qui sont protégés toute l'année, quant aux assurés avec une exposition proche de 0, cela peut correspondre à des personnes qui se sont assurés à la fin de l'année, à des personnes qui ont juste essayé l'assurance ou des personnes qui ont eu un accident peu de temps après leur souscription et ont vendu leur véhicule par la suite. C'est par ailleurs ce groupe qui provoque ces fréquences extrêmes.

Analysons désormais la distribution des variables catégorielles:

![Var_cat_df_train_freq](https://github.com/user-attachments/assets/f1c5d06e-6f7a-4b75-a0e0-10ff6d6b601d)

Encore une fois, ce tableau nous donne plusieurs points importants:
* **Zone  (`Area`):** Les zones C,D et E de la variable `Area` représentent plus de 70% de nos données et représentent les zones moyennement dense, à l'inverse la zone F représente que 2,65% des assurés malgré le fait qu'elle représente une forte densité de population. Ce faible pourcentage peut créer une plus forte volatilité sur cette zone. Ici, la variable `Area` possède une définition proche de celle de `Density`, nous regarderons le lien entre ces variables pour déterminer les choix effectués sur cette variable.

* **Motorisation :** On observe un équilibre quasi-parfait entre les deux différents types de motorisaton (51% essence/ 49% diesel), dans notre cas, le one-hot encoder semble le plus adapté.

* **Marques de véhicule :** On constate que notre dataset est dominé par trois grandes catégories, B12, B1 et B2, ces 3 catégories regroupent près de 72% des contrats. Ici, on va effectuer un binning des marques les moins présentes qu'on va appeler autre, puis on effectue de nouveau le one-hot encoder.

* **Région :** Par définition même de la variable `Region`, qui représente les différentes régions en France(découpage administratif avant 2016) et la définition de la variable `Area`, qui représente la densité de population, alors des dépendances apparaissent naturellement, par exemple la région R11 (Ile-de-France) est fortement corrélé aux zones E et F. La variable `Region`possède donc une distribution inégale qui reflète la géographie de la France. Cette variable sera donc testée pour capter des facteurs géographiques que la variable `Area` ne parviendrait pas à expliquer. Nous allons donc effectuer un target encoding sur la variable `Region` pour capturer du mieux possible les risques liés à la région.

#### Analyse bivariée

Cette section examine comment chaque variable influence la **fréquence annuelle de sinistralité**.

![plot_num_df_train_freq](https://github.com/user-attachments/assets/53e2bf7c-665b-4acd-b278-b9861228664c)

On remarque imédiatement que chacune de ces variables ont un pouvoir prédictif sur la variable cible `ClaimNb`, effectuons une analyse plus poussé sur chacune de ces variables:

* **Âge du conducteur :** On constate que les assurés les plus jeunes ont la plus grande fréquence (~0.37), les autres tranches d'âge affichent une fréquence moyenne plutôt stable qui se situe autour de 0.25, avec un maximum local à près de 0.3, cette information nous confirme notre choix d'effectuer un binning de l'âge du conducteur, et d'ensuite effectuer un one-hot encoder.

*  **Bonus-malus :** C'est le prédicteur le plus linéaire. La fréquence de sinistres augmente proportionnellement au coefficient de Bonus-Malus. Les assurés ayant un malus (coefficient > 100) ont une fréquence deux à trois fois supérieure à ceux bénéficiant du bonus maximal (50).

* **Âge du véhicule :** La sinistralité est maximale pour les véhicules neufs (0-1 an) puis diminue. Cela peut refléter une utilisation plus fréquente ou une tendance plus élevée à déclarer des sinistres pour des véhicules de forte valeur.

* **Puissance du véhicule :** Bien que plus volatile, la fréquence globale tend à augmenter avec la puissance fiscale du véhicule, notamment au-delà de 9 CV.

* **Densité de population :** Le graphique confirme une corrélation positive forte. Plus la densité augmente (notamment au-delà de 4 000 $hab/km^2$), plus la fréquence d'accidents (souvent urbains et à faible gravité) s'accroît.


Analysons désormais la distribution des variables catégorielles comparé à la variable fréquence :

![plot_cat_df_train_freq](https://github.com/user-attachments/assets/af24050b-0f3e-44ee-ac9b-774f147e4404)

Ces graphiques vont nous aider à valider ou non nos choix d'encodage, étudions les variable par variable:

-   **Zone (`Area`)** : On observe que le risque augmente de manière linéaire avec la densité. La zone F   (la plus urbaine) présente la fréquence moyenne la plus élevée, soit presque le double de la zone A. Cette tendance nous confirme la pertinence d'un ordinal encoding, afin de respecter cette hiérarchie.
    
-   **Marque (`VehBrand`)** : On constate ici que la marque B12 est beaucoup plus susceptible à l'apparition du sinistre (0.5 contre environ 0.2 sur les autres marques).
    
-   **Carburant (`VehGas`)** : Les véhicules Essence (Regular) affichent une fréquence moyenne supérieure aux véhicules Diesel, ce résultat s'explique par l'usage majoritairement urbain des moteurs essence, qui a une plus grande densité de circulation et augmente donc les probabilités de sinistre, contrairement à la motorisation diesel, plus fréquent chez les gros rouleurs sur autoroute.

  -   **Région (`Region`)** : La forte volatilité observée est également dû à un manque d'effectif sur certaines régions, ce qui confirme notre choix de target encoding lissé.

### EDA sévérité

#### Contrôle qualité des données et analyse univariée

Pour analyser notre dataset de sévérité, nous étudions la distribution de la variable `ClaimAmount` qui représente le coût de chaque sinistre, son étude est essentielle pour déterminer la manière dont on va modéliser la sévérité.

![ClaimAmount](https://github.com/user-attachments/assets/2c1f5e70-5c8c-4f56-9230-8fdffae01b18)

On remarque que le graphique est illisible, et ce résultat est prévisible car de nombreux sinistres sont sans suite,l'affichage est donc plombé par les sinistres ayant un montant de réclamation de 0.

![ClaimAmount_log](https://github.com/user-attachments/assets/9bc23776-e693-4da6-9a5f-0dfb1fa4a409)

Cette distribution en cloche au passage au log est la preuve que les valeurs de `ClaimAmount` suivent une loi log-Normale, pour la modélisation, on pourra donc utiliser le GLM Gamma avec une fonction de lien log, la loi Gamma est souvent plus robuste que la Log-Normale pour prédire la moyenne des coûts, tout en grdant le lien log pour la structure multiplicative des tarifs.


#### Analyse bivariée

Passons désormais à l'analyse bivariée de l'analyse sévérité, commençons par analyser la distribution de nos variables numériques comparé à la variable `Severity` :

![plot biv sev num](https://github.com/user-attachments/assets/cdc0a798-aa15-4fad-a554-8f6bcd7ace4c)

Ces graphiques nous aident à mieux comprendre l'impact desvariables numériques surnotre variable cible `Frequency`:

-   **Âge du Conducteur (`DrivAge`) :** On observe une sévérité bien plus marquée pour la tranche 18-26 ans (dépassant les 5 000 €). Ce montant marque une rupture par rapport aux autres variables. De plus,  on a vu précedemment que peu d'assurés sont dans cette tranche d'âge, ce résultat peut donc être causé par un gros sinistre dans cette tranche d'âge qui biaise l'ensemble des valeurs.
    
-   **Ancienneté du Véhicule (`VehAge`) :** Un pic de sévérité très prononcé apparaît pour les véhicules de 12 à 14 ans. Cela peut être traduit par des montants de réparation ou de dommage corporels très important, possiblement biaisé par un gros sinistre.
    
-   **Puissance du Véhicule (`VehPower`) :** La sévérité atteint son maximum pour les véhicules de puissance 8-9. Cette catégorie semble représenter un segment de risque spécifique (véhicules plus rapides ou plus coûteux à réparer) avant de redescendre pour les puissances extrêmes.
    
-   **Densité de Population (`Density`) :** Contrairement à la fréquence, la sévérité ne semble pas croître linéairement avec la densité. Un pic notable apparaît pour les densités intermédiaires (83-152), suggérant des types d'accidents plus graves en zone périurbaine qu'en hyper-centre congestionné.
    
-   **Bonus-Malus :** On constate une explosion de la sévérité pour les profils ayant un malus très élevé (> 95). Cela confirme que les conducteurs à forte sinistralité passée génèrent également les sinistres les plus coûteux en moyenne.

Poursuivons avec l'analyse de nos variables catégorielles comparés avec la variable `Severity`:

![plot biv sev cat](https://github.com/user-attachments/assets/4dfb3f5a-0448-4f78-a42b-ab3be1d1ac38)

Cette analyse permet d'identifier les disparités géographiques dans le coût moyen des sinistres.

-   **Zone Géographique (`Area`)** : La Zone B se détache nettement avec la sévérité moyenne la plus élevée (approchant les 4 000 €). À l'inverse, la Zone F présente la sévérité la plus faible, ce qui laisse penser que les accidents y sont moins coûteux, bien que potentiellement plus fréquents. 

* **Régions (`Region`)** : On observe de grands écarts entre les différentes régions. La Région R21 présente une sévérité bien supérieure au reste du pays. D'après nos analyses précédentes, cette région (Champagne-Ardennes) possède très peu d'asurés, et possède donc une grande variance, un seul gros sinistre peut justifier cette sévérité.

-   **Marque du Véhicule (`VehBrand`)** : Certaines marques (comme B11 et B2) affichent des coûts moyens de réparation bien supérieurs à la moyenne. Cela peut s'expliquer par le prix des pièces détachées ou des technologies embarquées plus coûteuses à remplacer., ou bien, dans le cas de B11, cela peut s'expliquer par la grande variance causée par le manque de données dans cette catégorie.
    
-   **Type de Carburant (`VehGas`)** : Les véhicules de type "Regular" (essence) présentent une sévérité moyenne supérieure aux véhicules "Diesel". De nombreux paramètres peut expliquer ce résultat, cela peut s'expliquer par le fait que l'essence est plus populaire chez les jeunes conducteurs, les moteurs essence sont aussi plus présents dans les voitures "sportives". De plus, les voitures essence sont plus présentes en ville, ce résultat peut donc aussi être causé par de petits accrochages répétitifs.  Nous ne pouvons donc pas faire de conclusion à l'heure actuelle sur ce résultat.

## Préprocessing et modélisation

Nous avons vu que la variable `ClaimAmount` présente une très grande variance, ce qui risque de poser problème pour l'entraînement de nos modèles de sévérite. Pour consolider nos modèles de sévérité, on écrette alors les montants de (`ClaimAmount`) jusqu'à hauteur de 99%, pour garder tout les sinistres les plus courants, sans prendre en compte les cas extrêmes.

Conne énoncé précedemment, nous effectuons un binning de notre variable `DrivAge`  en 6 compartiments que nous appelons `DrivAge_Bin`. Ce choix de 6 compartiments se repose sur la non-linéarité du risque vu précedemment afin de capturer au mieux chaque information sans créer trop de valeurs. En effet, on a une baisse brutale du risque à la fin de la période probatoire et remonte progressivement chez les séniors.

Dans le cadre de la modélisation en tarification automobile, nous avons mis en place un découpage distinct des données pour les deux composantes du modèle :

-   **Fréquence des sinistres**
    
-   **Sévérité des sinistres**
    

Les jeux de données sont désormais structurés comme suit :

 **Modèle de fréquence :**

-   `X_train_freq` : variables explicatives pour l'entraînement :
`Density`,`DrivAge_Bin`,`BonusMalus`,`Region`,`VehBrand`,`VehGas`,`vehPower` et`VehAge`.
    
-   `y_train_freq` : variable cible (fréquence) pour l'entraînement, contient la variable `ClaimNb`.
    
-   `X_test_freq` : variables explicatives pour le test :
 `Density`,`DrivAge_Bin`,`BonusMalus`,`Region`,`VehBrand`,`VehGas`,`vehPower` et`VehAge`.
    
-   `y_test_freq` : variable cible (fréquence) pour le test, contient la variable `ClaimNb`.
    

 **Modèle de sévérité :**

-   `X_train_sev` : variables explicatives pour l'entraînement, contient les variables : `Density`,`DrivAge_Bin`,`frequency`,`BonusMalus`,`Region`,`VehBrand`,`VehGas`,`vehPower` et`VehAge`.
    
-   `y_train_sev` : variable cible (sévérité) pour l'entraînement, contient la variable `ClaimAmount`.
    
-   `X_test_sev` : variables explicatives pour le test, contient les variables :
`Density`,`DrivAge_Bin`,`frequency`,`BonusMalus`,`Region`,`VehBrand`,`VehGas`,`vehPower` et`VehAge`.
    
-   `y_test_sev` : variable cible (sévérité) pour le test, contient la variable `ClaimAmount`.
    

Cette séparation permet :

-   d’entraîner indépendamment les modèles de fréquence et de sévérité
    
-   d’évaluer leurs performances respectives sur des jeux de test dédiés
    
-   de construire ensuite une prime pure via l’approche classique : $Prime\ Pure = Fréquence \times Sévérité$

Enfin, on récapitule nos choix sur le traitement de nos différentes variables:

**Variables numériques :**

* **log-scaler :** Nous passons au log notre variable `Density` pour lui donner une distribution proche d'une loi normale et ensuite nous appliquons le StandardScaler dessus pour le centrer autour de 0 et réduire sa variance à 1, ce qui permettra un meilleur apprentissage de nos modèles GLM et ML.

* **StandardScaler :** Pour nos variables `VehPower`,  `VehAge` et  `BonusMalus`, on les passe au StandardScaler, ce qui nous permet de donner la même importance à chacune de nos variables en les centrant (autour de 0) et en leur donnant la même variance qui est égal à 1.

**Variables catégorielles :**

* **One-hot encoding :** Nous allons effectuer cet encodage sur les variables : `VehGas` et `DrivAge_Bin`.

* **Target encoding :** On utilise cet encodage sur les variables `VehBrand` et `region` en raison de leur forte cardinalité.




La détermination de la **Prime Pure** repose sur une approche de modélisation en deux étapes  permettant de capturer au mieux le risque :

$$Prime\ Pure = Fréquence \times Sévérité$$


### 1. Modélisation de la Fréquence

L'objectif est d'estimer le nombre de sinistres par unité d'exposition. La durée de présence au contrat (`Exposure`) varie d'un assuré à l'autre. Pour obtenir une fréquence annuelle comparable, nous intégrons cette variable en tant qu'**offset**. Cela permet aux modèles de prédire un taux de sinistralité par an plutôt qu'un nombre brut de sinistres, garantissant une tarification équitable. Pour cette composante, nous confrontons le standard de l'industrie à la puissance du Machine Learning :

-   **Modèle de référence :** Le **GLM Poisson**. Nous avons utilisé **GridSearchCV** pour optimiser ses paramètres de régularisation, assurant un équilibre optimal entre interprétabilité et performance.
    
-   **Modèles de Machine Learning :** Nous challengerons ce standard avec **XGBoost**, **LightGBM** et **Random Forest**. Pour chacun, une recherche par grille (GridSearchCV) a été menée afin de calibrer les hyperparamètres et capturer au mieux les interactions non-linéaires.
    

### 2. Modélisation de la Sévérité

Cette étape consiste à estimer le coût moyen par sinistre. Afin de donner plus de poids statistique aux assurés ayant eu plusieurs sinistres, nous utilisons la variable `ClaimNb` comme **poids** lors de l'entraînement. Cela permet au modèle de distinguer les tendances réelles du bruit aléatoire. La distribution des coûts étant asymétrique avec une queue étalée, nous comparons :

-   **Modèle de référence :** Le **GLM Gamma**, optimisé par **GridSearchCV** pour garantir la meilleure convergence et précision du coût moyen.
    
-   **Modèles de Machine Learning :** L'utilisation de **XGBoost** et **LightGBM** (avec fonction de perte Gamma). Ces modèles ont également bénéficié d'un réglage fin par **GridSearchCV** pour affiner la prédiction des coûts et stabiliser les résultats face à la forte variance des sinistres.

##  Résultats & Performance

### Choix du modèle

Regardons nos résultats pour nos modèles de Fréquence: 

![graphique modèle freq](https://github.com/user-attachments/assets/02778729-e0d8-498d-9671-c5904e6a3c04)

La déviance de Poisson a été retenue comme métrique de référence pour départager nos modèles , l'objectif étant de minimiser cette valeur. Le modèle XGBoost ayant obtenu le score le plus bas (0,300225), c’est sur celui-ci que nous baserons nos analyses et nos résultats finaux.


![graphique modèle sev](https://github.com/user-attachments/assets/dcdee7d0-37ab-4fc0-b61d-2f6910a12d4f)

Afin d’identifier le modèle le plus performant, nous avons privilégié cette foois-ci la déviance Gamma, l'objectif étant toujours de minimiser cette valeur. Le modèle XGBoost a obtenu le score le plus bas (1,060572). Nous nous baserons donc encore sur ce modèle pour nos analyses et résultats finaux.

### Performance technique
Le recalibrage sur données cappées permet de valider la performance prédictive du modèle sur les sinistres courants. Il prouve que la segmentation est robuste et que le modèle a correctement appris le comportement moyen du risque sans être pollué par le bruit des événements extrêmes.

![Gini cappé](https://github.com/user-attachments/assets/7d2ceb0c-061b-461c-853d-08ee606e65aa)

Regardons désormais le Lift Chart pour avoir un aperçu de la manière dont notre modèle trie les risques:

![Lift chart cappé](https://github.com/user-attachments/assets/e727d301-3d3b-4c1f-9acc-a9c62a419a4d)

Ce graphique est l'outil de validation final de notre modèle. Il compare la prime pure prédite par le modèle combiné (Fréquence $\times$ Sévérité) à la prime pure réelle, calculée sur des montants de sinistres écrêtés au 99,5ème percentile.

On observe que notre modèle estime bien les différents déciles, ce qui nous confirme que notre modèle est bien calibré, pour 80 % de la population (déciles 0 à 7), l'erreur de tarification est très basse. La quasi-monotonie des deux courbes confirme également un fort pouvoir de segmentation, validé par un coefficient de Gini de 0.248. Ce qui nous confirme que notre modèle hiérarchise les assurés avec une bonne précision

### Validation business
Le coefficient de Gini permet d'évaluer le pouvoir discriminant du modèle, soit sa capacité à hiérarchiser les assurés selon leur niveau de risque. Analysons ses résultats après l'étape de recalibrage sur cette fois-ci les sinistres n'ayant subi aucun plafonnement :
![Gini non cappé](https://github.com/user-attachments/assets/aec379d0-617c-4941-a465-5758a190c506)

Regardons désormais le Lift Chart associé pour obtenir un meilleur aperçu de la gestion des risques de notre modèle :

![Lift chart non cappé](https://github.com/user-attachments/assets/106789d4-bcd9-442a-b4b9-6b4e4acd49e3)

Le point le plus frappant est la non-monotonie sévère du coût réel, particulièrement visible entre les déciles 4 premiers déciles.

-   **Constat :** On observe que le coût réel du décile 2 est bien plus élevé que celui des déciles 3 et 4, alors que le modèle prédit l'inverse.
    
-   **Explication :** C'est la preuve directe de l'impact des sinistres graves. Dans un jeu de données non cappé, un seul sinistre de 50 000 € tombé par "hasard" dans le décile 2 fait exploser sa moyenne. Sans écrêtage, le hasard prend le pas sur la vraie tendance statistique .

## Equité algorithmique

Les modèles de Machine Learning exploitent des interactions complexes et non-linéaires qui peuvent reproduire ou amplifier des biais discriminatoires non voulu. Cette section est dédiée à l'audit de l'équité de nos modèles : nous analysons les risques d'injustice algorithmique et proposons des mesures correctives pour garantir une tarification responsable.Cette analyse se basera sur 2 variables, la variable `Area`, qui représente les différentes zones urbaines et la variable `DrivAge_Bin`, qui représente l'âge du conducteur, commençons par la variable `Area` et analysons son audit d'équité :

![Audit équité Area](https://github.com/user-attachments/assets/b8049087-434e-47e5-a231-9bd7e00fdf11)

Cet audit d'quité nous montre que les habitants de la zone urbaine F, paie en moyenne 13% plus que les autres, l'audit ne semble pas injuste au premier abord, en effet, quand on regarde le MAE, le modèle est même plus précis chez ces habitants que dans les autres régions, cependant, la zone F représente seulement 2,6% des données, ce n'est donc pas assez pour s'assurer de la robustesse de notre modèle.

Effectuons désormais un audit d'équité de la variable `DrivAge_Bin` :

![Audit équité DrivAge](https://github.com/user-attachments/assets/caa7acff-4ab5-4578-8c13-967d703b9ac8)

Ici, on constate que les jeunes conducteurs paient le plus (~218€), et ce constat est logique puisque d'après notre EDA, ce sont eux qui ont la fréquence d'accidents la plus élevée, de plus, on retrouve également cette remontée de la prime jusqu'à environ 175€, ce qui était attendu d'après notre EDA, en revanche, on constate que pour les 18-21 ans, le MAE est énorme , près de 400€, ce qui nous laisse suggérer que pour ces assurés, c'est soit ils ont pas d'accidents, soit un accident coûteux (variance élevée), à l'invers, pour les 56-70 ans, le modèle est beaucoup plus stable avec une MAE de 183€.

##  Installation

Pour exécuter ce projet, vous devez disposer d'un environnement **Python 3.8+**.

**Toutes les étapes suivantes se font sur PowerShell ou Windows terminal :**
### 1.  Autoriser l'activation d'environnement
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
```
### 2. Récuperer le projet

```powershell
git clone https://github.com/Lucass84/tarification-auto-equite.git
cd tarification-auto-equite
```
### 3. Créer et activer l'environnement virtuel
```powershell
# Création de l'environnement 
python -m venv venv 

# Activation (Windows 11) 
.\venv\Scripts\activate
```

### 4. Installer les bibliothèques

```bash
# mise à jour pip
python -m pip install --upgrade pip 

Installation bibliothèques
pip install -r requirements.txt
```

### 5. Lancer programme
```powershell
jupyter notebook
```

## Amélioration future:
Le projet actuellement contient de nombreuses faiblesses qui vont être remédiées à l'avenir :
* EDA sévérité effectuée sur `ClaimAmount` cappé pour obtenir une meilleure visualisation de notre variable et des intéractions avec les autres variables.

* Ajout de modèle Tweedie en gridsearchCV pour modéliser directement la prime pure sans décomposition Fréquence/ Sévérité.

* Ajout du modèle Catboost sur la modélisation fréquence et sévérité avec gridsearchCV

* Approfondissement de la partie Equité algorithmique avec l'arrivé de Fairlearn et de différents graphiques, un ré entrainement du modèle avec un nouveau poids associé à chaque variable si modèle injuste et réévaluation du modèle.

* Amélioration des modèles déja existants (optimisation des hyperparamètres)

## Sources
[1] [Guide NumPy](https://pandas.pydata.org/docs/user_guide/index.html)

[2] [Guide Pandas](https://pandas.pydata.org/docs/user_guide/index.html)

[3] [Guide SciKit-Learn](https://scikit-learn.org/stable/user_guide.html)

[4] [Retour bonus-Malus](https://www.meilleurtaux.com/comparateur-assurance/assurance-auto/guide-assurance-auto/bonus-malus/recuperer-bonus-auto.html#:~:text=M%C3%AAme%20apr%C3%A8s%20un%20malus,%20le,progresser%20votre%20bonus%20de%205%25.)

[5] [Guide FairLearn](https://fairlearn.org/main/user_guide/fairness_in_machine_learning.html)

[6] [Guide XGBoost](https://www.datacamp.com/fr/tutorial/xgboost-in-python?dc_referrer=https://www.google.com/)

[7] [Guide LightGBM](https://lightgbm.readthedocs.io/en/stable/)

[8] [Guide RandomForest](https://www.datacamp.com/fr/tutorial/random-forests-classifier-python)

[9] [Insurance Premium prediction - NeuralNine](https://www.youtube.com/watch?v=tEH5EuKPHa8&t=2023s)
