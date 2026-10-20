# 05 · Résultats

Tous les chiffres viennent des fichiers de `training/results/`, calculés avec le
protocole décrit dans [Méthodologie](04-methodologie-scientifique.md). Sauf mention
contraire, ils portent sur les **parties test**, jamais utilisées pour entraîner ni pour
régler les modèles. Les intervalles entre parenthèses sont des intervalles de confiance
à 95 % (bootstrap).

## 1. Synthèse

| Modèle | ISIC 2020 (dermatoscope) : AUC | PAD-UFES-20 (smartphone) : AUC |
|---|---|---|
| Ancien modèle : EfficientNet-B0 / HAM10000 | 0,635 (0,575 à 0,694) | 0,843 (0,801 à 0,882) |
| **Modèle actuel : Derm Foundation + LR (PAD + ISIC)** | **0,857 (0,826 à 0,887)** | **0,962 (0,939 à 0,980)** |

Le modèle actuel est nettement meilleur sur les deux types d'images. Sur le dermatoscope,
les intervalles de confiance ne se recouvrent pas.

Deux classifieurs non linéaires, XGBoost et un petit réseau de neurones, ont été essayés à
la place de la régression logistique sur les mêmes embeddings : ils ne font pas mieux
(voir la [section 5](#5-classifieurs-non-linéaires--xgboost-et-réseau-de-neurones)).

## 2. ISIC 2020 — dermatoscope

6 162 images de test, 407 patients, **106 mélanomes** (seule lésion maligne de ce jeu).

| Mesure | Ancien modèle | Derm Foundation (ISIC seul) | **Derm Foundation (PAD + ISIC)** |
|---|---|---|---|
| AUC | 0,635 (0,575 à 0,694) | 0,860 (0,825 à 0,893) | **0,857 (0,826 à 0,887)** |
| Spécificité à 95 % de sensibilité | 8,4 % | 49,0 % | **57,6 %** |
| Mélanomes détectés au seuil à 95 % | 95,3 % | 96,2 % | 95,3 % |
| Mélanomes détectés au score 0,5 | 24,5 % | 62,3 % | 35,8 % |
| Seuil à 95 % (calculé sur le test) | 0,000 | 0,104 | 0,072 |

Lecture :

- **Ancien modèle** : pour attraper 95 % des mélanomes, son seuil tombe à 0 et il signale
  92 % des lésions bénignes. Il est presque inutilisable sur ces images.
- **Modèle actuel** : au même niveau de sensibilité, il ne signale plus que 42 % des
  lésions bénignes. C'est encore beaucoup en dépistage, mais l'écart est considérable.
- **Combiner les deux jeux ne dégrade pas le dermatoscope** : l'AUC est la même qu'avec
  ISIC seul, et la spécificité légèrement meilleure.
- La différence au score 0,5 (36 % contre 62 %) vient seulement d'un décalage des scores
  entre les deux classifieurs, pas d'une différence de qualité : seul le classement (AUC)
  et les seuils réglés comptent.

## 3. PAD-UFES-20 — smartphone

453 images de test, 272 patients, 367 malignes, **7 mélanomes**.

| Mesure | Ancien modèle | Derm Foundation (PAD seul) | **Derm Foundation (PAD + ISIC)** |
|---|---|---|---|
| AUC | 0,843 (0,801 à 0,882) | 0,974 (0,947 à 0,994) | **0,962 (0,939 à 0,980)** |
| Spécificité à 95 % de sensibilité | 36,0 % | 95,3 % | **82,6 %** |
| AUC mélanome / bénin | non calculée | non calculée | 0,899 |
| Mélanomes détectés au seuil à 95 % | 4 sur 7 | 3 sur 7 | **6 sur 7** |
| Mélanomes détectés au score 0,5 | 1 sur 7 | 4 sur 7 | 7 sur 7 |
| Seuil à 95 % (calculé sur le test) | 0,033 | 0,810 | 0,871 |

Lecture :

- Derm Foundation domine nettement l'ancien modèle (0,96 à 0,97 contre 0,84).
- Ajouter ISIC 2020 coûte un peu de spécificité sur smartphone (83 % contre 95 %), mais
  les intervalles de l'AUC se recouvrent largement, et le modèle combiné détecte mieux
  les mélanomes (6 sur 7 contre 3 sur 7 au seuil à 95 %). Sur 7 mélanomes, ces écarts ne
  sont pas significatifs.

### Validation croisée sur PAD-UFES-20

Derm Foundation + LR entraîné sur PAD-UFES-20 seul, 5 plis par patient, les 2 298 images
et 52 mélanomes :

| Mesure | Valeur |
|---|---|
| AUC | 0,982 (0,975 à 0,988) |
| Spécificité à 95 % de sensibilité | 93,3 % |
| AUC mélanome / bénin | 0,885 |
| Mélanomes détectés au seuil à 95 % | 51,9 % |
| Mélanomes détectés au score 0,5 | 59,6 % |

Le classement malin / bénin est excellent, mais **près d'un mélanome sur deux passe sous
le seuil** à 95 % de sensibilité globale. Le seuil est en effet surtout réglé par les
carcinomes et kératoses, beaucoup plus nombreux. C'est ce qui a motivé l'ajout d'ISIC
2020 et ses 584 mélanomes.

Pour comparaison, l'ancien modèle sur toutes les images de PAD-UFES-20 : AUC 0,82
(0,80 à 0,84), AUC mélanome / bénin **0,46**, c'est-à-dire pas mieux que le hasard sur le
mélanome.

## 4. Choix de la régularisation

AUC de validation selon `C`, pour le modèle PAD + ISIC (critère : moyenne des deux jeux) :

| C | PAD-UFES-20 | ISIC 2020 | Moyenne |
|---|---|---|---|
| 10⁻⁶ | 0,961 | 0,779 | 0,870 |
| 10⁻⁵ | 0,969 | 0,844 | 0,907 |
| **10⁻⁴** | **0,981** | **0,875** | **0,928** |
| 10⁻³ | 0,967 | 0,884 | 0,926 |
| 10⁻² | 0,963 | 0,871 | 0,917 |
| 10⁻¹ | 0,948 | 0,852 | 0,900 |
| 1 | 0,947 | 0,828 | 0,887 |

Pour le modèle ISIC seul : 0,855 / 0,875 / **0,895** / 0,893 / 0,877 / 0,859 / 0,836
(même grille), optimum également à `C = 10⁻⁴`.

## 5. Classifieurs non linéaires : XGBoost et réseau de neurones

Question posée : un classifieur plus souple que la régression logistique tire-t-il davantage
des embeddings de Derm Foundation ? Deux classifieurs ont été entraînés sur PAD-UFES-20 +
ISIC 2020, dans les mêmes conditions que le modèle actuel (protocole dans
[Méthodologie](04-methodologie-scientifique.md#classifieurs-de-comparaison--xgboost-et-réseau-de-neurones)) :

- **XGBoost** (arbres de décision boostés) : retenu avec une profondeur de 6 et 1 000 arbres ;
- **MLP** (réseau de neurones à une couche cachée de 128 neurones) : retenu avec
  `alpha = 0,1`.

### ISIC 2020 — dermatoscope

| Mesure | **LR (modèle actuel)** | XGBoost | MLP |
|---|---|---|---|
| AUC | **0,857 (0,826 à 0,887)** | 0,861 (0,824 à 0,895) | 0,845 (0,807 à 0,880) |
| Spécificité à 95 % de sensibilité | **57,6 %** | 42,6 % | 42,6 % |
| Mélanomes détectés au seuil à 95 % | 95,3 % | 95,3 % | 96,2 % |
| Mélanomes détectés au score 0,5 | 35,8 % | 5,7 % | 34,0 % |
| Seuil à 95 % (calculé sur le test) | 0,072 | 0,000 | 0,0001 |

### PAD-UFES-20 — smartphone

| Mesure | **LR (modèle actuel)** | XGBoost | MLP |
|---|---|---|---|
| AUC | **0,962 (0,939 à 0,980)** | 0,973 (0,953 à 0,988) | 0,968 (0,945 à 0,986) |
| Spécificité à 95 % de sensibilité | 82,6 % | 89,5 % | 88,4 % |
| AUC mélanome / bénin | **0,899** | 0,855 | 0,836 |
| Mélanomes détectés au seuil à 95 % | **6 sur 7** | 3 sur 7 | 3 sur 7 |
| Mélanomes détectés au score 0,5 | 7 sur 7 | 4 sur 7 | 6 sur 7 |
| Seuil à 95 % (calculé sur le test) | 0,871 | 0,738 | 0,973 |

### Choix des hyperparamètres

AUC de validation (moyenne des deux jeux, même critère que pour `C`) :

| XGBoost | 300 arbres | 1 000 arbres |
|---|---|---|
| Profondeur 3 | 0,925 | 0,929 |
| Profondeur 6 | 0,927 | **0,930** |

| MLP | `alpha = 0,01` | `alpha = 0,1` | `alpha = 1` |
|---|---|---|---|
| 1 couche (128) | 0,914 | **0,928** | 0,914 |
| 2 couches (256, 64) | 0,908 | 0,919 | 0,926 |

Pour comparaison, la régression logistique atteint 0,928 avec `C = 10⁻⁴`. Les trois
classifieurs sont à égalité en validation, à 0,002 près.

### Lecture

- **Aucun gain significatif** : les écarts d'AUC ne dépassent pas 0,011 et les intervalles
  de confiance se recouvrent presque entièrement, sur les deux jeux.
- **Sur le dermatoscope, la régression logistique reste la meilleure au point de
  fonctionnement de l'application** : à 95 % de sensibilité, elle écarte 58 % des lésions
  bénignes, contre 43 % pour XGBoost et le MLP. Une AUC égale ne garantit pas le même
  comportement au seuil d'alerte.
- **Les scores des modèles non linéaires s'écrasent vers 0 sur ISIC 2020** : le seuil à
  95 % y tombe à 0,000 (XGBoost) et 0,0001 (MLP), contre 0,072 pour la régression
  logistique. Signe probable qu'ils ont appris en partie à reconnaître la source de
  l'image (2 % de malins dans ISIC 2020, 78 % dans PAD-UFES-20), le raccourci que le
  critère de validation cherche justement à neutraliser. Des seuils aussi proches de 0
  seraient de plus fragiles à exporter dans l'application.
- **Sur smartphone, XGBoost et le MLP gagnent un peu de spécificité mais perdent des
  mélanomes** (3 sur 7 contre 6 sur 7). Sur 7 mélanomes, ce n'est pas significatif, mais
  cela ne plaide pas pour eux.
- **Coût** : environ 1 min 30 pour la régression logistique, 4 min 30 pour le MLP et
  24 min pour XGBoost (grille complète, 8 cœurs).

**Décision : la régression logistique reste le classifieur de l'application.** Elle fait
aussi bien, s'exporte en simples tableaux de nombres (`app/probe.npz`, sans bibliothèque
d'apprentissage dans l'image) et s'entraîne en quelques minutes.

### Pourquoi les modèles non linéaires ne font pas mieux

1. **Derm Foundation a été entraîné pour qu'une tête linéaire suffise.** Lors de son
   pré-entraînement, la dernière opération du réseau est linéaire (couche de classification,
   ou produit scalaire en apprentissage contrastif). L'entraînement pousse donc les couches
   profondes à transformer l'image jusqu'à ce que les notions utiles (pigmentation, bords,
   structures) deviennent des **directions** de l'embedding, qu'une somme pondérée suffit
   à combiner.
2. **La non-linéarité a déjà été appliquée.** L'embedding est la sortie de dizaines de
   couches non linéaires. Ajouter quelques couches de plus, ou des arbres, apporte peu
   quand la tâche est proche de celles apprises par le modèle.
3. **6 144 dimensions favorisent la séparation linéaire.** Plus la dimension est grande
   devant le nombre d'exemples, plus il est facile de séparer les classes par un hyperplan
   (théorème de Cover, la même idée que les noyaux des SVM).
4. **Peu de cas positifs : la souplesse devient du surapprentissage.** Avec 106 mélanomes
   en test ISIC et 52 dans tout PAD-UFES-20, un modèle de 1 000 arbres ou un réseau de
   neurones a assez de capacité pour apprendre les particularités des images
   d'entraînement, ou la source de l'image. La régression logistique, fortement
   régularisée, ne peut apprendre qu'une direction et reste plus robuste.

Conséquence : **le plafond vient de l'embedding, pas du classifieur.** Une information
absente de l'embedding ne peut être retrouvée par aucune tête, linéaire ou non. Les
pistes d'amélioration utiles se situent en amont (voir [section 7](#7-pistes-damélioration)).

## 6. Ce que montrent les résultats

1. **Les embeddings de Derm Foundation contiennent bien l'information de malignité**, sur
   smartphone comme sur dermatoscope : un simple classifieur linéaire suffit à dépasser
   largement un réseau spécialisé entraîné sur HAM10000.
2. **Le dermatoscope reste le cas difficile** (AUC 0,86) : distinguer un mélanome débutant
   d'un nævus atypique est réellement plus dur que séparer un carcinome d'un nævus, ce
   qui domine PAD-UFES-20.
3. **Les embeddings plafonnent autour de 0,86 sur ISIC 2020** : élargir la grille de `C`,
   se restreindre à ISIC ou remplacer la régression logistique par XGBoost ou un réseau de
   neurones ne change rien. Pour repère, les meilleures équipes du
   concours SIIM-ISIC 2020 dépassaient 0,94, avec de grands réseaux entièrement affinés,
   des ensembles de modèles et les métadonnées cliniques. Ces chiffres ont été obtenus sur
   un autre jeu de test (celui du concours) : la comparaison n'est qu'indicative.
4. **Le type de photo compte** : les seuils adaptés diffèrent d'un facteur 10 entre
   smartphone et dermatoscope. D'où le choix du type de photo dans l'interface.

## 7. Pistes d'amélioration

Déjà essayé sans gain : XGBoost et un petit réseau de neurones à la place de la régression
logistique ([section 5](#5-classifieurs-non-linéaires--xgboost-et-réseau-de-neurones)).

| Piste | Gain attendu | Coût |
|---|---|---|
| Ajouter les **métadonnées cliniques** d'ISIC (âge, sexe, localisation) au classifieur | modéré ; elles comptaient dans les meilleures solutions du concours. Un classifieur non linéaire peut alors redevenir utile, pour combiner des variables de nature différente | faible : quelques minutes de calcul, mais il faut les demander dans l'interface |
| **Moyenner les embeddings** de plusieurs recadrages ou retournements de la même image | faible à modéré | faible, mais multiplie le temps d'analyse |
| Exploiter les **autres lésions du même patient** (« vilain petit canard ») | réel en dépistage | moyen : change le mode d'utilisation (plusieurs photos) |
| **Affiner Derm Foundation** ou un autre réseau en entier | le plus élevé | élevé : GPU pendant des heures, risque de surapprentissage |
| Ajouter des jeux de données (HAM10000, BCN20000, Fitzpatrick17k…) | meilleure généralisation, meilleure couverture des peaux foncées | moyen : téléchargement, licences, embeddings |
| **Calibrer** les scores (régression isotonique, Platt) par type de photo | scores lisibles comme des probabilités | faible, mais demande un jeu de calibration dédié |
| **Validation externe** sur un jeu jamais consulté | mesure fiable de la généralisation | moyen |
