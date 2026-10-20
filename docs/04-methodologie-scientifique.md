# 04 · Méthodologie scientifique

Ce document décrit comment le modèle de SkinDetector a été construit et évalué : la
question posée, les données, les choix méthodologiques et leurs raisons. Les résultats
chiffrés sont dans [05 · Résultats](05-resultats.md), les commandes pour tout refaire
dans [06 · Reproduire l'entraînement](06-reproduire-l-entrainement.md).

## 1. Question posée

**À partir d'une seule photo d'une lésion cutanée, estimer si elle est maligne.**

- Tâche : **classification binaire**, maligne / bénigne.
- Est **maligne** : mélanome, carcinome basocellulaire, carcinome épidermoïde, et, par
  décision du projet, **kératose actinique**. Cette dernière est une lésion
  précancéreuse qui peut évoluer en carcinome épidermoïde et qui se traite : la manquer
  a un coût clinique. Ce choix rend la tâche plus proche de « faut-il montrer cette
  lésion à un dermatologue ? » que de « est-ce déjà un cancer ? ».
- Priorité : **ne pas manquer de lésion maligne**, en particulier de mélanome. On
  évalue donc le modèle surtout à **sensibilité élevée** (95 %) : combien de lésions
  bénignes faut-il signaler à tort pour attraper 95 % des malignes ?
- Deux contextes d'usage : photo de **smartphone** (médecin généraliste, patient) et
  image de **dermatoscope** (dermatologue).

## 2. Démarche générale

Trois approches étaient envisageables :

| Approche | Données nécessaires | Coût de calcul | Choix |
|---|---|---|---|
| Entraîner un réseau de zéro | des centaines de milliers d'images | très élevé | écartée |
| Affiner entièrement un réseau préentraîné | des dizaines de milliers d'images | élevé (GPU pendant des heures, à chaque essai) | écartée pour l'instant |
| **Modèle de fondation figé + classifieur simple** (*linear probe*) | quelques milliers d'images | un passage du modèle par image, puis quelques minutes de CPU | **retenue** |

L'approche retenue se déroule en deux temps :

1. **Extraction des embeddings** : chaque image passe une fois dans Derm Foundation, qui
   la résume en 6 144 nombres. C'est l'étape coûteuse, faite une fois pour toutes.
2. **Entraînement d'une régression logistique** sur ces embeddings : quelques minutes
   sur un ordinateur portable, ce qui permet de tester beaucoup de variantes.

Avantages : peu de données suffisent, chaque expérience coûte quelques minutes, et le
résultat mesure directement ce que les embeddings de Derm Foundation contiennent sur la
malignité. Inconvénient : un classifieur linéaire plafonne, voir [Résultats](05-resultats.md#5-ce-que-montrent-les-résultats).

Le modèle de départ (EfficientNet-B0 entraîné sur HAM10000, récupéré tel quel sur
Hugging Face) a été mesuré avec exactement le même protocole, pour servir de **référence
à battre**.

## 3. Données

### 3.1 PAD-UFES-20 — photos de smartphone

| | |
|---|---|
| Référence | Pacheco et al., *PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones*, Data in Brief 32, 106221, 2020 |
| Origine | programme de dermatologie de l'Université fédérale d'Espírito Santo (Brésil) |
| Images | **2 298** photos cliniques en gros plan (`clinical: close-up`), prises au smartphone, de tailles variées |
| Patients | **1 373** |
| Confirmation du diagnostic | histopathologie pour 1 342 images (58 %), consensus d'experts sur image pour 956 (42 %) |
| Licence | CC BY 4.0 |
| Téléchargement | archive ISIC, collection 406 (`training/download_pad_ufes_20.py`) |

Répartition des diagnostics :

| Diagnostic | Images | Étiquette |
|---|---|---|
| Carcinome basocellulaire | 845 | maligne |
| Kératose actinique (*solar or actinic keratosis*) | 730 | maligne (choix du projet) |
| Nævus | 244 | bénigne |
| Kératose séborrhéique | 235 | bénigne |
| Carcinome épidermoïde | 192 | maligne |
| Mélanome | 52 | maligne |
| **Total** | **2 298** | **1 819 malignes (79 %)** |

Particularités : jeu **majoritairement malin**, dominé par les carcinomes et les
kératoses actiniques ; **peu de mélanomes** (52) ; population brésilienne recrutée dans
un service de dermatologie (pas un dépistage en population générale).

### 3.2 ISIC 2020 — images de dermatoscope

| | |
|---|---|
| Référence | Rotemberg et al., *A patient-centric dataset of images and metadata for identifying melanomas using clinical context*, Scientific Data 8, 34, 2021 |
| Origine | concours SIIM-ISIC 2020 ; plusieurs centres (Europe, États-Unis, Australie) |
| Images | **33 126** images de dermatoscope (partie « Training » du concours, la seule dont les diagnostics sont publics) |
| Patients | **2 056** (16 images par patient en moyenne) |
| Métadonnées | sexe, âge approximatif (médiane 50 ans), localisation anatomique (tronc 51 %, membres inférieurs 25 %, membres supérieurs 15 %…) |
| Licence | CC BY-NC 4.0 (usage non commercial) |
| Téléchargement | `https://isic-challenge-data.s3.amazonaws.com/2020/` : `ISIC_2020_Training_JPEG.zip` (23 Go) et `ISIC_2020_Training_GroundTruth_v2.csv` |

Répartition des diagnostics :

| Diagnostic | Images | Étiquette |
|---|---|---|
| inconnu (*unknown*, lésion jugée bénigne sans diagnostic précis) | 27 124 | bénigne |
| nævus | 5 193 | bénigne |
| **mélanome** | **584** | **maligne** |
| kératose séborrhéique | 135 | bénigne |
| lentigo, kératose lichénoïde, lentigo solaire, autres | 90 | bénigne |
| **Total** | **33 126** | **584 malignes (1,8 %)** |

Particularités : jeu **très déséquilibré** (1,8 % de malins, comme en dépistage) ; seul le
mélanome y est compté comme malin ; beaucoup de lésions bénignes n'ont pas de diagnostic
précis (« unknown ») ; plusieurs lésions par patient, ce qui permet de comparer une lésion
aux autres du même patient (non exploité ici).

### 3.3 Pourquoi ces deux jeux

Ils couvrent les **deux contextes d'usage** visés (smartphone et dermatoscope) et sont
**disjoints** : PAD-UFES-20 est surtout composé de carcinomes et de kératoses, ISIC 2020
de nævus et de mélanomes. Les combiner apprend au classifieur l'ensemble des lésions
malignes visées, sur les deux types d'images. Les deux jeux sont publics, sous des
licences connues, et donnent l'identifiant du patient, indispensable pour un découpage
honnête.

HAM10000, qui a servi à entraîner l'ancien modèle, n'a pas été utilisé pour l'évaluer :
cela aurait avantagé ce modèle.

## 4. Étiquettes (`training/build_manifest.py`)

Toutes les images sont décrites dans un fichier unique, `training/data/manifest.csv` :

| Colonne | Contenu |
|---|---|
| `dataset` | `pad_ufes_20` ou `isic2020` |
| `image_id`, `path` | identifiant ISIC de l'image et chemin du fichier |
| `patient_id` | identifiant du patient |
| `diagnosis` | diagnostic d'origine |
| `malignant` | 1 si maligne (étiquette apprise) |
| `melanoma` | 1 si mélanome (pour des mesures dédiées) |
| `split` | `train`, `val` ou `test` |

Règles :

- **PAD-UFES-20** : `malignant = 1` si le diagnostic de premier niveau de l'archive ISIC
  (`diagnosis_1`) vaut `Malignant` ou `Indeterminate`. Dans cette archive, les
  kératoses actiniques sont classées `Indeterminate` : c'est ainsi qu'elles sont
  comptées malignes. `melanoma = 1` si le diagnostic détaillé commence par « Melanoma ».
- **ISIC 2020** : `malignant = melanoma = target` (colonne officielle du concours).

## 5. Découpage entraînement / validation / test

### Par patient

Toutes les images d'un même patient vont dans la **même partie**. Sinon, le modèle
pourrait reconnaître en test la peau, l'éclairage ou la lésion d'un patient déjà vu à
l'entraînement, et les résultats seraient surestimés. C'est particulièrement important
pour ISIC 2020 (16 images par patient en moyenne) et pour PAD-UFES-20 (plusieurs photos
d'une même lésion).

### Par empreinte, donc stable

La partie d'un patient est déterminée par l'empreinte SHA-256 de son identifiant :

```python
bucket = int(sha256(patient_id).hexdigest(), 16) % 100
split  = "test" if bucket < 20 else "val" if bucket < 30 else "train"
```

Soit environ **70 % entraînement, 10 % validation, 20 % test** des patients. Ce découpage
ne dépend d'aucun tirage aléatoire : il est identique à chaque exécution, et ajouter des
données ne déplace pas les patients existants.

### Effectifs obtenus

| Jeu | Partie | Patients | Images | Malignes | Mélanomes |
|---|---|---|---|---|---|
| PAD-UFES-20 | train | 961 | 1 610 | 1 268 | 39 |
| PAD-UFES-20 | val | 140 | 235 | 184 | 6 |
| PAD-UFES-20 | test | 272 | 453 | 367 | 7 |
| ISIC 2020 | train | 1 442 | 23 565 | 402 | 402 |
| ISIC 2020 | val | 207 | 3 399 | 76 | 76 |
| ISIC 2020 | test | 407 | 6 162 | 106 | 106 |

Rôle de chaque partie :

- **train** : apprendre les poids du classifieur ;
- **val** : choisir la régularisation et fixer les seuils d'alerte ;
- **test** : mesurer les performances finales, et rien d'autre.

## 6. Préparation des images et embeddings (`training/embed.py`)

Pour chaque image :

1. ouverture avec Pillow, conversion en RGB ;
2. **redimensionnement direct en 448 × 448** pixels, interpolation **bicubique**, sans
   recadrage (les proportions ne sont pas conservées) ;
3. encodage **PNG** (sans perte) dans un `tf.train.Example` ;
4. passage dans Derm Foundation : un vecteur de 6 144 nombres en `float32`.

Aucune augmentation de données (rotation, recadrage, couleurs) : le classifieur est
linéaire et les embeddings sont calculés une seule fois.

L'application applique **exactement la même préparation** (`app/derm_foundation.py`).
Elle redresse en plus la photo selon son orientation EXIF, ce que les images des jeux
publics n'exigent pas. La concordance a été vérifiée : l'application retrouve les scores
de l'entraînement à la quatrième décimale près (voir [Modèles](03-modeles.md#vérification-de-bout-en-bout)).

Les embeddings sont enregistrés dans `training/data/embeddings/derm_foundation/<jeu>.npz`
(clés `image_id` et `embedding`). Le calcul est relançable : il reprend où il s'est
arrêté.

| Jeu | Images | Où | Durée |
|---|---|---|---|
| PAD-UFES-20 | 2 298 | poste local, 8 cœurs ARM64 | environ 1 h 30 (environ 2,5 s par image) |
| ISIC 2020 | 33 126 | VM Google Cloud, GPU NVIDIA L4 | 1 h 15 de calcul (7,9 images/s) |

Contrôles du fichier ISIC 2020 rapatrié : empreinte SHA-256 identique à celle calculée
sur la VM, 33 126 vecteurs de 6 144 nombres, aucune valeur manquante, norme moyenne
(33,3) cohérente avec celle de PAD-UFES-20 (33,9).

## 7. Classifieur (`training/train_probe.py`)

### Modèle

```python
make_pipeline(
    StandardScaler(),
    LogisticRegression(C=C, class_weight="balanced", max_iter=2000),
)
```

- **Standardisation** : chaque dimension est centrée-réduite sur l'entraînement.
- **Régression logistique** régularisée L2 (solveur lbfgs).
- **`class_weight="balanced"`** : chaque classe pèse autant dans l'apprentissage, quel que
  soit son effectif. Sans cela, avec 1,8 % de malins dans ISIC 2020, le modèle
  apprendrait surtout à répondre « bénin ». Conséquence : le score n'est **pas une
  probabilité calibrée** (il surestime le risque réel) ; il se lit par rapport à un seuil.

### Choix de la régularisation

`C` (inverse de la force de régularisation) est cherché dans
`{10⁻⁶, 10⁻⁵, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹, 1}`. Pour chaque valeur : entraînement sur `train`,
AUC sur `val`.

Le critère est la **moyenne des AUC calculées séparément sur chaque jeu**, et non l'AUC
sur la validation mélangée. Raison : 78 % des images de PAD-UFES-20 sont malignes, contre
2 % pour ISIC 2020. Sur les deux jeux mélangés, un modèle qui reconnaîtrait simplement
« photo de smartphone » obtiendrait une bonne AUC sans rien comprendre aux lésions.
Calculer l'AUC jeu par jeu neutralise ce raccourci.

Valeur retenue : **`C = 10⁻⁴`**, une forte régularisation, attendue avec 6 144 variables.
Les valeurs voisines donnent de moins bons résultats des deux côtés : l'optimum n'est pas
au bord de la grille.

Le modèle final est celui entraîné sur `train` avec ce `C`. La validation n'est pas
réintégrée à l'entraînement : elle sert aussi à fixer les seuils.

## 8. Seuils d'alerte par type de photo (`training/export_probe.py`)

Un seul seuil ne peut pas convenir aux deux types de photo : les scores de PAD-UFES-20
sont globalement hauts (jeu majoritairement malin), ceux d'ISIC 2020 très bas. Un seuil
est donc fixé **par type de photo**, sur la partie **validation** du jeu correspondant :

> le seuil est le score le plus élevé qui laisse encore au moins **95 %** des lésions
> malignes de validation au-dessus de lui.

```python
positives = sort(scores[labels == 1])
threshold = positives[floor((1 - 0.95) × len(positives))]
```

| Type de photo | Jeu de validation | Malignes | Seuil |
|---|---|---|---|
| smartphone | PAD-UFES-20 val | 184 | **0,8963** |
| dermatoscope | ISIC 2020 val | 76 | **0,0785** |

Avec 76 mélanomes de validation, ce seuil est imprécis : il repose sur les 4 scores les
plus bas. Sa sensibilité réelle sur de nouvelles images peut s'écarter de 95 %.

## 9. Métriques (`training/metrics.py`)

Toutes les mesures sont calculées par le même code, pour tous les modèles.

| Mesure | Définition | Pourquoi |
|---|---|---|
| **AUC** | probabilité qu'une lésion maligne tirée au hasard ait un score plus élevé qu'une bénigne (statistique de Mann-Whitney, ex æquo comptés pour moitié) | qualité du classement, indépendante de tout seuil et de la proportion de malins |
| **IC 95 % de l'AUC** | percentiles 2,5 et 97,5 de l'AUC sur **1 000 rééchantillonnages bootstrap** des images (graine 0) | incertitude due à la taille de l'échantillon |
| **Seuil à 95 % de sensibilité** | voir section 8, calculé ici sur le jeu évalué lui-même | point de fonctionnement clinique visé |
| **Spécificité à 95 % de sensibilité** | part des lésions bénignes sous ce seuil | combien de fausses alertes coûte la sensibilité voulue |
| **AUC mélanome / bénin** | AUC calculée en ne gardant que les mélanomes et les lésions bénignes | le modèle repère-t-il le mélanome, et pas seulement les carcinomes ? |
| **Sensibilité au mélanome** | part des mélanomes au-dessus du seuil à 95 %, puis au-dessus de 0,5 | le cas le plus grave est-il détecté ? |

Deux précautions de lecture :

- La spécificité « à 95 % de sensibilité » publiée dans les résultats utilise un seuil
  calculé **sur le jeu de test lui-même**. C'est une mesure de capacité du modèle, un peu
  optimiste. L'application, elle, utilise les seuils de validation (section 8).
- Le seuil de 0,5 n'a pas de sens clinique ici (scores non calibrés) ; il est donné à
  titre de comparaison.

Chaque évaluation est enregistrée dans `training/results/<modèle>__<jeu>__<partie>.json`.

## 10. Contrôles complémentaires

### Validation croisée (`training/cross_validate_probe.py`)

La partie test de PAD-UFES-20 ne compte que 7 mélanomes. Pour une mesure plus stable, le
classifieur entraîné sur PAD-UFES-20 seul a aussi été évalué en **validation croisée à
5 plis par patient** : chaque image reçoit le score d'un modèle entraîné sans son patient
(plis fixés par l'empreinte SHA-256 de `"cv:" + patient_id`, `C = 0,1`). Toutes les
images, dont les 52 mélanomes, comptent alors dans la mesure.

### Recherche de raccourcis (`training/check_shortcuts.py`)

Un modèle peut « tricher » si les images malignes et bénignes n'ont pas été produites de
la même façon (appareil, résolution, compression). Test : peut-on deviner l'étiquette à
partir de la seule taille de l'image ou du poids du fichier ?

| Indice (PAD-UFES-20) | AUC |
|---|---|
| nombre de pixels | 0,569 |
| poids du fichier | 0,594 |

Ces valeurs, proches de 0,5, écartent un raccourci grossier lié au format des images.

### Concordance application / entraînement

Voir [Modèles](03-modeles.md#vérification-de-bout-en-bout) : les scores servis par
l'application sont identiques à ceux de l'évaluation.

## 11. Chronologie des expériences

| # | Expérience | Résultat principal |
|---|---|---|
| 1 | Référence : EfficientNet-B0 / HAM10000 sur PAD-UFES-20 | AUC 0,84 (test) |
| 2 | Derm Foundation + LR sur PAD-UFES-20 seul (`C = 0,1`) | AUC 0,97 (test), 0,98 (validation croisée) |
| 3 | Recherche de raccourcis sur PAD-UFES-20 | aucun raccourci de format |
| 4 | Référence sur ISIC 2020 | AUC 0,64 (test) |
| 5 | Derm Foundation + LR sur PAD-UFES-20 + ISIC 2020 | AUC 0,96 (PAD) et 0,86 (ISIC) |
| 6 | Grille de `C` élargie (10⁻⁶ à 1) | optimum confirmé à 10⁻⁴ |
| 7 | Derm Foundation + LR sur ISIC 2020 seul | AUC 0,86 (ISIC) : combiner les jeux ne coûte rien |
| 8 | Export, seuils de validation, intégration | scores de l'application identiques |

La partie test a été consultée à chaque expérience. Aucun hyperparamètre n'a été choisi
sur elle (seulement sur la validation), mais le choix entre les variantes 5 et 7 a été
fait en connaissant les résultats de test : un léger biais d'optimisme est possible. Une
évaluation vraiment indépendante demanderait un jeu externe jamais consulté
(voir [Limites](07-licences-conformite-et-limites.md#4-limites-scientifiques)).

## 12. Reproductibilité

| Élément | Comment il est figé |
|---|---|
| Derm Foundation | révision Hugging Face `a16a6ab4…` |
| Ancien modèle | révision Hugging Face `9429e5a4…` |
| Découpage | empreinte SHA-256 de l'identifiant patient, sans aléa |
| Bootstrap | graine 0 |
| Dépendances | `uv.lock` (application) et `training/uv.lock` (TensorFlow 2.21, scikit-learn 1.9, NumPy 2.5) |
| Environnement | images Docker construites depuis ces fichiers |
| Embeddings ISIC 2020 | empreinte SHA-256 `efc25f1e06d5ba88f9811c3047369dd5e8b0c71b47c551c97691c136b08f2340` |
| Résultats | fichiers JSON versionnés dans `training/results/` |

Les embeddings calculés sur GPU diffèrent de ceux du CPU d'au plus 0,0016 (arrondis de
calcul) : relancer le calcul sur un autre matériel peut faire varier la troisième
décimale des résultats.
