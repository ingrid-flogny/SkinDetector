---
license: other
license_name: hai-def-terms-and-cc-by-nc-4.0
license_link: https://developers.google.com/health-ai-developer-foundations/terms
language:
  - fr
library_name: numpy
pipeline_tag: image-classification
base_model: google/derm-foundation
tags:
  - dermatology
  - skin-lesion
  - melanoma
  - medical
  - derm-foundation
  - hai-def
---

# SkinDetector : classifieur de malignité sur Derm Foundation

Classifieur linéaire (régression logistique) qui attribue un **score de malignité** à une
lésion cutanée, à partir de l'**embedding Derm Foundation** de sa photo, prise au
smartphone ou au dermatoscope.

> **Avertissement.** Projet de recherche et d'apprentissage. Ce n'est **pas un dispositif
> médical** : ni marquage CE, ni validation clinique, aucun diagnostic. Un score bas
> n'exclut pas une lésion maligne. Ne pas utiliser sur de vrais patients en l'état.

## Contenu du dépôt

| Fichier | Contenu |
|---|---|
| `probe.npz` | le classifieur : uniquement des tableaux NumPy, aucun code exécuté au chargement |
| `NOTICE` | mention exigée par les conditions HAI-DEF |
| `HAI-DEF-TERMS.md` | copie des *Health AI Developer Foundations Terms of Use* |

`probe.npz` contient :

| Clé | Forme | Rôle |
|---|---|---|
| `mean`, `scale` | (6144,) | standardisation de l'embedding |
| `coef`, `intercept` | (6144,), () | régression logistique |
| `threshold_smartphone` | () | seuil d'alerte pour les photos de smartphone : 0,896 |
| `threshold_dermatoscope` | () | seuil d'alerte pour les images de dermatoscope : 0,079 |

Les seuils détectaient 95 % des lésions malignes de la partie validation de chaque jeu.

## Utilisation

Ce dépôt ne contient **pas** Derm Foundation. Il faut obtenir son propre accès à
[`google/derm-foundation`](https://huggingface.co/google/derm-foundation), et utiliser
**la même révision** que celle de l'entraînement : une autre version produirait des
embeddings que le classifieur ne connaît pas.

```python
import io

import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

path = snapshot_download(
    "google/derm-foundation",
    revision="a16a6ab4f87888948fe248136e697ed28146a1c6",
    allow_patterns=["*.pb", "variables/*"],
)
infer = tf.saved_model.load(path).signatures["serving_default"]
probe = np.load(hf_hub_download("ingridflogny/skindetector-derm-probe", "probe.npz"))

# Préparation identique à l'entraînement : RGB, 448 × 448 bicubique, encodage PNG.
image = Image.open("lesion.jpg").convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
buffer = io.BytesIO()
image.save(buffer, "PNG")
feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[buffer.getvalue()]))
example = tf.train.Example(features=tf.train.Features(feature={"image/encoded": feature}))
embedding = infer(inputs=tf.constant([example.SerializeToString()]))["embedding"].numpy()[0]

logit = ((embedding - probe["mean"]) / probe["scale"]) @ probe["coef"] + probe["intercept"]
score = 1 / (1 + np.exp(-logit))
alert = score >= probe["threshold_dermatoscope"]  # ou threshold_smartphone
```

Le score **n'est pas une probabilité calibrée** (classes rééquilibrées à l'entraînement) :
il se lit uniquement par rapport au seuil du type de photo, qui diffère d'un facteur 10
entre smartphone et dermatoscope.

## Entraînement

- **Embeddings** : Derm Foundation, révision `a16a6ab4…`, images en 448 × 448.
- **Données** : PAD-UFES-20 (2 298 photos de smartphone) et ISIC 2020 (33 126 images de
  dermatoscope), découpées en entraînement / validation / test **par patient**.
- **Modèle** : `StandardScaler` + `LogisticRegression(C=1e-4, class_weight="balanced")`
  (scikit-learn), exporté en tableaux NumPy.
- **Choix de `C`** : moyenne des AUC de validation calculées séparément sur chaque jeu,
  pour ne pas récompenser la simple reconnaissance de la source de l'image.

XGBoost et un petit réseau de neurones ont été essayés sur les mêmes embeddings : aucun
gain significatif, et une moins bonne spécificité au seuil d'alerte sur le dermatoscope.

## Résultats (parties test, jamais vues à l'entraînement)

Intervalles de confiance à 95 % par bootstrap.

| Mesure | ISIC 2020 (dermatoscope) | PAD-UFES-20 (smartphone) |
|---|---|---|
| Images de test | 6 162, dont 106 mélanomes | 453, dont 367 malignes et 7 mélanomes |
| AUC | 0,857 (0,826 à 0,887) | 0,962 (0,939 à 0,980) |
| Spécificité à 95 % de sensibilité | 57,6 % | 82,6 % |
| AUC mélanome / bénin | 0,857 | 0,899 |

Le seuil des mesures de spécificité est fixé sur le test lui-même : ce sont des mesures de
capacité, légèrement optimistes. Les seuils livrés dans `probe.npz` viennent de la
validation.

## Limites

- **Pas de validation externe** : le test vient des mêmes centres et appareils que
  l'entraînement.
- **Peu de mélanomes au smartphone** (7 en test) : les mesures sur le mélanome au
  smartphone sont très incertaines.
- **Peaux majoritairement claires** : performances inconnues, et probablement moins
  bonnes, sur les phototypes V et VI.
- **Une photo, une lésion** : aucune information clinique (âge, antécédents, évolution),
  aucun contrôle de la qualité d'image.
- **Type de photo déclaré** : un mauvais choix applique un seuil faux d'un facteur 10.
- **Chevauchement possible** entre les données de test et le préentraînement de Derm
  Foundation, qui a utilisé des images publiques non listées.

## Licence et conditions d'utilisation

`probe.npz` est un **Model Derivative** de HAI-DEF au sens des *Health AI Developer
Foundations Terms of Use* : il a été créé par Ingrid Flogny à partir des sorties de
Derm Foundation, et n'est ni distribué ni approuvé par Google.

HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms of Use
found at https://developers.google.com/health-ai-developer-foundations/terms. Une copie
de ces conditions est jointe ([`HAI-DEF-TERMS.md`](HAI-DEF-TERMS.md)).

En téléchargeant ou en utilisant ce classifieur, vous acceptez ces conditions et vous
vous engagez à respecter leurs **restrictions d'usage (section 3.2)**, qui s'appliquent
comme une clause des présentes conditions d'utilisation : ne pas l'utiliser

- pour les usages restreints décrits dans la *Health AI Developer Foundations Prohibited
  Use Policy* (https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy) ;
- pour un usage qui pourrait amener une autorité de santé à considérer Google comme
  fabricant d'un dispositif médical ;
- en violation des lois et règlements applicables.

Tout **usage clinique** (diagnostic ou traitement de patients, y compris en recherche)
demande, le cas échéant, l'autorisation de l'autorité de santé compétente.

**Usage non commercial uniquement** : le classifieur a été entraîné sur ISIC 2020, publié
sous licence CC BY-NC 4.0.

## Citations

- **PAD-UFES-20** : Pacheco A. G. C. et al., *PAD-UFES-20: a skin lesion dataset composed
  of patient data and clinical images collected from smartphones*, Data in Brief 32,
  106221, 2020. Licence CC BY 4.0.
- **ISIC 2020** : Rotemberg V. et al., *A patient-centric dataset of images and metadata
  for identifying melanomas using clinical context*, Scientific Data 8, 34, 2021.
  Licence CC BY-NC 4.0.
- **Derm Foundation** : Google, *Health AI Developer Foundations*,
  https://huggingface.co/google/derm-foundation.
