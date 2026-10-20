# 03 · Modèles

SkinDetector contient deux modèles interchangeables (voir [Architecture](02-architecture-et-api.md#le-modèle-est-un-composant-remplaçable)) :

| Identifiant (`SKIN_MODEL`) | Statut | Composition |
|---|---|---|
| `derm-foundation-logreg` | **par défaut** | Derm Foundation (Google) + régression logistique entraînée dans ce dépôt |
| `ham10000-efficientnet-b0` | conservé, en option | EfficientNet-B0 entraîné sur HAM10000 par un tiers |

## 1. Derm Foundation

### Fiche d'identité

| | |
|---|---|
| Dépôt Hugging Face | [`google/derm-foundation`](https://huggingface.co/google/derm-foundation) (accès contrôlé) |
| Révision utilisée | `a16a6ab4f87888948fe248136e697ed28146a1c6` (figée dans `app/derm_foundation.py`) |
| Auteur | Google, collection *Health AI Developer Foundations* (HAI-DEF) |
| Version | 1.0.0, modèle créé le 19 décembre 2023, entraîné avec JAX |
| Architecture | réseau convolutif **BiT-M ResNet101x3** (*Big Transfer*) |
| Format | TensorFlow SavedModel (`saved_model.pb` + `variables/`, environ 1,1 Go) |
| Entrée | image PNG de **448 × 448** pixels, encodée dans un `tf.train.Example` sous la clé `image/encoded` |
| Sortie | **embedding** de **6 144** nombres (clé `embedding` de la signature `serving_default`) |
| Conditions | *Health AI Developer Foundations Terms of Use* |

### Comment il a été entraîné (d'après la fiche du modèle)

1. **Préentraînement contrastif** sur un grand nombre de paires image-texte publiques
   issues d'internet, dans l'esprit de ConVIRT (Zhang et al., 2020).
2. **Affinage supervisé** sur des images de dermatologie : téléconsultations des
   États-Unis et de Colombie, un jeu de cancers cutanés australien, et des « images
   publiques supplémentaires ». Les appareils incluent smartphones, autres appareils
   photo et dermatoscopes.

La fiche ne précise ni le nombre d'images ni la liste des images publiques utilisées.
Conséquence pour notre évaluation : il est **possible** que des images de PAD-UFES-20 ou
d'ISIC aient été vues par Derm Foundation pendant son affinage, ce qui gonflerait les
résultats (voir [Limites](07-licences-conformite-et-limites.md#4-limites-scientifiques)).

### Ce qu'il fait et ne fait pas

Derm Foundation **ne pose aucun diagnostic** : il transforme une image en un résumé
chiffré (l'embedding) où des lésions visuellement semblables ont des vecteurs proches.
Un petit modèle entraîné sur ces vecteurs suffit ensuite pour une tâche précise. Google
indique qu'un classifieur linéaire sur ces embeddings gagne 10 à 15 points de précision
par rapport au même classifieur sur un modèle BiT-M générique.

La fiche signale que la qualité baisse sur des photos extrêmes (trop claires, trop
sombres) et que les données d'affinage, limitées en pays et en contextes, peuvent mal se
généraliser à d'autres populations.

### Particularité technique : un modèle exporté pour le CPU seulement

Le modèle a été converti de JAX vers TensorFlow avec `jax2tf`, pour le seul CPU. Sur un
GPU, il refuse de s'exécuter :

```
The current platform CUDA is not among the platforms required by the module: [CPU]
```

`training/embed.py` (fonction `load_model`) lève ce verrou : il ajoute `"platform"` à
l'attribut `disabled_checks` des nœuds `XlaCallModule` du graphe, dans une copie du
modèle, l'original restant intact. TensorFlow compile alors le même calcul pour le GPU.
Le script `training/cloud/check_gpu.py` a vérifié que les embeddings obtenus sur GPU sont
identiques à ceux du CPU (similarité cosinus de 1,00000, écart maximal de 0,0016, dû aux
arrondis). L'application tourne sur CPU et n'a pas besoin de cette modification.

## 2. Le classifieur : `app/probe.npz`

Le classifieur est une **régression logistique** posée sur l'embedding, après
standardisation. Il est stocké dans un simple fichier NumPy, produit par
`training/export_probe.py` :

| Clé | Forme | Contenu |
|---|---|---|
| `mean` | 6 144 | moyenne de chaque dimension de l'embedding sur l'entraînement |
| `scale` | 6 144 | écart-type de chaque dimension |
| `coef` | 6 144 | poids de la régression logistique |
| `intercept` | 1 | biais |
| `threshold_smartphone` | 1 | seuil d'alerte pour les photos de smartphone : **0,8963** |
| `threshold_dermatoscope` | 1 | seuil d'alerte pour les images de dermatoscope : **0,0785** |

Calcul du score pour un embedding `x` :

```
z     = Σᵢ ((xᵢ − meanᵢ) / scaleᵢ) × coefᵢ + intercept
score = 1 / (1 + e^(−z))
```

`export_probe.py` vérifie que ce calcul refait à la main donne exactement les scores de
scikit-learn (écart maximal mesuré : 0). Ce format a été préféré au `pickle` de
scikit-learn parce qu'il ne contient que des nombres : l'ouvrir ne peut exécuter aucun
code, et l'application n'a pas besoin de scikit-learn.

Entraînement : régularisation `C = 0,0001`, classes rééquilibrées
(`class_weight="balanced"`), 25 175 images de PAD-UFES-20 et d'ISIC 2020. Détails dans
[Méthodologie](04-methodologie-scientifique.md).

### Vérification de bout en bout

Quatre images de test ont été analysées par l'application en marche ; les scores sont
identiques, à la quatrième décimale, à ceux calculés à partir des embeddings
d'entraînement :

| Image | Type | Diagnostic | Score attendu | Score de l'application |
|---|---|---|---|---|
| ISIC_9118085 | smartphone | kératose actinique (maligne) | 0,9939 | 0,9939 |
| ISIC_7286812 | smartphone | nævus (bénin) | 0,1574 | 0,1574 |
| ISIC_0074311 | dermatoscope | bénin | 0,0266 | 0,0266 |
| ISIC_0149568 | dermatoscope | mélanome | 0,0718 | 0,0718 (sous le seuil : manqué) |

Le dernier cas illustre la limite d'un seuil à 95 % de sensibilité : environ un mélanome
sur vingt passe sous le seuil.

## 3. Ancien modèle : EfficientNet-B0 / HAM10000

C'était le premier modèle de SkinDetector. Il est conservé pour pouvoir y revenir et
comme point de comparaison (voir [Résultats](05-resultats.md)).

| | |
|---|---|
| Dépôt Hugging Face | [`pthennakoon25/skinvision-ai-model`](https://huggingface.co/pthennakoon25/skinvision-ai-model) |
| Révision utilisée | `9429e5a4ef81bb90064d8cdacc5895c4fc442cf7` (figée dans `app/ham10000.py`) |
| Fichier | `best_model.pth` (PyTorch) |
| Licence | MIT |
| Statut | « Educational project — not a medical diagnostic tool » (fiche du modèle) |
| Architecture | EfficientNet-B0 (`timm`), 7 sorties |
| Données d'entraînement | HAM10000 (Tschandl et al., 2018) : images de dermatoscope |
| Entrée | 224 × 224, interpolation bilinéaire, normalisation ImageNet |
| Performances annoncées | sur 1 431 images de test HAM10000 : exactitude 82,4 %, F1 macro 0,663, AUC macro 0,953 |

Les 7 classes et leur statut dans SkinDetector :

| Code | Lésion | Comptée maligne |
|---|---|---|
| `akiec` | kératose actinique / carcinome intra-épithélial | oui |
| `bcc` | carcinome basocellulaire | oui |
| `mel` | mélanome | oui |
| `bkl` | kératose bénigne | non |
| `df` | dermatofibrome | non |
| `nv` | nævus mélanocytaire | non |
| `vasc` | lésion vasculaire | non |

Le score de malignité est la **somme des probabilités des trois classes malignes**.

Détails de chargement : le fichier enveloppe le réseau dans un module `backbone` ; les
clés sont donc renommées (`backbone.` retiré) avant `load_state_dict`. Le chargement se
fait avec `torch.load(…, weights_only=True)`, qui refuse tout objet autre que des
tenseurs : le fichier vient d'un tiers et un `pickle` arbitraire pourrait exécuter du
code.

Mesuré sur nos jeux de test, il est nettement moins bon que Derm Foundation (AUC 0,64 sur
ISIC 2020, 0,84 sur PAD-UFES-20) : il a été entraîné sur des images de dermatoscope
cadrées d'une seule source et se généralise mal.

## 4. Publier le classifieur sur Hugging Face

Le classifieur est publié sur
[`ingridflogny/skindetector-derm-probe`](https://huggingface.co/ingridflogny/skindetector-derm-probe),
en accès libre ; Derm Foundation lui-même n'est pas republié. La fiche du modèle
explique comment l'utiliser avec la révision figée de Derm Foundation. Les conditions
respectées (fichier `NOTICE`, copie des conditions HAI-DEF, usage non commercial à cause
d'ISIC 2020) sont détaillées dans
[Licences](07-licences-conformite-et-limites.md#2-publier-le-classifieur).

## Références

- Kolesnikov et al., *Big Transfer (BiT): General Visual Representation Learning*,
  arXiv:1912.11370, 2019.
- Zhang et al., *Contrastive Learning of Medical Visual Representations from Paired
  Images and Text*, arXiv:2010.00747, 2020.
- Tschandl, Rosendahl, Kittler, *The HAM10000 dataset*, Scientific Data 5, 180161, 2018.
- Tan, Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019.
