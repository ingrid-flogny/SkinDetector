# 02 · Architecture et API

## 1. Vue d'ensemble

SkinDetector est une seule application web : un serveur **FastAPI** qui sert la page
web et l'API, et qui exécute le modèle dans le même processus.

```
Navigateur ──photo + type──▶ FastAPI (app/main.py)
                               │  redresse la photo (EXIF), la valide
                               ▼
                       get_classifier()  ◀── SKIN_MODEL
                               │
              ┌────────────────┴─────────────────┐
              ▼                                  ▼
  DermFoundationLogReg (défaut)        Ham10000EfficientNet (option)
  app/derm_foundation.py               app/ham10000.py
   1. réduction 448×448, PNG            1. réduction 224×224
   2. Derm Foundation → 6 144 nombres   2. EfficientNet-B0 → 7 probabilités
   3. régression logistique → score     3. somme des classes malignes → score
   4. seuil d'alerte du type de photo
              └────────────────┬─────────────────┘
                               ▼
                  SkinAnalysis (JSON) ──▶ Navigateur
```

### Le modèle est un composant remplaçable

`app/skin.py` définit le contrat que tout modèle doit respecter :

```python
class SkinClassifier(Protocol):
    def analyze(self, image: Image.Image, source: Source) -> SkinAnalysis: ...
```

Pour brancher un nouveau modèle : écrire une classe qui implémente `analyze`, l'ajouter
au dictionnaire `CLASSIFIERS`, puis la sélectionner avec la variable d'environnement
`SKIN_MODEL`. Ni l'API ni l'interface ne changent. Chaque modèle importe ses dépendances
seulement s'il est choisi : l'image par défaut ne contient pas PyTorch.

### Chargement au démarrage

Le modèle est chargé dans le `lifespan` de FastAPI, avant que le serveur accepte des
requêtes (environ 30 secondes, plus le téléchargement au tout premier démarrage). La
première analyse est ainsi aussi rapide que les suivantes : 2 à 4 secondes sur CPU.

## 2. API HTTP

### `POST /api/analyses`

Analyse une photo. Requête `multipart/form-data` :

| Champ | Type | Obligatoire | Description |
|---|---|---|---|
| `image` | fichier | oui | photo de la lésion, dans un format lisible par Pillow (JPEG, PNG, WebP, HEIF selon l'installation…) |
| `source` | texte | oui | `smartphone` ou `dermatoscope` : type d'appareil, qui détermine le seuil d'alerte |
| `crop_x`, `crop_y` | entier ≥ 0 | non | coin supérieur gauche du cadre à analyser, en pixels de la photo redressée (orientation EXIF appliquée) |
| `crop_width`, `crop_height` | entier > 0 | non | taille du cadre, en pixels |

Les quatre champs du cadre vont ensemble : tous ou aucun. Sans cadre, toute la photo est
analysée. Avec un cadre, la zone est découpée **dans la photo d'origine, en pleine
résolution**, puis analysée comme une photo entière (redimensionnée en 448 × 448). Le
modèle ayant été entraîné sur des gros plans, le cadre sert à isoler une lésion sur une
photo plus large.

Réponse `200`, objet `SkinAnalysis` :

```json
{
  "model": "Derm Foundation + régression logistique (PAD-UFES-20 + ISIC 2020)",
  "malignancy_score": 0.9939,
  "alert_threshold": 0.8963,
  "classes": []
}
```

| Champ | Description |
|---|---|
| `model` | nom lisible du modèle qui a produit le score |
| `malignancy_score` | score de 0 à 1, croissant avec le risque. **Ce n'est pas une probabilité** : il dépend de la façon dont le classifieur a été entraîné (classes rééquilibrées) et ne s'interprète que par rapport au seuil |
| `alert_threshold` | seuil qui détectait 95 % des lésions malignes en validation pour ce type de photo ; `null` pour un modèle qui n'en fournit pas |
| `classes` | détail par type de lésion (code, libellé, probabilité, malin ou non) ; vide pour Derm Foundation, 7 entrées pour HAM10000 |

Erreurs :

| Code | Cause |
|---|---|
| `422` | image illisible, champ manquant, `source` différent de `smartphone` / `dermatoscope`, cadre incomplet, de taille nulle ou qui dépasse de la photo |
| `500` | erreur du modèle (ne devrait pas arriver avec une image valide) |

Exemple :

```bash
curl -F "image=@lesion.jpg" -F "source=dermatoscope" http://localhost:8081/api/analyses
# Seulement un cadre de 300 × 300 pixels à partir du point (1200, 800)
curl -F "image=@main.jpg" -F "source=smartphone" -F crop_x=1200 -F crop_y=800 \
     -F crop_width=300 -F crop_height=300 http://localhost:8081/api/analyses
```

### `GET /healthz`

Renvoie `{"status": "ok"}` dès que le modèle est chargé. Sert aux sondes de disponibilité
(Docker, Kubernetes, OpenShift).

### `GET /`

La page web (`app/static/index.html`).

## 3. Interface utilisateur

Une page unique, sans framework ni dépendance externe :

1. choix du type de photo (**Smartphone** par défaut, ou **Dermatoscope**) ;
2. dépôt de la photo par glisser-déposer ou sélection, avec aperçu ;
3. recadrage facultatif : un cadre tracé à la souris ou au doigt sur l'aperçu, pour isoler
   une lésion sur une photo large. Le navigateur n'envoie que ses coordonnées, en pixels
   de la photo d'origine ; le découpage est fait par le serveur, sans perte de résolution.
   Un clic simple ou le bouton « Effacer le cadre » revient à la photo entière ;
4. bouton « Analyser » ;
5. résultat : score en pourcentage, message « au-dessus / en dessous du seuil d'alerte »
   avec la valeur du seuil, zone analysée (cadre ou photo entière), détail par classe
   (ancien modèle uniquement), nom du modèle et rappel que l'outil n'est pas un
   dispositif médical.

Sur le Space Hugging Face, l'utilisateur clique sur deux coins opposés de la photo ; un
aperçu montre la zone qui sera analysée. Comme dans l'application FastAPI, la zone est
découpée par le serveur dans la photo d'origine, en pleine résolution. L'éditeur d'image
de Gradio 6 n'est pas utilisé : quand seul son outil de recadrage est activé, sa barre
d'outils disparaît et il ne reste qu'un curseur de pinceau inutilisable.

Le type de photo est demandé parce que les deux sources n'ont pas du tout la même
distribution de scores : le seuil adapté au smartphone (0,896) est plus de dix fois plus
élevé que celui du dermatoscope (0,079). Voir [Méthodologie](04-methodologie-scientifique.md#8-seuils-dalerte-par-type-de-photo-trainingexport_probepy).

## 4. Données de santé

Une photo de lésion est une donnée de santé. L'application :

- ne **journalise** ni la photo ni le score (les messages d'erreur ne contiennent jamais
  le contenu de la requête) ;
- n'**écrit rien sur disque** : la photo est traitée en mémoire puis oubliée ;
- n'envoie la photo à **aucun service extérieur** : le modèle tourne dans le conteneur.

Seul Derm Foundation est téléchargé depuis Hugging Face, une fois, au premier démarrage.

Ces garanties valent pour l'application FastAPI. La version en ligne sur un Space Hugging
Face traite les photos sur les serveurs de Hugging Face, et Gradio en garde une copie
temporaire (voir [Licences](07-licences-conformite-et-limites.md#ce-que-fait-lapplication)).

## 5. Configuration

| Variable | Défaut | Rôle |
|---|---|---|
| `HF_TOKEN` | — | jeton Hugging Face, nécessaire au premier téléchargement de Derm Foundation |
| `SKIN_MODEL` | `derm-foundation-logreg` | modèle utilisé : `derm-foundation-logreg` ou `ham10000-efficientnet-b0` |
| `HF_HOME` | `/opt/hf` | cache des modèles dans le conteneur ; à monter sur un volume pour éviter de retélécharger |
| `HF_HUB_OFFLINE` | non défini | à `1`, interdit tout téléchargement (utile une fois le cache rempli) |

## 6. Image Docker

`Dockerfile`, basé sur `python:3.12-slim` :

1. installe uv puis les dépendances verrouillées (`uv sync --frozen`) ; avec
   `--build-arg EXTRAS=ham10000`, ajoute PyTorch et télécharge les poids de l'ancien
   modèle dans l'image ;
2. copie `app/` (dont `probe.npz`, quelques dizaines de Ko) ;
3. prépare `/opt/hf` et donne les droits au groupe 0 : **OpenShift** lance les conteneurs
   avec un identifiant d'utilisateur arbitraire appartenant à ce groupe ;
4. s'exécute en tant qu'utilisateur 1001, sur le port 8080.

Derm Foundation n'est **pas** copié dans l'image : ses conditions d'utilisation imposent
que chaque utilisateur les accepte, et l'image pourrait être partagée. Il est téléchargé
au démarrage avec le jeton du déploiement.

| Image | Taille |
|---|---|
| `skindetector:dev` (défaut) | environ 6,5 Go (TensorFlow) |
| `skindetector:ham10000` | environ 9,5 Go (PyTorch en plus) |

Mémoire à l'exécution : environ 2,3 Go après chargement de Derm Foundation.

## 7. Choix techniques et alternatives écartées

| Choix | Raison | Alternative écartée |
|---|---|---|
| Modèle dans le même processus que l'API | une seule image à construire, déployer et surveiller ; latence minimale | service de modèle séparé, utile seulement quand une autre application partage le conteneur (c'était le cas dans MedicalAssistant) |
| Classifieur exporté en `npz` | uniquement des nombres : rien n'est exécuté au chargement ; ne dépend pas de scikit-learn | `pickle` de scikit-learn, qui peut exécuter du code arbitraire et lie l'application à une version de scikit-learn |
| Révision de Derm Foundation figée | les embeddings d'une autre version ne correspondraient plus au classifieur | branche `main`, qui peut changer sans prévenir |
| TensorFlow sur CPU | 2 à 4 s par photo suffisent pour un usage interactif ; pas de GPU sur le cluster visé | GPU, inutile pour une photo à la fois |
| Réduction de la photo comme à l'entraînement | garantit des scores identiques à ceux mesurés | toute autre préparation (autre interpolation, recadrage) modifie les scores |
