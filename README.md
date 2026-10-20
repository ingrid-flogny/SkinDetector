# SkinDetector

Application web qui attribue un **score de malignité** à la photo d'une lésion cutanée
(grain de beauté, tache, lésion suspecte), prise au **smartphone** ou au **dermatoscope**.

Le score est produit par **Derm Foundation**, un modèle de fondation de Google pour la
dermatologie, surmonté d'un classifieur linéaire entraîné dans ce dépôt sur deux jeux de
données publics, **PAD-UFES-20** (smartphone) et **ISIC 2020** (dermatoscope).

> **Avertissement.** SkinDetector est un projet de recherche et d'apprentissage. Ce
> n'est **pas un dispositif médical** : il n'a ni marquage CE ni validation clinique,
> et il ne pose aucun diagnostic. Un score bas n'exclut pas une lésion maligne. Il ne
> doit pas être utilisé sur de vrais patients en l'état (voir
> [Licences, conformité et limites](docs/07-licences-conformite-et-limites.md)).

## En bref

| | |
|---|---|
| Entrée | une photo (JPEG, PNG…) + le type d'appareil : `smartphone` ou `dermatoscope` |
| Sortie | un score de 0 à 1 et le seuil d'alerte propre au type de photo |
| Modèle | Derm Foundation (`google/derm-foundation`, révision figée) + régression logistique |
| Classifieur publié | [`ingridflogny/skindetector-derm-probe`](https://huggingface.co/ingridflogny/skindetector-derm-probe) sur Hugging Face, en accès libre |
| Performance (test) | AUC **0,86** sur ISIC 2020 (dermatoscope), **0,96** sur PAD-UFES-20 (smartphone) |
| Ancien modèle | EfficientNet-B0 entraîné sur HAM10000, conservé en option : AUC 0,64 et 0,84 |
| Technique | Python 3.12, FastAPI, TensorFlow (CPU), Docker, environ 2,5 Go de mémoire |

## Démarrage rapide

Prérequis : Docker, et un jeton Hugging Face ayant accès à `google/derm-foundation`
(détails dans [Prérequis et installation](docs/01-prerequis-et-installation.md)).

```bash
cp .env.example .env          # puis renseigner HF_TOKEN
docker compose up --build     # premier démarrage : téléchargement de Derm Foundation (~1 Go)
```

Ouvrir ensuite **http://localhost:8081**, choisir le type de photo, déposer une image,
puis cliquer sur « Analyser ».

## Documentation

| Document | Contenu |
|---|---|
| [01 · Prérequis et installation](docs/01-prerequis-et-installation.md) | comptes, accès au modèle, poste de travail, Docker, PyCharm |
| [02 · Architecture et API](docs/02-architecture-et-api.md) | composants, flux d'une analyse, API HTTP, configuration, image Docker |
| [03 · Modèles](docs/03-modeles.md) | Derm Foundation, classifieur exporté, ancien modèle HAM10000 |
| [04 · Méthodologie scientifique](docs/04-methodologie-scientifique.md) | question posée, données, étiquettes, découpage, entraînement, métriques |
| [05 · Résultats](docs/05-resultats.md) | tous les résultats chiffrés et leur interprétation |
| [06 · Reproduire l'entraînement](docs/06-reproduire-l-entrainement.md) | les commandes pas à pas, y compris le calcul sur GPU dans Google Cloud |
| [07 · Licences, conformité et limites](docs/07-licences-conformite-et-limites.md) | conditions d'utilisation, données de santé, biais, réglementation |
| [08 · Dépannage](docs/08-depannage.md) | problèmes rencontrés et leurs solutions |

## Organisation du dépôt

```
SkinDetector/
├── app/                       Application servie en production
│   ├── main.py                API FastAPI et page web
│   ├── skin.py                Interface commune des modèles, choix par SKIN_MODEL
│   ├── derm_foundation.py     Modèle par défaut : Derm Foundation + classifieur
│   ├── ham10000.py            Ancien modèle (option EXTRAS=ham10000)
│   ├── probe.npz              Poids du classifieur et seuils d'alerte
│   └── static/index.html      Interface utilisateur
├── training/                  Données, entraînement et évaluation (hors de l'image)
│   ├── download_pad_ufes_20.py, build_manifest.py
│   ├── embed.py, train_probe.py, cross_validate_probe.py, export_probe.py
│   ├── evaluate_baseline.py, check_shortcuts.py, metrics.py
│   ├── cloud/                 Calcul des embeddings sur une VM GPU Google Cloud
│   ├── results/               Résultats chiffrés (JSON), versionnés
│   └── data/                  Images, embeddings, modèles : non versionné
├── huggingface/               Publication du classifieur sur Hugging Face
├── docs/                      Cette documentation
├── Dockerfile, compose.yaml   Image et lancement local
└── pyproject.toml, uv.lock    Dépendances de l'application
```

## Origine

SkinDetector est né comme un onglet de l'application MedicalAssistant, dont il a été
séparé pour évoluer à son rythme : ses besoins (TensorFlow, environ 2,5 Go de mémoire,
jeux de données de 49 Go) n'ont rien de commun avec ceux de la rédaction d'observations.
