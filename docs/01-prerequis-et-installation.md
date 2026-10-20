# 01 · Prérequis et installation

Ce document liste tout ce qu'il faut pour faire tourner SkinDetector, puis pour
entraîner le modèle. L'application seule demande peu de choses ; l'entraînement demande
de l'espace disque, du temps de calcul et, pour une partie, un GPU dans le cloud.

## 1. Comptes et accès

### Hugging Face (obligatoire)

Derm Foundation est hébergé sur Hugging Face dans un dépôt **à accès contrôlé**
(« gated ») : il faut accepter les conditions de Google avant de pouvoir le télécharger.

1. Créer un compte sur https://huggingface.co.
2. Ouvrir https://huggingface.co/google/derm-foundation, lire les conditions
   d'utilisation (*Health AI Developer Foundations Terms of Use*) et demander l'accès.
   L'accès est généralement accordé immédiatement.
3. Créer un jeton dans **Settings → Access Tokens** :
   - type **Read** (lecture seule) : c'est le plus simple et le plus sûr ;
   - si vous choisissez un jeton **fine-grained**, cocher impérativement
     *« Read access to contents of all public gated repos you can access »*. Sans cette
     case, le téléchargement échoue avec une erreur 403 (voir [Dépannage](08-depannage.md)).
4. Copier `.env.example` en `.env` et y coller le jeton : `HF_TOKEN=hf_…`.
   Le fichier `.env` est exclu de Git ; le jeton ne doit jamais être commité ni collé
   dans une conversation.

Le jeton n'est utilisé qu'au premier démarrage, pour télécharger Derm Foundation.

### Google Cloud (facultatif, entraînement uniquement)

Seul le calcul des embeddings d'ISIC 2020 (33 126 images) a demandé un GPU. Il faut :
un projet Google Cloud personnel avec facturation, une alerte de budget, et un quota
« GPUs (all regions) » d'au moins 1. La marche à suivre et les coûts réels sont dans
[Reproduire l'entraînement](06-reproduire-l-entrainement.md#5-embeddings-disque-sur-gpu-google-cloud).

### GitHub (facultatif)

Le dépôt est privé : `ingrid-flogny/SkinDetector`. Sur le poste de développement, Git
communique avec GitHub en SSH (voir [Dépannage](08-depannage.md#git-ne-peut-pas-pousser-en-https)).

## 2. Poste de travail

| Élément | Minimum | Utilisé pour le développement |
|---|---|---|
| Système | Windows, macOS ou Linux | Windows 11, processeur ARM64 |
| Docker | Docker Desktop récent avec BuildKit | Docker Desktop 29.5 |
| Mémoire allouée à Docker | 4 Go pour l'application, 8 Go pour l'entraînement | 15,4 Go |
| Disque | 8 Go pour l'application | 49 Go de données d'entraînement + images Docker |
| Python (facultatif) | 3.12, pour PyCharm | CPython 3.12 x86_64 (émulé), via uv |
| uv (facultatif) | 0.9 ou plus | gestion des dépendances et des environnements |

**Tout s'exécute dans Docker** : l'application comme les scripts d'entraînement. Python
installé sur le poste ne sert qu'à l'éditeur (complétion, navigation dans le code).

### Particularités du poste de développement

Ces contraintes ont façonné plusieurs choix du projet ; elles ne concernent pas
forcément d'autres postes.

- **Antivirus Avast avec inspection HTTPS** : Avast remplace les certificats des sites
  par le sien. Les outils qui ne font pas confiance au magasin de certificats de Windows
  échouent (`invalid peer certificate: UnknownIssuer`). D'où :
  - `uv … --native-tls` pour les commandes uv lancées sur le poste ;
  - le certificat racine d'Avast exporté dans `.certs/local-ca.pem` (non versionné) et
    transmis aux builds Docker comme secret BuildKit `extra_ca` ;
  - le fichier `compose.override.yaml` (non versionné), qui ajoute ce certificat au
    build et au conteneur.
- **Smart App Control (Windows)** bloque certaines DLL non signées, notamment celles de
  quelques bibliothèques Python et le `libcurl` de Git. C'est une protection voulue :
  on ne la contourne pas, on exécute le code dans Docker et on utilise Git en SSH.
- **Python natif et SSL** : sur ce poste, les scripts Python qui ouvrent une connexion
  HTTPS plantent (`OPENSSL_Uplink … no OPENSSL_Applink`). Raison de plus pour tout lancer
  dans Docker.

## 3. Installation de l'application

```bash
git clone git@github.com:ingrid-flogny/SkinDetector.git
cd SkinDetector
cp .env.example .env                  # renseigner HF_TOKEN
docker compose up --build
```

- Premier démarrage : environ 5 minutes (construction de l'image, puis téléchargement de
  Derm Foundation, environ 1,1 Go, conservé dans le volume Docker `hf-cache`).
- Démarrages suivants : environ 30 secondes, le temps de charger le modèle en mémoire.
- L'application répond sur **http://localhost:8081**. Le port 8081 a été choisi parce
  que MedicalAssistant occupe déjà le 8080 sur le poste de développement.

Sur un poste dont le HTTPS est inspecté, créer `compose.override.yaml` (modèle dans le
dépôt de développement) et placer le certificat racine dans `.certs/local-ca.pem`.

Arrêter : `docker compose down`. Le volume `hf-cache` est conservé ; pour le supprimer
aussi : `docker compose down -v`.

### Avec l'ancien modèle (HAM10000)

```bash
docker build --build-arg EXTRAS=ham10000 -t skindetector:ham10000 .
docker run --rm -p 8081:8080 -e SKIN_MODEL=ham10000-efficientnet-b0 skindetector:ham10000
```

L'image est alors plus lourde d'environ 3 Go (PyTorch), mais n'a besoin ni de
Derm Foundation ni de `HF_TOKEN`.

## 4. Ouvrir le projet dans PyCharm

1. **File → Open**, choisir le dossier `SkinDetector`, puis **New Window** pour le garder
   séparé de MedicalAssistant.
2. Créer l'environnement local (une fois) : `uv sync --native-tls` dans le terminal du
   projet. Il installe TensorFlow et les autres dépendances dans `.venv`.
3. **Settings → Project → Python Interpreter → Add Interpreter → Existing** et choisir
   `.venv\Scripts\python.exe`.

Le dossier `training/` a son propre `pyproject.toml` (scikit-learn, TensorFlow) : pour
la complétion dans ces scripts, créer de même `training/.venv` avec
`uv sync --native-tls --project training`, ou se contenter de l'environnement principal.

## 5. Prérequis de l'entraînement

| Besoin | Détail |
|---|---|
| Disque | PAD-UFES-20 : 227 Mo ; ISIC 2020 : 23 Go d'images + 23 Go pour l'archive (supprimable après extraction) ; embeddings : 870 Mo |
| Mémoire Docker | 8 Go conseillés : Derm Foundation occupe environ 5 Go en calcul par lots |
| Calcul | PAD-UFES-20 : environ 1 h 30 sur 8 cœurs ; ISIC 2020 : environ 23 h sur 8 cœurs, ou 1 h 15 sur un GPU NVIDIA L4 |
| Image Docker | `skindetector-training` (`training/Dockerfile`) |
| Volume Docker | `skindetector-hf` pour le cache Hugging Face (ne fonctionne pas dans un dossier partagé avec Windows) |

Toutes les commandes sont détaillées dans [Reproduire l'entraînement](06-reproduire-l-entrainement.md).
