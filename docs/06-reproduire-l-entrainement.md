# 06 · Reproduire l'entraînement

Ce guide refait toute la chaîne, des données brutes au fichier `app/probe.npz` utilisé
par l'application. Les commandes sont écrites pour **PowerShell**, depuis la racine du
dépôt. Ne pas les lancer depuis Git Bash : il transforme les chemins comme `/app` en
chemins Windows et casse les arguments `-e PYTHONPATH=/app` (voir [Dépannage](08-depannage.md)).

Durée totale : une journée environ, dont l'essentiel en attente (téléchargements,
calcul des embeddings).

## Vue d'ensemble

```
download_pad_ufes_20.py ─┐
téléchargement ISIC 2020 ┴─▶ build_manifest.py ─▶ manifest.csv
                                                     │
                     ┌───────────────────────────────┼──────────────────────────┐
                     ▼                               ▼                          ▼
        evaluate_baseline.py              embed.py (PAD : local,         check_shortcuts.py
        (ancien modèle)                    ISIC : GPU Google Cloud)
                                                     │
                                                     ▼
                              train_probe.py ─▶ cross_validate_probe.py
                                                     │
                                                     ▼
                                   export_probe.py ─▶ app/probe.npz
```

## 0. Préparer l'environnement

```powershell
# Image d'entraînement (TensorFlow, scikit-learn). Le secret n'est utile que si le
# HTTPS est inspecté par un antivirus ou un proxy.
docker build -t skindetector-training --secret id=extra_ca,src=.certs/local-ca.pem training

# Jeton Hugging Face lu dans .env, sans l'afficher
$env:HF_TOKEN = (Get-Content .env | Select-String '^HF_TOKEN=').Line.Split('=', 2)[1]
```

Les scripts lisent et écrivent dans `training/` (monté dans le conteneur) ; le cache des
modèles Hugging Face est dans le volume Docker `skindetector-hf`.

Raccourci utilisé dans la suite :

```powershell
function train { docker run --rm -v "${PWD}\training:/training" skindetector-training python @args }
```

## 1. Télécharger PAD-UFES-20

```powershell
train download_pad_ufes_20.py
```

- Interroge l'API de l'archive ISIC (collection 406), écrit
  `training/data/pad_ufes_20/metadata.csv` puis télécharge les 2 298 images (227 Mo)
  dans `images/`, 8 à la fois.
- Relançable : les images déjà présentes sont sautées.

## 2. Télécharger ISIC 2020

Depuis le serveur officiel du concours, dans `training/data/isic2020/` :

| Fichier | Taille |
|---|---|
| `https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip` | 24 707 698 022 octets (23 Go) |
| `https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth_v2.csv` | 2,4 Mo |

```powershell
New-Item -ItemType Directory -Force training\data\isic2020
cd training\data\isic2020
curl.exe -L -C - -O https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth_v2.csv
curl.exe -L -C - -O https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip
tar -xf ISIC_2020_Training_JPEG.zip      # crée train\ avec 33 126 images
cd ..\..\..
```

`-C -` reprend un téléchargement interrompu. Vérifier la taille de l'archive, puis le
nombre d'images extraites (33 126). L'archive peut ensuite être supprimée.

## 3. Construire le manifeste

```powershell
train build_manifest.py
```

Écrit `training/data/manifest.csv` et affiche les effectifs par jeu et par partie (voir
[Méthodologie](04-methodologie-scientifique.md#effectifs-obtenus)).

## 4. Mesurer l'ancien modèle (référence)

Dans l'image de l'application construite avec l'ancien modèle :

```powershell
docker build --build-arg EXTRAS=ham10000 --secret id=extra_ca,src=.certs/local-ca.pem -t skindetector:ham10000 .
docker run --rm -e PYTHONPATH=/app -e SKIN_MODEL=ham10000-efficientnet-b0 -e HF_HUB_OFFLINE=1 `
    -v "${PWD}\training:/app/training" skindetector:ham10000 `
    python training/evaluate_baseline.py pad_ufes_20            # partie test par défaut
# variantes : … evaluate_baseline.py pad_ufes_20 --split all
#             … evaluate_baseline.py isic2020
```

Résultats dans `training/results/ham10000-efficientnet-b0__<jeu>__<partie>.json`.
Environ 2 minutes pour PAD-UFES-20, 15 minutes pour le test d'ISIC 2020.

## 5. Calculer les embeddings

### PAD-UFES-20, sur le poste (environ 1 h 30)

```powershell
docker run --rm -e HF_TOKEN -v "${PWD}\training:/training" -v skindetector-hf:/hf_cache `
    skindetector-training python embed.py pad_ufes_20
```

Sur un poste dont le HTTPS est inspecté, ajouter le certificat pour le premier
téléchargement de Derm Foundation :

```powershell
docker run --rm -e HF_TOKEN -v "${PWD}\training:/training" -v skindetector-hf:/hf_cache `
    -v "${PWD}\.certs\local-ca.pem:/local-ca.pem:ro" skindetector-training sh -c `
    "cat /etc/ssl/certs/ca-certificates.crt /local-ca.pem > /tmp/ca.pem && export SSL_CERT_FILE=/tmp/ca.pem REQUESTS_CA_BUNDLE=/tmp/ca.pem && python embed.py pad_ufes_20"
```

- Options : `--limit N` (N premières images, pour un essai), `--batch-size N` (8 par
  défaut ; 32 sur GPU).
- Le calcul enregistre son avancement toutes les 200 images et reprend où il s'est
  arrêté. Il occupe tous les cœurs et environ 5 Go de mémoire.
- Empêcher la mise en veille du PC : elle suspend le conteneur.

### ISIC 2020, sur GPU dans Google Cloud

33 126 images demanderaient environ 23 heures sur le poste. Elles ont été calculées sur
une VM Google Cloud avec un GPU NVIDIA L4, en 1 h 15 de calcul.

#### a. Préparer le projet (une fois)

Dans la console Google Cloud, avec un compte **personnel** (un compte d'entreprise peut
interdire la création de projets ou de VM) :

1. créer un projet et y associer la facturation ; un essai gratuit n'autorise pas les GPU,
   il faut passer au compte complet ;
2. **Facturation → Budgets et alertes** : créer une alerte (par exemple 5 €). Elle prévient
   mais ne coupe rien ;
3. dans **Cloud Shell** (icône `>_`), une ligne à la fois :

```bash
gcloud services enable compute.googleapis.com
gcloud services enable cloudquotas.googleapis.com
gcloud config set billing/quota_project <ID_DU_PROJET>
# Quota global de GPU : souvent 0 sur un nouveau projet
gcloud compute project-info describe --flatten=quotas --format="value(quotas.metric,quotas.limit)" | grep GPUS_ALL_REGIONS
gcloud beta quotas preferences create --service=compute.googleapis.com \
  --quota-id=GPUS-ALL-REGIONS-per-project --preferred-value=1 \
  --project=<ID_DU_PROJET> --email=<EMAIL> \
  --justification="Calcul ponctuel d'embeddings d'images"
```

La réponse arrive par e-mail (quelques minutes à deux jours). Le quota est accordé quand
la commande `describe` affiche `1.0`.

#### b. Créer la VM

Les GPU L4 sont souvent épuisés dans une zone : essayer plusieurs zones. En Europe, les
trois zones de `europe-west4` étaient pleines ; `us-central1-b` a fonctionné. Les données
étant publiques et anonymisées, une région américaine ne pose pas de problème.

```bash
IMG=common-cu129-ubuntu-2204-nvidia-580
OPTS="--maintenance-policy=TERMINATE --boot-disk-size=100GB --metadata=install-nvidia-driver=True"
ZONES="us-central1-a us-central1-b us-central1-c us-east1-b us-east4-a us-west1-a"
for z in $ZONES; do gcloud compute instances create derm-embed --zone=$z --machine-type=g2-standard-8 --image-family=$IMG --image-project=deeplearning-platform-release $OPTS && break; done
```

- `g2-standard-8` : 8 processeurs, 32 Go de mémoire, un GPU L4 (24 Go) inclus.
- Image *Deep Learning VM* : pilote NVIDIA déjà prévu. Lister les familles disponibles :
  `gcloud compute images list --project=deeplearning-platform-release --no-standard-images --filter="family~common-cu12" --format="value(family)" | sort -u`.
- **Coller les commandes ligne par ligne** : le terminal web casse les lignes longues et
  ajoute des espaces après les `\`.
- **La facturation commence à la création** (environ 0,75 €/h).

#### c. Lancer le calcul

1. Console → **Compute Engine → Instances de VM → SSH** sur `derm-embed`. Vérifier le GPU :
   `nvidia-smi` doit afficher « NVIDIA L4 ».
2. Bouton **Importer un fichier** : envoyer `training/cloud/run_on_vm.sh`,
   `training/embed.py`, `training/data/manifest.csv`.
3. `bash run_on_vm.sh`. Le script, relançable à chaque étape :
   - installe uv, Python 3.12 et TensorFlow avec CUDA (`tensorflow[and-cuda]`) ;
   - vérifie que TensorFlow voit le GPU ;
   - télécharge et décompresse ISIC 2020 depuis le serveur officiel (environ 10 minutes à
     35 Mo/s, puis 5 minutes), puis supprime l'archive ;
   - demande le jeton Hugging Face (saisie masquée ; les caractères invisibles ajoutés
     par le collage sont retirés) ;
   - lance `embed.py isic2020 --batch-size 32` en arrière-plan (`nohup`) : la fenêtre
     SSH peut être fermée.
4. Suivre : `tail -f ~/training/embed.log`. Vitesse observée : 7,9 images/s, limitée par
   les 8 processeurs qui préparent les images plutôt que par le GPU.
5. Facultatif, pour s'assurer que le GPU donne les mêmes embeddings que le CPU : importer
   `training/cloud/check_gpu.py`, le placer dans `~/training`, puis lancer
   `.venv/bin/python check_gpu.py` et `CUDA_VISIBLE_DEVICES= .venv/bin/python check_gpu.py`.
   La seconde commande doit finir par `OK` (ce contrôle a été fait une fois, avec une
   similarité cosinus de 1,00000).

#### d. Rapatrier le résultat et tout supprimer

Le téléchargement direct depuis la fenêtre SSH (800 Mo) a échoué à répétition, et la
copie vers Cloud Shell plafonnait à 200 Ko/s. La voie fiable passe par un bucket :

```bash
# Sur la VM : se connecter avec son compte (la VM n'a pas le droit d'écrire dans un bucket)
sha256sum ~/training/data/embeddings/derm_foundation/isic2020.npz
gcloud auth login --no-launch-browser
gcloud config set project <ID_DU_PROJET>
gcloud storage buckets create gs://<ID_DU_PROJET>-embeddings --location=us-central1
gcloud storage cp ~/training/data/embeddings/derm_foundation/isic2020.npz gs://<ID_DU_PROJET>-embeddings/
gcloud auth revoke --all
```

Puis, dans Cloud Shell :

```bash
gcloud compute instances delete derm-embed --zone=<ZONE> --quiet
gcloud compute instances list     # doit être vide
gcloud compute disks list         # doit être vide
```

Télécharger le fichier depuis la console (**Cloud Storage → Buckets**), le placer dans
`training/data/embeddings/derm_foundation/`, vérifier son empreinte SHA-256, puis
supprimer le bucket ou le projet entier (`gcloud projects delete <ID_DU_PROJET>`).

#### Coût réel

| Poste | Quantité | Coût |
|---|---|---|
| GPU L4 | 6,71 h | 3,31 € |
| Processeurs de la VM | 53,7 h (8 × 6,71 h) | 1,18 € |
| Mémoire de la VM | 214,7 Go·h | 0,55 € |
| Disque 100 Go | 6,7 h | 0,08 € |
| Réseau, licences, essais ratés | | 0,01 € |
| **Total** | | **5,13 €** |

La VM a tourné 6 h 43 pour 1 h 15 de calcul : l'installation, puis surtout les difficultés
à rapatrier le fichier ont triplé la facture. Avec la méthode du bucket dès le départ,
compter **2 à 3 €**.

## 6. Entraîner le classifieur

```powershell
train train_probe.py pad_ufes_20 isic2020      # modèle de l'application
train train_probe.py pad_ufes_20               # variante PAD seul
train train_probe.py isic2020                  # variante ISIC seul
```

- Affiche l'AUC de validation pour chaque `C`, retient le meilleur, puis évalue la partie
  test de chaque jeu.
- Écrit `training/results/derm-foundation-logreg_<jeux>__<jeu>__test.json` et
  `training/data/models/derm-foundation-logreg_<jeux>.pkl`.
- Quelques minutes pour PAD + ISIC (7 entraînements sur 25 175 images).

## 7. Contrôles

```powershell
train cross_validate_probe.py pad_ufes_20      # validation croisée 5 plis par patient
train check_shortcuts.py pad_ufes_20           # AUC à partir de la seule taille d'image
```

## 8. Exporter le classifieur pour l'application

```powershell
docker run --rm -v "${PWD}:/repo" -w /repo/training skindetector-training python export_probe.py
```

- Recharge `derm-foundation-logreg_pad_ufes_20+isic2020.pkl`, calcule les seuils à 95 %
  de sensibilité sur la validation de chaque jeu, vérifie que le calcul manuel redonne
  les scores de scikit-learn, et écrit **`app/probe.npz`**.
- Reconstruire ensuite l'image de l'application : `docker compose up --build`.

## 9. Vérifier l'application

```powershell
curl.exe -F "image=@training/data/pad_ufes_20/images/ISIC_9118085.jpg" -F "source=smartphone" http://localhost:8081/api/analyses
```

Le score doit valoir 0,9939 (voir les autres cas de contrôle dans
[Modèles](03-modeles.md#vérification-de-bout-en-bout)).

## Fichiers produits

| Fichier | Versionné | Contenu |
|---|---|---|
| `training/data/pad_ufes_20/`, `isic2020/` | non | images et métadonnées (49 Go) |
| `training/data/manifest.csv` | non | liste des images, étiquettes, découpage |
| `training/data/embeddings/derm_foundation/*.npz` | non | embeddings (57 Mo + 816 Mo) |
| `training/data/models/*.pkl` | non | classifieurs scikit-learn |
| `training/results/*.json` | oui | toutes les mesures |
| `app/probe.npz` | oui | classifieur exporté pour l'application |
