# 08 · Dépannage

Problèmes rencontrés pendant le développement, avec leur cause et leur solution.

## Application

### Le premier démarrage est long, puis échoue avec une erreur Hugging Face

| Message | Cause | Solution |
|---|---|---|
| `401 Unauthorized` ou `GatedRepoError` | accès à `google/derm-foundation` non demandé, ou `HF_TOKEN` absent de `.env` | demander l'accès sur la page du modèle ; vérifier `.env` |
| `403 Forbidden: Please enable access to public gated repositories in your fine-grained token settings` | jeton *fine-grained* sans le droit sur les dépôts à accès contrôlé | dans les réglages du jeton, cocher *Read access to contents of all public gated repos you can access*, ou utiliser un jeton *Read* |
| `LocalEntryNotFoundError` | message générique de `huggingface_hub` qui cache la vraie erreur (souvent le 403 ci-dessus, ou un problème réseau) | lire la ligne `Caused by` ou relancer pour voir l'erreur d'origine |
| `invalid peer certificate: UnknownIssuer`, `CERTIFICATE_VERIFY_FAILED` | HTTPS inspecté par un antivirus ou un proxy | placer le certificat racine dans `.certs/local-ca.pem` et utiliser `compose.override.yaml` |

Le téléchargement (1,1 Go) peut prendre plusieurs minutes sans rien afficher ; suivre
la progression avec `docker compose logs -f app`.

### `422` : « Input should be 'smartphone' or 'dermatoscope' »

Le champ `source` manque ou a une autre valeur. L'interface l'envoie toujours ; avec
`curl`, ajouter `-F "source=smartphone"`.

### Le score semble faux pour une photo de dermatoscope

Vérifier le type de photo choisi : les seuils diffèrent d'un facteur 10. Un score de
0,10 est **au-dessus** du seuil dermatoscope (0,079) mais très en dessous du seuil
smartphone (0,896).

### Le port 8081 est déjà utilisé

Changer le port publié dans `compose.yaml` (`"8082:8080"` par exemple).

## Docker

### Docker Desktop ne répond plus, ou ne s'ouvre plus

Symptômes : `request returned 500 Internal Server Error … dockerDesktopLinuxEngine`,
commandes `docker` qui ne rendent pas la main, journal de Docker rempli de
`no route to host`. La machine virtuelle de Docker est figée.

```powershell
Get-Process | Where-Object { $_.Name -match '^(Docker Desktop|com\.docker\..*|docker)$' } | Stop-Process -Force
wsl --shutdown
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

Si **Kubernetes** est activé dans Docker Desktop sans être utilisé (« Waiting for node to
be ready » en boucle), le désactiver dans **Settings → Kubernetes** : il consomme de la
mémoire et a coïncidé avec ces blocages.

### Les conteneurs sont tués ou le calcul ralentit

Mémoire insuffisante : l'application demande environ 2,5 Go, le calcul des embeddings
environ 5 Go. Augmenter la mémoire dans **Settings → Resources**, ou arrêter les autres
conteneurs (MedicalAssistant).

### Le calcul des embeddings s'arrête pendant la nuit

La mise en veille du PC suspend Docker. Désactiver la veille pendant le calcul ; le
script reprend ensuite où il s'était arrêté.

### Le cache Hugging Face ne fonctionne pas dans `training/data/hf_cache`

Les dossiers partagés avec Windows ne supportent pas les liens symboliques et les
verrous dont le cache a besoin. Utiliser le volume Docker `skindetector-hf`, monté sur
`/hf_cache`.

## Windows

### `No module named 'app'` ou chemins étranges avec `docker run`

Git Bash convertit les arguments qui ressemblent à des chemins Unix (`/app` devient
`C:/Program Files/Git/app`). Lancer ces commandes depuis **PowerShell**, ou préfixer par
`MSYS_NO_PATHCONV=1` dans Git Bash.

### `DLL load failed … stratégie de contrôle d'application`

**Smart App Control** bloque une DLL non signée (par exemple celle de PyAV). C'est une
protection voulue : ne pas la désactiver ; exécuter le code dans Docker.

### `OPENSSL_Uplink … no OPENSSL_Applink`

Plantage de Python natif dès qu'il ouvre une connexion HTTPS sur ce poste. Exécuter le
script dans Docker.

### `uv` : `invalid peer certificate: UnknownIssuer`

Ajouter `--native-tls` : uv utilise alors le magasin de certificats de Windows, qui
contient celui de l'antivirus.

### Un fichier `.env` créé avec PowerShell n'est pas lu correctement

`Set-Content -Encoding utf8` ajoute un BOM en tête du fichier : la première variable
n'est plus reconnue. Enregistrer en UTF-8 **sans** BOM (éditeur de PyCharm, ou
`[IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))`).

### Git ne peut pas pousser en HTTPS

`failed to load library 'libcurl-4.dll'` : Smart App Control bloque le libcurl de Git.
Le dépôt utilise SSH avec le client OpenSSH de Windows :

```powershell
git config core.sshCommand "C:/Windows/System32/OpenSSH/ssh.exe -i C:/Users/<vous>/.ssh/id_ed25519_github -o IdentitiesOnly=yes"
git remote set-url origin git@github.com:ingrid-flogny/SkinDetector.git
```

La clé publique correspondante doit être ajoutée dans **GitHub → Settings → SSH and GPG
keys**.

## Google Cloud

| Problème | Cause | Solution |
|---|---|---|
| `PERMISSION_DENIED … serviceusage.services.enable` ou création de projet désactivée | compte d'entreprise verrouillé par ses administrateurs | utiliser un compte personnel, ou demander un projet dédié aux administrateurs |
| aucune GPU utilisable malgré un quota L4 de 1 | quota global `GPUS_ALL_REGIONS` à 0 | demander 1 (voir [Reproduire](06-reproduire-l-entrainement.md#a-préparer-le-projet-une-fois)) ; impossible pendant l'essai gratuit |
| `Exactly one of (--folder \| --organization \| --project) must be specified` | `gcloud beta quotas` n'utilise pas le projet par défaut | ajouter `--project=<ID>` |
| `… requires a quota project` / `SERVICE_DISABLED` sur `cloudquotas` | appel facturé au projet technique de Cloud Shell | `gcloud services enable cloudquotas.googleapis.com` puis `gcloud config set billing/quota_project <ID>` |
| `ZONE_RESOURCE_POOL_EXHAUSTED` | plus de GPU libre dans la zone | essayer d'autres zones ou régions, ou réessayer plus tard |
| `resource … family/… was not found` ; `--image-project: command not found` | ligne collée coupée en deux par le terminal web | coller des lignes courtes, une à la fois, avec des variables |
| une VM inattendue existe (`derm-embed` dans une autre zone) | une commande coupée a créé une VM avec les réglages par défaut | `gcloud compute instances list`, puis supprimer |
| `The current platform CUDA is not among the platforms required by the module: [CPU]` | Derm Foundation exporté pour le CPU seulement | utiliser `embed.py`, qui lève ce verrou (voir [Modèles](03-modeles.md#particularité-technique--un-modèle-exporté-pour-le-cpu-seulement)) |
| `Trying to access resource … located in device GPU:0 from device CPU:0` | GPU et CPU utilisés dans le même processus TensorFlow | faire les deux calculs dans deux processus (`check_gpu.py` le fait) |
| `BadRequestError` de Hugging Face sur la VM | caractères invisibles ajoutés au jeton par le collage | `run_on_vm.sh` les retire ; sinon nettoyer avec `tr -cd '[:alnum:]_'` |
| téléchargement de 800 Mo impossible depuis la fenêtre SSH ; `gcloud compute scp` à 200 Ko/s | limites du terminal web et de Cloud Shell | passer par un bucket Cloud Storage |
| `gcloud storage cp` refusé depuis la VM | compte de service de la VM en lecture seule sur le stockage | `gcloud auth login --no-launch-browser` sur la VM, puis `gcloud auth revoke --all` |
| alertes de budget reçues après suppression de la VM | les coûts sont comptabilisés avec jusqu'à 24 h de retard | vérifier dans **Facturation → Rapports**, regroupé par SKU, que le total n'augmente plus |
