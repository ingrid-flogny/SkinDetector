# 07 · Licences, conformité et limites

> Cette page résume une lecture attentive des textes cités, sans constituer un avis
> juridique. Avant tout usage commercial ou clinique, la faire valider par un juriste et
> par une personne qualifiée en affaires réglementaires.

## 1. Licences des composants

| Composant | Licence / conditions | Ce qu'elles impliquent ici |
|---|---|---|
| Derm Foundation | *Health AI Developer Foundations Terms of Use* (Google) | accès après acceptation ; pas de redistribution sans conditions ; restrictions d'usage (voir ci-dessous) |
| ISIC 2020 | CC BY-NC 4.0 | citation obligatoire ; **pas d'usage commercial** des données |
| PAD-UFES-20 | CC BY 4.0 | citation obligatoire |
| Ancien modèle (`pthennakoon25/skinvision-ai-model`) | MIT | usage libre, mention de la licence |
| HAM10000 (données de l'ancien modèle) | CC BY-NC 4.0 | concerne l'ancien modèle, non redistribué ici |
| Code de SkinDetector | dépôt privé, aucune licence publiée | à choisir avant toute ouverture |

### Conditions de Derm Foundation

Points principaux des *HAI-DEF Terms of Use* :

- **Restrictions d'usage** : pas d'usage interdit par la politique d'usage de Google, ni
  d'usage qui amènerait une autorité de santé à considérer Google comme fabricant d'un
  dispositif médical.
- **Usage clinique** (diagnostic ou traitement de patients, y compris en recherche) :
  il faut, le cas échéant, l'**autorisation de l'autorité de santé** compétente.
- **Sorties** (les embeddings) : Google n'en revendique pas la propriété ; l'utilisateur
  en est responsable.
- **Dérivés** : tout modèle « créé par transfert des motifs des poids, paramètres,
  opérations ou sorties » de HAI-DEF est un *Model Derivative*. Notre classifieur,
  entraîné sur les embeddings, en est très probablement un.

C'est pourquoi l'image Docker n'embarque pas Derm Foundation : il est téléchargé par
chaque déploiement, avec un jeton dont le titulaire a accepté les conditions.

## 2. Publier le classifieur

`app/probe.npz` est publié sur Hugging Face :
**https://huggingface.co/ingridflogny/skindetector-derm-probe**. Les conditions HAI-DEF de
redistribution (section 3.1) y sont remplies ainsi :

| Condition | Comment elle est remplie |
|---|---|
| Joindre une **copie des conditions HAI-DEF** | `HAI-DEF-TERMS.md`, copie du texte du 15 novembre 2024 |
| Fichier **`NOTICE`** avec la phrase imposée : *« HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms of Use »* | `NOTICE`, qui précise aussi que `probe.npz` est un dérivé non approuvé par Google |
| Signaler que les fichiers sont modifiés ou dérivés de HAI-DEF | `NOTICE` et section « Licence » de la fiche |
| Reprendre les **restrictions d'usage** (section 3.2) comme clause opposable | section « Licence » de la fiche : télécharger ou utiliser le classifieur vaut acceptation |
| Usage **non commercial** (ISIC 2020) et citations | fiche du modèle |

Choix de publication :

- **Public, sans demande d'accès** : tout le monde peut télécharger `probe.npz`. Les
  conditions HAI-DEF n'imposent pas de formulaire ; en contrepartie, aucun engagement
  explicite n'est recueilli et l'on ne sait pas qui télécharge le classifieur ni pour
  quel usage. Un accès contrôlé (`gated`) peut être rétabli à tout moment dans les
  réglages du dépôt.
- **Derm Foundation n'est pas republié** : chaque utilisateur du classifieur doit obtenir
  son propre accès, et utiliser la révision figée indiquée dans la fiche.
- **Format `npz`** : uniquement des nombres, aucun code exécuté au chargement, contrairement
  à un `pickle`.

Le contenu du dépôt est versionné dans `huggingface/model/` ; la commande de publication
est dans [Reproduire l'entraînement](06-reproduire-l-entrainement.md#10-publier-le-classifieur-sur-hugging-face).

### L'application en ligne

L'application est aussi publiée, en accès libre, sur le Space
**https://huggingface.co/spaces/ingridflogny/SkinDetector** : même API et même page que la
version locale. Son code, dont `probe.npz`, y est public : le Space contient donc lui
aussi la `NOTICE` et la copie des conditions HAI-DEF, et sa fiche reprend les
restrictions d'usage et la clause non commerciale. Sur le Space, la page affiche en tête
un avis d'hébergement : photos analysées par Hugging Face, « pas un dispositif médical »,
lien vers le classifieur et mention HAI-DEF.

Hébergement gratuit : les Spaces Gradio et Docker demandent désormais un abonnement
payant à la création. Le Space réutilise un Space Gradio créé avant ce changement, sur le
matériel CPU Basic gratuit. Il reste déclaré de type Gradio, mais exécute l'application
FastAPI elle-même (`huggingface/space/space.py`) : l'éditeur d'image de Gradio 6 n'offrait
pas de recadrage utilisable. Rien ne garantit que Hugging Face maintienne cette gratuité.

## 3. Données de santé et réglementation

### Ce que fait l'application

En local, aucune photo n'est enregistrée, journalisée ni envoyée à un tiers (voir
[Architecture](02-architecture-et-api.md#4-données-de-santé)). Le développement n'a
utilisé que des **jeux publics anonymisés**.

Sur le **Space Hugging Face**, c'est différent : les photos sont envoyées aux serveurs de
Hugging Face, un hébergeur tiers situé hors de l'Union européenne et non certifié HDS.
L'application ne les conserve ni ne les journalise, mais rien ne garantit ce que fait
l'infrastructure de Hugging Face. Le Space ne convient donc qu'à des photos de
démonstration, jamais à des photos de patients : la page demande de ne pas envoyer de
photo permettant d'identifier une personne.

### Avant tout usage avec de vrais patients

| Exigence | Pourquoi |
|---|---|
| **Statut de dispositif médical** | un logiciel qui aide à décider si une lésion est maligne a une finalité médicale : en Europe, il relève du règlement (UE) 2017/745 (MDR), probablement en classe IIa ou plus, avec marquage CE, évaluation clinique et système de management de la qualité |
| **Validation clinique** | les mesures actuelles portent sur des jeux publics ; il faut une étude sur la population, les appareils et le parcours de soins visés |
| **Hébergement de données de santé** | en France, un hébergeur certifié HDS dès que des données de patients transitent par un serveur |
| **RGPD** | base légale, information du patient, analyse d'impact (AIPD), registre des traitements |
| **Authentification** | accès réservé aux professionnels de santé identifiés |
| **Autorisation prévue par HAI-DEF** | pour un usage clinique de Derm Foundation |
| **Licences des données** | ISIC 2020 interdit l'usage commercial : un produit commercial demanderait d'autres données ou un accord |

## 4. Limites scientifiques

### Données

- **Peu de mélanomes en contexte smartphone** : 52 dans PAD-UFES-20, dont 7 en test. Les
  mesures sur le mélanome au smartphone sont très incertaines.
- **Populations limitées** : patients brésiliens (PAD-UFES-20) et de centres européens,
  américains et australiens (ISIC 2020), majoritairement à **peau claire**. Les
  performances sur les phototypes foncés (Fitzpatrick V et VI) ne sont **pas connues** et
  sont probablement moins bonnes.
- **Diagnostics imparfaits** : 42 % des diagnostics de PAD-UFES-20 reposent sur un
  consensus d'experts sans histologie ; 82 % des images d'ISIC 2020 sont « unknown »
  (jugées bénignes, sans diagnostic précis).
- **Recrutement hospitalier** : PAD-UFES-20 provient d'un service de dermatologie (79 % de
  lésions malignes), loin de la proportion rencontrée en médecine générale.
- **Types de lésions** : seules les lésions présentes dans ces jeux sont couvertes. Un
  carcinome de Merkel, un lymphome cutané ou une lésion des muqueuses n'ont jamais été
  vus.

### Modèle

- **Chevauchement possible avec le préentraînement** : Derm Foundation a été affiné sur des
  « images publiques supplémentaires » non listées. Si PAD-UFES-20 ou ISIC en font partie,
  les résultats sont surestimés.
- **Score non calibré** : il ne s'interprète pas comme une probabilité (classes
  rééquilibrées à l'entraînement).
- **Seuils fragiles** : le seuil dermatoscope repose sur 76 mélanomes de validation ; sa
  sensibilité réelle peut s'écarter sensiblement de 95 %.
- **Type de photo déclaré** : un mauvais choix (smartphone au lieu de dermatoscope)
  applique un seuil faux d'un facteur 10. Une photo de smartphone prise à travers un
  adaptateur dermoscopique est un cas non étudié.
- **Une photo, une lésion** : pas de comparaison avec les autres lésions du patient, pas
  de suivi dans le temps, pas d'informations cliniques (âge, antécédents, évolution).
- **Cadrage** : le modèle a été entraîné sur des gros plans d'une seule lésion. Sur une
  photo large, l'utilisateur doit encadrer la lésion ; rien ne l'y oblige, et la
  résolution minimale d'un cadre pour un score fiable n'a pas été mesurée.
- **Qualité d'image** : aucun contrôle automatique de netteté, d'exposition ou de cadrage ; Google
  signale une baisse de qualité sur les images extrêmes.

### Évaluation

- **Pas de validation externe** : le test vient des mêmes sources que l'entraînement
  (patients différents, mêmes centres, mêmes appareils).
- **Test consulté plusieurs fois** : les hyperparamètres ont été choisis sur la
  validation, mais le choix entre variantes a été fait en connaissant les résultats de
  test (voir [Méthodologie](04-methodologie-scientifique.md#11-chronologie-des-expériences)).
- **Spécificité à 95 % de sensibilité** calculée avec un seuil fixé sur le jeu de test
  lui-même : mesure de capacité, légèrement optimiste.

## 5. Citations

Toute publication ou présentation des résultats doit citer :

- **PAD-UFES-20** : Pacheco A. G. C. et al., *PAD-UFES-20: a skin lesion dataset composed
  of patient data and clinical images collected from smartphones*, Data in Brief 32,
  106221, 2020.
- **ISIC 2020** : Rotemberg V. et al., *A patient-centric dataset of images and metadata
  for identifying melanomas using clinical context*, Scientific Data 8, 34, 2021.
- **Derm Foundation** : Google, *Health AI Developer Foundations*,
  https://huggingface.co/google/derm-foundation.
- **HAM10000** (ancien modèle) : Tschandl P., Rosendahl C., Kittler H., *The HAM10000
  dataset, a large collection of multi-source dermatoscopic images of common pigmented
  skin lesions*, Scientific Data 5, 180161, 2018.
