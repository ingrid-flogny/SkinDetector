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

Publier `app/probe.npz` sur Hugging Face (compte `ingridflogny`) est possible à ces
conditions :

1. joindre une **copie des conditions HAI-DEF** ;
2. ajouter un fichier **`NOTICE`** contenant la phrase imposée :
   *« HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms
   of Use »* ;
3. signaler que les fichiers sont modifiés ou dérivés de HAI-DEF ;
4. reprendre les **restrictions d'usage** de Google comme clause opposable ;
5. à cause d'ISIC 2020, publier pour un **usage non commercial**, et citer ISIC et
   PAD-UFES-20.

Derm Foundation lui-même n'est pas republié : chaque utilisateur du classifieur doit
obtenir son propre accès. Recommandations : commencer par un dépôt **privé** ou à accès
contrôlé ; rédiger une fiche de modèle reprenant [Résultats](05-resultats.md) et les
limites ci-dessous ; garder le format `npz`, qui ne contient que des nombres, plutôt
qu'un `pickle`.

## 3. Données de santé et réglementation

### Ce que fait l'application

Aucune photo n'est enregistrée, journalisée ni envoyée à un tiers (voir
[Architecture](02-architecture-et-api.md#4-données-de-santé)). Le développement n'a
utilisé que des **jeux publics anonymisés**.

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
- **Qualité d'image** : aucun contrôle de netteté, d'exposition ou de cadrage ; Google
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
