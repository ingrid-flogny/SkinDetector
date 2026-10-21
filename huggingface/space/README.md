---
title: SkinDetector
emoji: 🔬
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: space.py
license: other
short_description: Score de malignité de lésions cutanées (non médical)
models:
  - google/derm-foundation
  - ingridflogny/skindetector-derm-probe
---

# SkinDetector

Application qui attribue un **score de malignité** à la photo d'une lésion cutanée, prise
au smartphone ou au dermatoscope, avec Derm Foundation (Google) et le classifieur
[`ingridflogny/skindetector-derm-probe`](https://huggingface.co/ingridflogny/skindetector-derm-probe).

> **Avertissement.** Projet de recherche et d'apprentissage. Ce n'est **pas un dispositif
> médical** : ni marquage CE, ni validation clinique, aucun diagnostic. Un score bas
> n'exclut pas une lésion maligne. Ne pas utiliser pour prendre une décision médicale.

## Fonctionnement

- `space.py` : lance l'application FastAPI sur le port du Space ;
- `app/` : l'application elle-même (API, page web, modèle), identique à la version locale
  du projet ;
- `app/probe.npz` : le classifieur, dérivé de HAI-DEF (voir `NOTICE`).

Le Space est déclaré de type Gradio pour rester sur le matériel gratuit, mais il n'utilise
pas l'interface de Gradio : il sert la page de l'application, où l'on encadre la lésion à
la souris ou au doigt. L'API est aussi accessible : `POST /api/analyses`.

Le secret `HF_TOKEN` du Space donne accès à Derm Foundation, téléchargé au démarrage
(environ 1,1 Go). Les photos ne sont gardées que le temps de l'analyse ; l'application ne
les conserve ni ne les journalise.

## Licence et conditions

HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms of Use
found at https://developers.google.com/health-ai-developer-foundations/terms. Une copie de
ces conditions est jointe (`HAI-DEF-TERMS.md`). Utiliser cette application vaut acceptation
de leurs restrictions d'usage (section 3.2). Usage non commercial uniquement (ISIC 2020,
CC BY-NC 4.0).
