---
title: SkinDetector
emoji: 🔬
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: interface.py
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

- `interface.py` : interface Gradio ;
- `app/` : le modèle, identique à celui de l'application FastAPI du projet ;
- `app/probe.npz` : le classifieur, dérivé de HAI-DEF (voir `NOTICE`).

Le secret `HF_TOKEN` du Space donne accès à Derm Foundation, téléchargé au démarrage
(environ 1,1 Go). Les photos ne sont pas journalisées ; la copie temporaire que garde
Gradio est effacée après 5 minutes.

## Licence et conditions

HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms of Use
found at https://developers.google.com/health-ai-developer-foundations/terms. Une copie de
ces conditions est jointe (`HAI-DEF-TERMS.md`). Utiliser cette application vaut acceptation
de leurs restrictions d'usage (section 3.2). Usage non commercial uniquement (ISIC 2020,
CC BY-NC 4.0).
