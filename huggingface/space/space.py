"""Lance l'application FastAPI de SkinDetector sur le Space Hugging Face.

Le Space est déclaré de type Gradio : c'est un Space créé avant que Hugging Face rende
payants les Spaces Gradio et Docker, et il le reste pour garder la gratuité. La
plateforme se contente d'exécuter ce fichier et d'exposer le port 7860 : le Space sert
donc la même application et la même page que la version locale (app/main.py), avec le
cadre tracé à la souris ou au doigt. L'éditeur d'image de Gradio 6 n'offrait pas de
recadrage utilisable.
"""

import uvicorn

from app.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
