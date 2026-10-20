"""Interface Gradio de SkinDetector pour un Space Hugging Face.

Même analyse que l'application FastAPI (app/main.py) : le modèle vient de app/skin.py,
inchangé ; seule l'interface diffère. HF_TOKEN (secret du Space) donne accès à Derm
Foundation, téléchargé au démarrage.
"""

import gradio as gr
from PIL import Image, ImageOps

from app.skin import get_classifier

DISCLAIMER = (
    "**Avertissement.** Modèle expérimental, non validé cliniquement, qui n'est **pas un "
    "dispositif médical**. Le score ne constitue pas un diagnostic : un score bas n'exclut "
    "pas une lésion maligne. En cas de doute sur une lésion, consultez un médecin.\n\n"
    "Les photos sont analysées sur les serveurs de Hugging Face. L'application ne les "
    "journalise pas, et la copie temporaire que garde Gradio est effacée après 5 minutes. "
    "N'envoyez pas de photo permettant d'identifier une personne."
)

# Gradio garde une copie de chaque fichier envoyé : effacée toutes les 5 minutes si elle
# a plus de 5 minutes, au lieu d'être conservée jusqu'au redémarrage du Space.
DELETE_CACHE = (300, 300)

# Modèle chargé au démarrage (environ une minute) plutôt qu'à la première photo.
classifier = get_classifier()


def percent(p: float) -> str:
    return f"{p * 100:.1f} %".replace(".", ",")


def analyze(edited: dict | None, source: str) -> str:
    # Image après recadrage éventuel ; toute la photo si l'utilisateur n'a pas recadré.
    path = (edited or {}).get("composite")
    if path is None:
        raise gr.Error("Déposez d'abord une photo.")
    try:
        # Fichier d'origine, lu comme le fait l'application FastAPI.
        photo = ImageOps.exif_transpose(Image.open(path))
        photo.load()
    except Exception as e:
        raise gr.Error("Image illisible ou format non pris en charge.") from e
    result = classifier.analyze(photo, source)

    above = result.malignancy_score >= result.alert_threshold
    return (
        f"## Score de malignité : {percent(result.malignancy_score)}\n\n"
        "Score de 0 à 100 %. Ce n'est pas une probabilité : il se lit par rapport au seuil "
        "d'alerte.\n\n"
        f"**{'Au-dessus' if above else 'En dessous'} du seuil d'alerte** "
        f"({percent(result.alert_threshold)}). Ce seuil détectait 95 % des lésions malignes "
        "en validation, au prix de fausses alertes.\n\n"
        f"*Modèle : {result.model}.*"
    )


with gr.Blocks(title="SkinDetector", delete_cache=DELETE_CACHE) as demo:
    gr.Markdown(
        "# SkinDetector\n\n"
        "Déposez la photo d'une lésion cutanée pour obtenir un score de malignité."
    )
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        with gr.Column():
            source = gr.Radio(
                [("Smartphone", "smartphone"), ("Dermatoscope", "dermatoscope")],
                value="smartphone",
                label="Type de photo",
                info="Le type de photo règle le seuil d'alerte : les deux sources ne donnent "
                "pas les mêmes scores.",
            )
            # Éditeur réduit au recadrage. Le canevas prend la taille de la photo (pas de
            # réduction), et le PNG évite la compression avec perte du WebP par défaut.
            # Mode RGB : en RGBA (défaut), Gradio ne peut pas réenregistrer une photo JPEG.
            photo = gr.ImageEditor(
                type="filepath",
                format="png",
                image_mode="RGB",
                label="Photo",
                sources=["upload", "webcam"],
                transforms=("crop",),
                brush=False,
                eraser=False,
                layers=False,
            )
            gr.Markdown(
                "Recadrez sur la lésion à analyser (outil de recadrage de l'éditeur), en "
                "laissant un peu de peau autour. Le modèle a été entraîné sur des gros plans : "
                "sur une photo large (une main entière, plusieurs grains de beauté), gardez une "
                "seule lésion. Sans recadrage, toute la photo est analysée."
            )
            button = gr.Button("Analyser", variant="primary")
        with gr.Column():
            output = gr.Markdown()
    button.click(analyze, inputs=[photo, source], outputs=output)
    gr.Markdown(
        "Derm Foundation (Google) + régression logistique entraînée sur PAD-UFES-20 et "
        "ISIC 2020. Classifieur : "
        "[ingridflogny/skindetector-derm-probe](https://huggingface.co/ingridflogny/skindetector-derm-probe). "
        "HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms "
        "of Use found at https://developers.google.com/health-ai-developer-foundations/terms."
    )

if __name__ == "__main__":
    demo.launch()
