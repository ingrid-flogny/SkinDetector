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


def load_photo(path: str) -> Image.Image:
    """Photo d'origine, redressée selon son orientation EXIF comme dans l'application FastAPI."""
    try:
        photo = ImageOps.exif_transpose(Image.open(path))
        photo.load()
    except Exception as e:
        raise gr.Error("Image illisible ou format non pris en charge.") from e
    return photo


def select_corner(path: str | None, corner: tuple | None, evt: gr.SelectData):
    """Deux clics sur la photo : deux coins opposés du cadre autour de la lésion.

    Les coordonnées données par Gradio sont en pixels de la photo affichée, donc redressée :
    le cadre est découpé dans la photo d'origine, en pleine résolution.
    """
    if path is None:
        return None, None, None, ""
    x, y = evt.index
    if corner is None:
        return (x, y), None, None, "Premier coin placé : cliquez sur le coin opposé."
    photo = load_photo(path)
    box = (min(corner[0], x), min(corner[1], y), max(corner[0], x), max(corner[1], y))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None, None, None, "Cadre trop petit : recommencez en cliquant sur deux coins opposés."
    return None, box, photo.crop(box), f"Cadre de {box[2] - box[0]} × {box[3] - box[1]} pixels."


def reset_crop():
    return None, None, None, ""


def analyze(path: str | None, box: tuple | None, source: str) -> str:
    if path is None:
        raise gr.Error("Déposez d'abord une photo.")
    photo = load_photo(path)
    if box is not None:
        photo = photo.crop(box)
    result = classifier.analyze(photo, source)

    above = result.malignancy_score >= result.alert_threshold
    scope = (
        f"le cadre de {box[2] - box[0]} × {box[3] - box[1]} pixels" if box is not None
        else "la photo entière"
    )
    return (
        f"## Score de malignité : {percent(result.malignancy_score)}\n\n"
        "Score de 0 à 100 %. Ce n'est pas une probabilité : il se lit par rapport au seuil "
        "d'alerte.\n\n"
        f"**{'Au-dessus' if above else 'En dessous'} du seuil d'alerte** "
        f"({percent(result.alert_threshold)}). Ce seuil détectait 95 % des lésions malignes "
        "en validation, au prix de fausses alertes.\n\n"
        f"*Zone analysée : {scope}. Modèle : {result.model}.*"
    )


with gr.Blocks(title="SkinDetector", delete_cache=DELETE_CACHE) as demo:
    gr.Markdown(
        "# SkinDetector\n\n"
        "Déposez la photo d'une lésion cutanée pour obtenir un score de malignité."
    )
    gr.Markdown(DISCLAIMER)
    # Premier coin cliqué, en attente du second, puis cadre (gauche, haut, droite, bas).
    corner = gr.State(None)
    box = gr.State(None)
    with gr.Row():
        with gr.Column():
            source = gr.Radio(
                [("Smartphone", "smartphone"), ("Dermatoscope", "dermatoscope")],
                value="smartphone",
                label="Type de photo",
                info="Le type de photo règle le seuil d'alerte : les deux sources ne donnent "
                "pas les mêmes scores.",
            )
            # Composant image standard plutôt que l'éditeur de Gradio 6, dont la barre
            # d'outils disparaît quand seul le recadrage est activé.
            photo = gr.Image(type="filepath", label="Photo", sources=["upload", "webcam"])
            gr.Markdown(
                "Pour isoler une lésion, **cliquez sur deux coins opposés** autour d'elle, en "
                "laissant un peu de peau autour. Le modèle a été entraîné sur des gros plans : "
                "sur une photo large (une main entière, plusieurs grains de beauté), encadrez "
                "une seule lésion. Sans cadre, toute la photo est analysée."
            )
            crop_status = gr.Markdown()
            preview = gr.Image(label="Zone qui sera analysée", interactive=False, format="png")
            with gr.Row():
                button = gr.Button("Analyser", variant="primary")
                clear = gr.Button("Effacer le cadre")
        with gr.Column():
            output = gr.Markdown()
    crop_outputs = [corner, box, preview, crop_status]
    photo.select(select_corner, inputs=[photo, corner], outputs=crop_outputs)
    photo.change(reset_crop, outputs=crop_outputs)
    clear.click(reset_crop, outputs=crop_outputs)
    button.click(analyze, inputs=[photo, box, source], outputs=output)
    gr.Markdown(
        "Derm Foundation (Google) + régression logistique entraînée sur PAD-UFES-20 et "
        "ISIC 2020. Classifieur : "
        "[ingridflogny/skindetector-derm-probe](https://huggingface.co/ingridflogny/skindetector-derm-probe). "
        "HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms "
        "of Use found at https://developers.google.com/health-ai-developer-foundations/terms."
    )

if __name__ == "__main__":
    demo.launch()
