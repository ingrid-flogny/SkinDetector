import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from app.skin import SkinAnalysis, Source, get_classifier

STATIC_DIR = Path(__file__).parent / "static"

# Sur un Space Hugging Face (SPACE_ID est défini par la plateforme), la page précise où
# les photos sont traitées, et cite le classifieur publié et les conditions HAI-DEF.
HOSTED = bool(os.getenv("SPACE_ID"))
HOSTED_NOTICE = """<section class="hosted">
    <p><strong>Démonstration hébergée sur Hugging Face.</strong> Les photos sont analysées sur
    ses serveurs ; l'application ne les conserve ni ne les journalise. N'envoyez pas de photo
    permettant d'identifier une personne. Ce n'est <strong>pas un dispositif médical</strong> :
    en cas de doute sur une lésion, consultez un médecin.</p>
    <p class="note">Classifieur : <a href="https://huggingface.co/ingridflogny/skindetector-derm-probe">ingridflogny/skindetector-derm-probe</a>.
    HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms of Use found at
    <a href="https://developers.google.com/health-ai-developer-foundations/terms">https://developers.google.com/health-ai-developer-foundations/terms</a>.</p>
  </section>"""


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Modèle chargé au démarrage (environ une minute) plutôt qu'à la première photo.
    get_classifier()
    yield


app = FastAPI(title="SkinDetector", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


# `def` et non `async def` : l'analyse est bloquante, FastAPI l'exécute dans un thread.
@app.post("/api/analyses")
def create_analysis(
    image: UploadFile,
    source: Source = Form(),
    # Cadre facultatif autour de la lésion, en pixels de la photo une fois redressée
    # (orientation EXIF appliquée) : le modèle a été entraîné sur des gros plans.
    crop_x: int | None = Form(None, ge=0),
    crop_y: int | None = Form(None, ge=0),
    crop_width: int | None = Form(None, gt=0),
    crop_height: int | None = Form(None, gt=0),
) -> SkinAnalysis:
    try:
        photo = ImageOps.exif_transpose(Image.open(image.file))
        photo.load()
    except Exception as e:
        # La photo est une donnée de santé : rien n'est journalisé ni enregistré.
        raise HTTPException(422, "Image illisible ou format non pris en charge.") from e

    crop = (crop_x, crop_y, crop_width, crop_height)
    if any(v is not None for v in crop):
        if None in crop:
            raise HTTPException(422, "Cadre incomplet : crop_x, crop_y, crop_width et crop_height sont requis ensemble.")
        if crop_x + crop_width > photo.width or crop_y + crop_height > photo.height:
            raise HTTPException(422, "Le cadre dépasse de la photo.")
        # Découpé dans la photo d'origine, en pleine résolution.
        photo = photo.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
    return get_classifier().analyze(photo, source)


@app.get("/")
def index() -> HTMLResponse:
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(page.replace("<!-- avis-hebergement -->", HOSTED_NOTICE if HOSTED else ""))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
