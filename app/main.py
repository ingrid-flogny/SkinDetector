from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from app.skin import SkinAnalysis, Source, get_classifier

STATIC_DIR = Path(__file__).parent / "static"


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
def create_analysis(image: UploadFile, source: Source = Form()) -> SkinAnalysis:
    try:
        photo = ImageOps.exif_transpose(Image.open(image.file))
        photo.load()
    except Exception as e:
        # La photo est une donnée de santé : rien n'est journalisé ni enregistré.
        raise HTTPException(422, "Image illisible ou format non pris en charge.") from e
    return get_classifier().analyze(photo, source)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
