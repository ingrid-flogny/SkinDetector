"""Déploie l'application sur le Space public `<compte>/SkinDetector`.

Les Spaces Gradio et Docker demandent désormais un abonnement payant à la création. Le
Space réutilise donc un Space Gradio créé avant ce changement (Final_Assignment_Agent_Course,
sur CPU Basic gratuit), renommé SkinDetector au premier déploiement. Il exécute
huggingface/space/space.py, qui lance l'application FastAPI elle-même : même API et même
page que la version locale.

Fichiers envoyés : huggingface/space/ (lanceur, dépendances, fiche), l'application
(app/main.py, app/static/index.html, app/skin.py, app/derm_foundation.py, app/probe.npz,
inchangés), la NOTICE et la copie des conditions HAI-DEF. Les autres fichiers du Space
sont supprimés.

Jetons lus dans l'environnement : HF_WRITE_TOKEN (écriture, pour modifier le Space) et
HF_TOKEN (lecture, transmis au Space comme secret pour télécharger Derm Foundation).
Relançable : chaque déploiement remplace les fichiers et redémarre le Space.

    python huggingface/publish_space.py
"""

import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent
SPACE_NAME = "SkinDetector"
FORMER_NAME = "Final_Assignment_Agent_Course"
FILES = {
    "README.md": HERE / "space" / "README.md",
    "space.py": HERE / "space" / "space.py",
    "requirements.txt": HERE / "space" / "requirements.txt",
    "app/__init__.py": ROOT / "app" / "__init__.py",
    "app/main.py": ROOT / "app" / "main.py",
    "app/static/index.html": ROOT / "app" / "static" / "index.html",
    "app/skin.py": ROOT / "app" / "skin.py",
    "app/derm_foundation.py": ROOT / "app" / "derm_foundation.py",
    "app/probe.npz": ROOT / "app" / "probe.npz",
    "NOTICE": HERE / "model" / "NOTICE",
    "HAI-DEF-TERMS.md": HERE / "model" / "HAI-DEF-TERMS.md",
}


def main() -> None:
    api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
    user = api.whoami()["name"]
    space = f"{user}/{SPACE_NAME}"

    if not api.repo_exists(space, repo_type="space"):
        api.move_repo(f"{user}/{FORMER_NAME}", space, repo_type="space")
    api.add_space_secret(space, "HF_TOKEN", os.environ["HF_TOKEN"])
    # Fichiers d'un déploiement précédent qui ne sont plus envoyés (ancienne interface…).
    obsolete = [
        f for f in api.list_repo_files(space, repo_type="space")
        if f not in FILES and f != ".gitattributes"
    ]
    api.create_commit(
        space,
        repo_type="space",
        operations=[
            *(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in FILES.items()),
            *(CommitOperationDelete(path_in_repo=name) for name in obsolete),
        ],
        commit_message="Déploie l'application SkinDetector",
    )
    print(f"Space : https://huggingface.co/spaces/{space}" + (f" (supprimés : {', '.join(obsolete)})" if obsolete else ""))


if __name__ == "__main__":
    main()
