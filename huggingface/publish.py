"""Publie le classifieur sur Hugging Face.

Dépôt modèle `<compte>/skindetector-derm-probe` : app/probe.npz, sa fiche, la NOTICE et
la copie des conditions HAI-DEF (contenu de huggingface/model/). Public, sans demande
d'accès : les conditions d'utilisation sont dans la fiche, la NOTICE et la copie jointe.

Jeton lu dans l'environnement : HF_WRITE_TOKEN (écriture, pour créer et remplir le
dépôt). Relançable : un dépôt existant est mis à jour.

    python huggingface/publish.py
"""

import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent
MODEL_NAME = "skindetector-derm-probe"


def add(path: Path, path_in_repo: str) -> CommitOperationAdd:
    return CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(path))


def main() -> None:
    api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
    model_repo = f"{api.whoami()['name']}/{MODEL_NAME}"

    # Créé privé, rempli, puis rendu public : jamais visible sans sa fiche et ses
    # conditions.
    api.create_repo(model_repo, private=True, exist_ok=True)
    api.create_commit(
        model_repo,
        operations=[
            *(add(f, f.name) for f in sorted((HERE / "model").iterdir())),
            add(ROOT / "app" / "probe.npz", "probe.npz"),
        ],
        commit_message="Publie le classifieur SkinDetector",
    )
    api.update_repo_settings(model_repo, gated=False, private=False)
    print(f"Modèle : https://huggingface.co/{model_repo}")


if __name__ == "__main__":
    main()
