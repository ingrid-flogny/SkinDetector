"""Télécharge PAD-UFES-20 (photos de smartphone, licence CC-BY) depuis l'archive ISIC.

Écrit training/data/pad_ufes_20/metadata.csv et les images dans images/.
Relançable : les images déjà présentes ne sont pas retéléchargées.
"""

import csv
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://api.isic-archive.com/api/v2/images/search/?collections=406&limit=100"
OUT = Path(__file__).parent / "data" / "pad_ufes_20"
FIELDS = ["isic_id", "patient_id", "lesion_id", "diagnosis_1", "diagnosis_2", "diagnosis_3",
          "diagnosis_confirm_type", "image_type", "url"]


def list_images() -> list[dict]:
    rows, url = [], API
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            page = json.load(response)
        for item in page["results"]:
            clinical = item["metadata"]["clinical"]
            rows.append({
                "isic_id": item["isic_id"],
                "image_type": item["metadata"]["acquisition"].get("image_type", ""),
                "url": item["files"]["full"]["url"],
                **{key: clinical.get(key, "") for key in FIELDS[1:7]},
            })
        url = page["next"]
    return rows


def download(row: dict) -> None:
    target = OUT / "images" / f"{row['isic_id']}.jpg"
    if not target.exists():
        partial = target.with_suffix(".part")
        urllib.request.urlretrieve(row["url"], partial)
        partial.rename(target)


def main() -> None:
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows = list_images()
    with open(OUT / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, rows))
    print(f"{len(rows)} images décrites, {len(list((OUT / 'images').glob('*.jpg')))} présentes")


if __name__ == "__main__":
    main()
