"""Calcule les embeddings Derm Foundation des images d'un jeu de données.

Chaque image devient un vecteur de 6 144 nombres, enregistré dans
training/data/embeddings/derm_foundation/<jeu>.npz. Le calcul est relançable : il
reprend là où il s'est arrêté.

    python embed.py pad_ufes_20 [--limit N] [--batch-size N]

Sur un GPU, prendre des lots plus gros (--batch-size 32) : la préparation des images,
faite en parallèle sur tous les cœurs, devient alors l'étape la plus lente.
"""

import argparse
import csv
import io
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download
from PIL import Image
from tensorflow.core.protobuf import saved_model_pb2

DATA = Path(__file__).parent / "data"
OUT = DATA / "embeddings" / "derm_foundation"
IMAGE_SIZE = 448
SAVE_EVERY = 200  # images


def to_example(path: Path) -> bytes:
    """Met l'image au format attendu par le modèle : PNG 448x448 dans un tf.train.Example."""
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    resized.save(buffer, "PNG")
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[buffer.getvalue()]))
    return tf.train.Example(
        features=tf.train.Features(feature={"image/encoded": feature})
    ).SerializeToString()


def load_model(model_dir: Path):
    """Charge Derm Foundation en l'autorisant à tourner sur GPU.

    Le modèle publié a été exporté depuis JAX pour le seul CPU : ses nœuds XlaCallModule
    refusent tout autre matériel (« The current platform CUDA is not among the platforms
    required by the module »). Désactiver ce contrôle, comme le prévoit jax2tf, permet à
    TensorFlow de compiler le même calcul pour le GPU. La copie modifiée est écrite à part,
    le téléchargement d'origine reste intact.
    """
    saved_model = saved_model_pb2.SavedModel()
    saved_model.ParseFromString((model_dir / "saved_model.pb").read_bytes())
    for graph in saved_model.meta_graphs:
        functions = graph.graph_def.library.function
        for node in [*graph.graph_def.node, *(n for f in functions for n in f.node_def)]:
            checks = node.attr["disabled_checks"].list.s if node.op == "XlaCallModule" else None
            if checks is not None and b"platform" not in checks:
                checks.append(b"platform")
    patched = Path(tempfile.mkdtemp(prefix="derm-foundation-"))
    (patched / "saved_model.pb").write_bytes(saved_model.SerializeToString())
    (patched / "variables").symlink_to((model_dir / "variables").resolve())
    return tf.saved_model.load(str(patched))


def save(path: Path, ids: list[str], vectors: list[np.ndarray]) -> None:
    partial = path.with_suffix(".part.npz")
    np.savez(partial, image_id=np.array(ids), embedding=np.stack(vectors).astype(np.float32))
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--limit", type=int, help="ne traiter que les N premières images")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    with open(DATA / "manifest.csv", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["dataset"] == args.dataset]
    rows = rows[: args.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.dataset}.npz"
    ids, vectors = [], []
    if path.exists():
        done = np.load(path)
        ids, vectors = list(done["image_id"]), list(done["embedding"])
    known = set(ids)
    todo = [row for row in rows if row["image_id"] not in known]
    print(f"{len(known)} déjà calculés, {len(todo)} à faire", flush=True)
    if not todo:
        return

    model = load_model(Path(
        snapshot_download("google/derm-foundation", allow_patterns=["*.pb", "variables/*"])
    ))
    infer = model.signatures["serving_default"]

    start = time.time()
    last_save = 0
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        for offset in range(0, len(todo), args.batch_size):
            batch = todo[offset : offset + args.batch_size]
            examples = tf.constant(list(pool.map(lambda row: to_example(DATA / row["path"]), batch)))
            vectors.extend(infer(inputs=examples)["embedding"].numpy())
            ids.extend(row["image_id"] for row in batch)
            done_count = offset + len(batch)
            if done_count - last_save >= SAVE_EVERY:
                last_save = done_count
                save(path, ids, vectors)
                rate = done_count / (time.time() - start)
                print(f"{done_count}/{len(todo)} — {rate:.2f} images/s, "
                      f"reste {(len(todo) - done_count) / rate / 60:.0f} min", flush=True)
    save(path, ids, vectors)
    print(f"terminé : {len(ids)} embeddings dans {path}", flush=True)


if __name__ == "__main__":
    main()
