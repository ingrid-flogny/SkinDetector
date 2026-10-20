"""Premier modèle de SkinDetector, conservé pour pouvoir y revenir.

Pour le réactiver : construire l'image avec --build-arg EXTRAS=ham10000 (PyTorch et
timm, poids embarqués), puis démarrer avec SKIN_MODEL=ham10000-efficientnet-b0. Il
fonctionne alors sans Derm Foundation ni HF_TOKEN.

Résultats mesurés (training/results/), à comparer avec app/derm_foundation.py :
- ISIC 2020 (dermatoscope, test, 6 162 images dont 106 mélanomes) : AUC 0,64
  (0,58 à 0,69) ; spécificité de 8 % à 95 % de sensibilité ;
- PAD-UFES-20 (smartphone, test, 453 images) : AUC 0,84 (0,80 à 0,88) ; spécificité
  de 36 % à 95 % de sensibilité.
"""

from PIL import Image

from app.skin import ClassScore, SkinAnalysis, Source


class Ham10000EfficientNet:
    """EfficientNet-B0 affiné sur HAM10000 (images de dermatoscope, 7 types de lésion).

    Projet éducatif sous licence MIT, non validé cliniquement : rappel du mélanome
    de 0,58 sur son propre jeu de test.
    """

    REPO = "pthennakoon25/skinvision-ai-model"
    # Révision figée : les poids ne changent pas sans modification explicite ici.
    REVISION = "9429e5a4ef81bb90064d8cdacc5895c4fc442cf7"
    FILENAME = "best_model.pth"
    IMAGE_SIZE = 224
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    # Dans l'ordre des sorties du modèle. Le score de malignité est la somme des
    # classes marquées True : mélanome, carcinome basocellulaire, et kératose
    # actinique / carcinome intra-épithélial (lésion précancéreuse ou in situ).
    CLASSES = (
        ("akiec", "Kératose actinique / carcinome intra-épithélial", True),
        ("bcc", "Carcinome basocellulaire", True),
        ("bkl", "Kératose bénigne", False),
        ("df", "Dermatofibrome", False),
        ("mel", "Mélanome", True),
        ("nv", "Nævus mélanocytaire", False),
        ("vasc", "Lésion vasculaire", False),
    )

    def __init__(self) -> None:
        # Imports différés, comme pour Whisper : PyTorch est lourd à charger.
        import timm
        import torch
        from huggingface_hub import hf_hub_download

        self._torch = torch
        path = hf_hub_download(self.REPO, self.FILENAME, revision=self.REVISION)
        # weights_only : le fichier vient d'un tiers, on n'en charge que des tenseurs.
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Les auteurs enveloppent le réseau timm dans un module `backbone`.
        state = state.get("model_state_dict", state)
        state = {key.removeprefix("backbone."): value for key, value in state.items()}
        self._net = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=len(self.CLASSES)
        )
        self._net.load_state_dict(state)
        self._net.eval()

    def analyze(self, image: Image.Image, source: Source) -> SkinAnalysis:
        # `source` est ignoré : ce modèle n'a pas de seuil d'alerte par type de photo.
        import numpy as np

        torch = self._torch
        resized = image.convert("RGB").resize(
            (self.IMAGE_SIZE, self.IMAGE_SIZE), Image.Resampling.BILINEAR
        )
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        pixels = (pixels - np.array(self.MEAN, dtype=np.float32)) / np.array(
            self.STD, dtype=np.float32
        )
        batch = torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(self._net(batch), dim=1)[0].tolist()

        classes = [
            ClassScore(code=code, label=label, probability=p, malignant=malignant)
            for (code, label, malignant), p in zip(self.CLASSES, probabilities)
        ]
        return SkinAnalysis(
            model=f"{self.REPO} (EfficientNet-B0, HAM10000)",
            malignancy_score=sum(c.probability for c in classes if c.malignant),
            classes=sorted(classes, key=lambda c: c.probability, reverse=True),
        )
