"""
Data loader utilities for the Kaggle Dogs vs Cats dataset.

Expected dataset layout (either flat or nested):

  Flat (original Kaggle format):
    <root>/train/
        cat.0.jpg
        cat.1.jpg
        dog.0.jpg
        dog.1.jpg
        ...

  Nested:
    <root>/train/cats/  *.jpg
    <root>/train/dogs/  *.jpg

Usage:
    dataset = get_animal_dataset(
        root      = "C:/Users/26384/Downloads/dogs-vs-cats",
        transform = transforms.Compose([...]),
        animal    = "dog",   # or "cat"
    )
"""

import os
from pathlib import Path
from typing import Callable, Literal, Optional

from PIL import Image
from torch.utils.data import Dataset


# ── Core Dataset ───────────────────────────────────────────────────────────────

class AnimalDataset(Dataset):
    """
    Loads images of a single animal class from the Dogs vs Cats dataset.

    Args:
        root      : path to the dataset root (the folder that contains 'train/')
        animal    : "dog" or "cat"
        transform : torchvision transform to apply
        split     : subfolder name, default "train"
    """

    def __init__(
        self,
        root: str,
        animal: Literal["dog", "cat"] = "dog",
        transform: Optional[Callable] = None,
        split: str = "train",
    ):
        self.transform = transform
        self.animal    = animal.lower()
        self.paths     = []

        root_path  = Path(root)
        split_path = root_path / split

        if not split_path.exists():
            # Some downloads unzip directly without a 'train' sub-folder
            split_path = root_path

        # ── nested layout: train/dogs/ or train/cats/ ────────────────────────
        nested_dir = split_path / (self.animal + "s")
        if nested_dir.exists():
            self.paths = sorted(nested_dir.glob("*.jpg")) + \
                         sorted(nested_dir.glob("*.jpeg")) + \
                         sorted(nested_dir.glob("*.png"))

        # ── flat layout: dog.0.jpg, cat.1.jpg … ─────────────────────────────
        else:
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for p in split_path.glob(ext):
                    if p.stem.lower().startswith(self.animal):
                        self.paths.append(p)
            self.paths = sorted(self.paths)

        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No images found for '{self.animal}' under '{split_path}'.\n"
                f"Check that the dataset is extracted and the animal name is correct."
            )

        print(f"[AnimalDataset] Found {len(self.paths)} '{self.animal}' images in '{split_path}'")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


# ── Public helper ──────────────────────────────────────────────────────────────

def get_animal_dataset(
    root: str,
    transform: Optional[Callable] = None,
    animal: str = "dog",
    split: str = "train",
) -> AnimalDataset:
    """
    Convenience factory used by train.py.

    Args:
        root      : dataset root folder (e.g. "C:/Users/.../dogs-vs-cats")
        transform : torchvision transform pipeline
        animal    : "dog" or "cat"
        split     : dataset split sub-folder ("train", "test", etc.)
    """
    return AnimalDataset(root=root, animal=animal, transform=transform, split=split)


# ── Legacy alias (keeps compatibility with old train.py that called get_celeba_dataset) ──
def get_celeba_dataset(root: str, transform: Optional[Callable] = None) -> AnimalDataset:
    """Deprecated alias — use get_animal_dataset instead."""
    import warnings
    warnings.warn(
        "get_celeba_dataset() is deprecated. Use get_animal_dataset() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_animal_dataset(root, transform, animal="dog")


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(128),
        T.CenterCrop(128),
        T.ToTensor(),
    ])

    # Replace with your actual path
    ds = get_animal_dataset(
        root="C:/Users/26384/Downloads/dogs-vs-cats",
        transform=transform,
        animal="dog",
    )
    img = ds[0]
    print("Image tensor shape:", img.shape)   # (3, 128, 128)