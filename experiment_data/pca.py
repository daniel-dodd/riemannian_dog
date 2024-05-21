"""PCA datasets."""
from dataclasses import (
    dataclass,
    field,
)
import glob
import io
import os
import zipfile

import cv2
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)
import requests
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from experiment_utils.dataset import Dataset

__all__ = [
    "PCADataset",
    "iris",
    "wine",
    "glass",
    "eeg_eye_state",
    "waveform",
    "tiny_image_net",
]

dir, _ = os.path.split(__file__)


def _local_path(filename: str) -> str:
    """Return the local path of a file."""
    return os.path.join(dir, filename)


@dataclass
class PCADataset(Dataset):
    """Load dataset and create PCA problem."""

    num_components: int = None
    name: str = None
    minimiser: float = None
    minimum_loss: float = None

    def _preprocess(self, X: Float[Array, "N *"]) -> None:
        # Standardize the data.
        ss = StandardScaler()
        data = ss.fit_transform(X)
        self.X = jnp.array(data)

        # Use sklearn PCA to reduce the dimensionality.
        pca = PCA(self.num_components, svd_solver="full")
        pca.fit(data)
        minimiser = pca.components_.T
        self.minimiser = minimiser

        # Compute the minimum loss.
        minimum_loss = (
            jnp.linalg.norm(data.T - minimiser @ (minimiser.T @ data.T)) ** 2
        ) / data.shape[0]
        self.minimum_loss = minimum_loss


@dataclass
class _SklearnPCADataset(PCADataset):
    """Load dataset and create PCA problem via sklearn."""

    num_components: int = 1
    name: str = field(init=False, repr=False)
    minimiser: float = field(init=False)
    minimum_loss: float = field(init=False)

    def __post_init__(self):
        self.X, _ = datasets.fetch_openml(
            self.name, version=1, return_X_y=True, as_frame=False
        )
        self._preprocess(self.X)


def _fetch_tiny_image_net(head: int) -> Float[Array, "N *"]:
    """Fetch tiny image net dataset.

    Args:
        head: Number of images to fetch.

    Returns:
        Array of images.
    """

    path = _local_path("_cache/tiny-imagenet-200")

    # Download the dataset if it does not exist locally.
    if not os.path.exists(path):
        # Create a directory to cache the dataset if it does not exist.
        if not os.path.exists(_local_path("_cache")):
            os.makedirs(_local_path("_cache"))

        # Download the dataset.
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        response = requests.get(url)
        file = zipfile.ZipFile(io.BytesIO(response.content))
        file.extractall(_local_path("_cache"))

    # Load the images.
    image_path = _local_path(path + "/train/*/images/*.JPEG")
    indices = sorted(glob.glob(image_path))[:head]
    images = [cv2.imread(file) for file in tqdm(indices)]
    data = jnp.asarray(images[:head], dtype=jnp.float64).reshape(head, -1) / 255

    return data


@dataclass
class _TinyImageNetPCADataset(PCADataset):
    """Load tiny image dataset."""

    num_components: int = 1
    num_images: int = 5000
    name: str = field(init=False, repr=False)
    minimiser: float = field(init=False)
    minimum_loss: float = field(init=False)

    def __post_init__(self):
        X = _fetch_tiny_image_net(self.num_images)
        self._preprocess(X)


def iris(num_components: int = 1) -> PCADataset:
    """Load iris dataset.

    Args:
        num_components: Number of PCA components to reduce to.

    Returns:
        The iris dataset centered and reduced to `num_components` dimensions.
    """
    return type(
        "Iris",
        (_SklearnPCADataset,),
        {"num_components": num_components, "name": "iris"},
    )(num_components=num_components)


def wine(num_components: int = 1) -> PCADataset:
    """Load wine dataset.

    Args:
        num_components: Number of PCA components to reduce to.

    Returns:
        The wine dataset centered and reduced to `num_components` dimensions.
    """
    return type(
        "Wine",
        (_SklearnPCADataset,),
        {"num_components": num_components, "name": "wine"},
    )(num_components=num_components)


def glass(num_components: int = 1) -> PCADataset:
    """Load glass dataset.

    Args:
        num_components: Number of PCA components to reduce to.

    Returns:
        The glass dataset centered and reduced to `num_components` dimensions.
    """
    return type(
        "Glass",
        (_SklearnPCADataset,),
        {"num_components": num_components, "name": "glass"},
    )(num_components=num_components)


def eeg_eye_state(num_components: int = 1) -> PCADataset:
    """Load eeg eye state dataset.

    Args:
        num_components: Number of PCA components to reduce to.

    Returns:
        The eeg eye state dataset centered and reduced to `num_components` dimensions.
    """
    return type(
        "EEGEyeState",
        (_SklearnPCADataset,),
        {"num_components": num_components, "name": "eeg-eye-state"},
    )(num_components=num_components)


def waveform(num_components: int = 1) -> PCADataset:
    """Load waveform dataset.

    Args:
        num_components: Number of PCA components to reduce to.

    Returns:
        The waveform dataset centered and reduced to `num_components` dimensions.
    """
    return type(
        "Waveform",
        (_SklearnPCADataset,),
        {"num_components": num_components, "name": "waveform-5000"},
    )(num_components=num_components)


def tiny_image_net(num_components: int = 1, num_images: int = 5000) -> PCADataset:
    """Load tiny image net dataset.

    Args:
        num_components: Number of PCA components to reduce to.
        num_images: Number of images to load.

    Returns:
        The tiny image net dataset centered and reduced to `num_components` dimensions.
    """
    return type(
        "TinyImageNet",
        (_TinyImageNetPCADataset,),
        {"num_components": num_components, "num_images": num_images},
    )(num_components=num_components, num_images=num_images)
