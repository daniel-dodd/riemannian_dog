"""Dataset. Thanks to GPJax! (Except for we drop the Pytree stuff.)"""
from dataclasses import dataclass
from typing import (
    Optional,
    Tuple,
    Union,
)
import warnings

from jax import random
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
    Key,
)

__all__ = [
    "Dataset",
    "train_test_split",
    "get_batch",
]


@dataclass
class Dataset:
    r"""Base class for datasets.

    Attributes
    ----------
        X: input data.
        y: output data.
    """

    X: Optional[Float[Array, "N D"]] = None
    y: Optional[Float[Array, "N Q"]] = None

    def __post_init__(self) -> None:
        r"""Checks that the shapes of $`X`$ and $`y`$ are compatible,
        and provides warnings regarding the precision of $`X`$ and $`y`$."""
        _check_shape(self.X, self.y)
        _check_precision(self.X, self.y)

    def __repr__(self) -> str:
        r"""Returns a string representation of the dataset."""
        repr = f"- Number of observations: {self.n}\n- Input dimension: {self.in_dim}"
        return repr

    def is_supervised(self) -> bool:
        r"""Returns `True` if the dataset is supervised."""
        return self.X is not None and self.y is not None

    def is_unsupervised(self) -> bool:
        r"""Returns `True` if the dataset is unsupervised."""
        return self.X is None and self.y is not None

    def __add__(self, other: "Dataset") -> "Dataset":
        r"""Combine two datasets. Right hand dataset is stacked beneath the left."""
        X = None
        y = None

        if self.X is not None and other.X is not None:
            X = jnp.concatenate((self.X, other.X))

        if self.y is not None and other.y is not None:
            y = jnp.concatenate((self.y, other.y))

        return Dataset(X=X, y=y)

    @property
    def n(self) -> int:
        r"""Number of observations."""
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        r"""Dimension of the inputs, $`X`$."""
        return self.X.shape[1]

    def __getitem__(self, idx: Union[int, slice]) -> "Dataset":
        """Get slice of observations."""
        return Dataset(X=self.X[idx], y=self.y[idx])

    def __len__(self) -> int:
        """Get number of observations."""
        return self.n

    def train_test_split(
        self,
        key: Key,
        test_size: float = 0.2,
        scale_inputs: bool = False,
    ) -> Tuple["Dataset", "Dataset"]:
        """Split the dataset into a training and test set.

        Args:
            dataset: The dataset to split.
            key: The random key to use for the split.
            test_size: The size of the test set. Defaults to 0.2.
            scale_inputs: Whether to scale the inputs.

        Returns:
            The training and test datasets.
        """
        return train_test_split(self, key, test_size, scale_inputs)

    def get_batch(self, key: Key, batch_size: int, replace=True) -> "Dataset":
        """Get a batch from the dataset.

        Args:
            key: The random key to use for the batch selection.
            batch_size: The batch size.
            replace: Whether to sample with replacement.

        Returns:
            The batched dataset.
        """
        return get_batch(self, key, batch_size, replace)


def train_test_split(
    dataset: "Dataset",
    key,
    test_size: float = 0.2,
    scale_inputs: bool = False,
) -> Tuple["Dataset", "Dataset"]:
    """Split the dataset into a training and test set.

    Args:
        dataset: The dataset to split.
        key: The random key to use for the split.
        test_size: The size of the test set. Defaults to 0.2.
        scale_inputs: Whether to scale the inputs.

    Returns:
        The training and test datasets.
    """

    # Subsample test dataset indices.
    test_size = jnp.round(dataset.n * test_size).astype(int)
    test_indices = random.choice(key, dataset.n, (test_size,), replace=False)

    # Split the dataset.
    test = dataset[test_indices]
    train = dataset[jnp.setdiff1d(jnp.arange(dataset.n), test_indices)]

    if scale_inputs:
        # Scale the inputs.
        X_mean = train.X.mean(0)
        X_std = train.X.std(0)
        train.X = (train.X - X_mean) / X_std
        test.X = (test.X - X_mean) / X_std

    return train, test


def get_batch(dataset: "Dataset", key: Key, batch_size: int, replace=True) -> "Dataset":
    """Get a batch from the dataset.

    Args:
        dataset: The dataset to subsample.
        key: The random key to use for the batch selection.
        batch_size: The batch size.
        replace: Whether to sample with replacement.

    Returns:
        The batched dataset.
    """
    indices = random.choice(key, len(dataset), (batch_size,), replace=True)
    return dataset[indices]


def _check_shape(
    X: Optional[Float[Array, "..."]], y: Optional[Float[Array, "..."]]
) -> None:
    r"""Checks that the shapes of $`X`$ and $`y`$ are compatible."""
    if X is not None and y is not None and X.shape[0] != y.shape[0]:
        raise ValueError(
            "Inputs, X, and outputs, y, must have the same number of rows."
            f" Got X.shape={X.shape} and y.shape={y.shape}."
        )

    if X is not None and X.ndim != 2:
        raise ValueError(
            f"Inputs, X, must be a 2-dimensional array. Got X.ndim={X.ndim}."
        )

    if y is not None and y.ndim != 2:
        raise ValueError(
            f"Outputs, y, must be a 2-dimensional array. Got y.ndim={y.ndim}."
        )


def _check_precision(
    X: Optional[Float[Array, "..."]], y: Optional[Float[Array, "..."]]
) -> None:
    r"""Checks the precision of $`X`$ and $`y`."""
    if X is not None and X.dtype != jnp.float64:
        warnings.warn(
            "X is not of type float64. "
            f"Got X.dtype={X.dtype}. This may lead to numerical instability. ",
            stacklevel=2,
        )

    if y is not None and y.dtype != jnp.float64:
        warnings.warn(
            "y is not of type float64."
            f"Got y.dtype={y.dtype}. This may lead to numerical instability.",
            stacklevel=2,
        )
