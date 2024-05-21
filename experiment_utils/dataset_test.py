"""Test dataset. From GPJax!"""
from dataclasses import is_dataclass

from jax import (
    config,
    random,
)
import jax.numpy as jnp
import pytest

from experiment_utils.dataset import Dataset

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_dataset_init(n: int, in_dim: int) -> None:
    # Create dataset
    x = jnp.ones((n, in_dim))
    y = jnp.ones((n, 1))

    D = Dataset(X=x, y=y)

    # Test dataset shapes
    assert D.n == n
    assert D.in_dim == in_dim

    # Test representation
    assert D.__repr__() == f"- Number of observations: {n}\n- Input dimension: {in_dim}"

    # Ensure dataclass
    assert is_dataclass(D)

    # Test supervised and unsupervised
    assert Dataset(X=x, y=y).is_supervised() is True
    assert Dataset(y=y).is_unsupervised() is True


@pytest.mark.parametrize("n1", [1, 2, 10])
@pytest.mark.parametrize("n2", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_dataset_add(n1: int, n2: int, in_dim: int) -> None:
    # Create first dataset
    x1 = jnp.ones((n1, in_dim))
    y1 = jnp.ones((n1, 1))
    D1 = Dataset(X=x1, y=y1)

    # Create second dataset
    x2 = 2 * jnp.ones((n2, in_dim))
    y2 = 2 * jnp.ones((n2, 1))
    D2 = Dataset(X=x2, y=y2)

    # Add datasets
    D = D1 + D2

    # Test shapes
    assert D.n == n1 + n2
    assert D.in_dim == in_dim

    # Test representation
    assert (
        D.__repr__()
        == f"- Number of observations: {n1 + n2}\n- Input dimension: {in_dim}"
    )

    # Ensure dataclass
    assert is_dataclass(D)

    # Test supervised and unsupervised
    assert (Dataset(X=x1, y=y1) + Dataset(X=x2, y=y2)).is_supervised() is True
    assert (Dataset(y=y1) + Dataset(y=y2)).is_unsupervised() is True


@pytest.mark.parametrize(("nx", "ny"), [(1, 2), (2, 1), (10, 5), (5, 10)])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_dataset_incorrect_lengths(nx: int, ny: int, in_dim: int) -> None:
    # Create input and output pairs of different lengths
    x = jnp.ones((nx, in_dim))
    y = jnp.ones((ny, 1))

    # Ensure error is raised upon dataset creation
    with pytest.raises(ValueError):
        Dataset(X=x, y=y)


@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_2d_inputs(n: int, in_dim: int) -> None:
    # Create dataset where output dimension is incorrectly not 2D
    x = jnp.ones((n, in_dim))
    y = jnp.ones((n,))

    # Ensure error is raised upon dataset creation
    with pytest.raises(ValueError):
        Dataset(X=x, y=y)

    # Create dataset where input dimension is incorrectly not 2D
    x = jnp.ones((n,))
    y = jnp.ones((n, 1))

    # Ensure error is raised upon dataset creation
    with pytest.raises(ValueError):
        Dataset(X=x, y=y)


@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_y_none(n: int, in_dim: int) -> None:
    # Create a dataset with no output
    x = jnp.ones((n, in_dim))
    D = Dataset(X=x)

    # Ensure is dataclass
    assert is_dataclass(D)

    # Ensure output is None
    assert D.y is None


@pytest.mark.parametrize(
    ("prec_x", "prec_y"),
    [
        (jnp.float32, jnp.float64),
        (jnp.float64, jnp.float32),
        (jnp.float32, jnp.float32),
    ],
)
@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_precision_warning(
    n: int, in_dim: int, prec_x: jnp.dtype, prec_y: jnp.dtype
) -> None:
    # Create dataset
    x = jnp.ones((n, in_dim)).astype(prec_x)
    y = jnp.ones((n, 1)).astype(prec_y)

    # Check for warnings if dtypes are not float64
    expected_warnings = 0
    if prec_x != jnp.float64:
        expected_warnings += 1
    if prec_y != jnp.float64:
        expected_warnings += 1

    with pytest.warns(UserWarning, match=".* is not of type float64.*") as record:
        Dataset(X=x, y=y)

    assert len(record) == expected_warnings


@pytest.mark.parametrize("in_dim", [1, 2, 5])
@pytest.mark.parametrize("n, batch_size", [(1, 1), (2, 1), (10, 2)])
def test_get_batch(n: int, in_dim: int, batch_size: int) -> None:
    # Create dataset
    x = jnp.ones((n, in_dim))
    y = jnp.ones((n, 1))
    D = Dataset(X=x, y=y)

    key = random.PRNGKey(42)

    # Get batch
    batch = D.get_batch(key, batch_size=batch_size)

    # Check batch shape
    assert batch.X.shape == (batch_size, in_dim)
    assert batch.y.shape == (batch_size, 1)

    # Check batch is a dataclass
    assert is_dataclass(batch)

    # Check batch is a Dataset
    assert isinstance(batch, Dataset)

    # Check batch is a subset of the original dataset
    assert (batch.X == x[:batch_size]).all()
    assert (batch.y == y[:batch_size]).all()

    # Check batch is a subset of the original dataset
    assert (batch.X == x[:batch_size]).all()
    assert (batch.y == y[:batch_size]).all()

    # Check batch is a subset of the original dataset
    assert (batch.X == x[:batch_size]).all()
    assert (batch.y == y[:batch_size]).all()

    # Check batch is a subset of the original dataset
    assert (batch.X == x[:batch_size]).all()
    assert (batch.y == y[:batch_size]).all()


def test_train_test_split() -> None:
    # Create dataset
    x = jnp.ones((10, 2))
    y = jnp.ones((10, 1))
    D = Dataset(X=x, y=y)

    key = random.PRNGKey(42)

    # Split dataset
    train, test = D.train_test_split(key, test_size=0.2)

    # Check train and test shapes
    assert train.X.shape == (8, 2)
    assert train.y.shape == (8, 1)
    assert test.X.shape == (2, 2)
    assert test.y.shape == (2, 1)

    # Check train and test are dataclasses
    assert is_dataclass(train)
    assert is_dataclass(test)

    # Check train and test are Datasets
    assert isinstance(train, Dataset)
    assert isinstance(test, Dataset)

    # Check train and test are subsets of the original dataset
    assert (train.X == x[:8]).all()
    assert (train.y == y[:8]).all()
    assert (test.X == x[8:]).all()
    assert (test.y == y[8:]).all()

    # Check train and test are subsets of the original dataset
    assert (train.X == x[:8]).all()
    assert (train.y == y[:8]).all()
    assert (test.X == x[8:]).all()
    assert (test.y == y[8:]).all()

    # Check train and test are subsets of the original dataset
    assert (train.X == x[:8]).all()
    assert (train.y == y[:8]).all()
    assert (test.X == x[8:]).all()
    assert (test.y == y[8:]).all()

    # Check train and test are subsets of the original dataset
    assert (train.X == x[:8]).all()
    assert (train.y == y[:8]).all()
    assert (test.X == x[8:]).all()
    assert (test.y == y[8:]).all()

    # Check train and test are subsets of the original dataset
    assert (train.X == x[:8]).all()
    assert (train.y == y[:8]).all()
    assert (test.X == x[8:]).all()
    assert (test.y == y[8:]).all()
