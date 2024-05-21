"""Test the AbstractManifold class."""
import pytest

from manifold.geometry.base import AbstractManifold


class TestAbstractManifold:
    def test_initialise_abstract_manifold(self) -> None:
        # Creating a manifold should raise an error. Due to abstract methods!
        with pytest.raises(TypeError):
            AbstractManifold()
