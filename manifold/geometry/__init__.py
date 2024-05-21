"""Manfolds submodule. Get your geometry here!"""
from manifold.geometry.base import AbstractManifold
from manifold.geometry.euclidean import Euclidean
from manifold.geometry.grassmann import Grassmann
from manifold.geometry.poincare import Poincare
from manifold.geometry.product_same import ProductSame
from manifold.geometry.sphere import Sphere
from manifold.geometry.zeta import zeta

__all__ = [
    "AbstractManifold",
    "Grassmann",
    "Sphere",
    "Hyperbolic",
    "ProductSame",
    "Poincare",
    "Euclidean",
    "zeta",
]
