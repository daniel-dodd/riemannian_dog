"""Test Utils. Adapted from Pymanopt."""
from jax import config
import numpy as np
from numpy import testing as np_testing
import pytest

config.update("jax_enable_x64", True)

from manifold.geometry.utils import (
    multiqr,
    multitransp,
)


class TestMulti:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = 40
        self.n = 50
        self.p = 40
        self.k = 10

    def test_multitransp_singlemat(self):
        A = np.random.normal(size=(self.m, self.n))
        np_testing.assert_array_equal(A.T, multitransp(A))

    def test_multitransp(self):
        A = np.random.normal(size=(self.k, self.m, self.n))

        C = np.zeros((self.k, self.n, self.m))
        for i in range(self.k):
            C[i] = A[i].T

        np_testing.assert_array_equal(C, multitransp(A))

    def test_multiqr_singlemat(self):
        shape = (self.m, self.n)
        A_real = np.random.normal(size=shape)
        q, r = multiqr(A_real)
        np_testing.assert_allclose(q @ r, A_real)

        A_complex = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
        q, r = multiqr(A_complex)
        np_testing.assert_allclose(q @ r, A_complex)

    def test_multiqr(self):
        shape = (self.k, self.m, self.n)
        A_real = np.random.normal(size=shape)
        q, r = multiqr(A_real)
        np_testing.assert_allclose(q @ r, A_real)

        A_complex = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
        q, r = multiqr(A_complex)
        np_testing.assert_allclose(q @ r, A_complex)
