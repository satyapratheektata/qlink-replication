"""Tests for the expressibility metric (KL vs Haar)."""

import numpy as np
import pytest

from qlink_replication.architectures.vanilla import build_vanilla_circuit
from qlink_replication.core.expressibility import (
    compute_expressibility,
    haar_fidelity_pmf,
)


def test_haar_pmf_sums_to_one():
    """The Haar PMF integrated over [0, 1] must sum to 1 to machine precision."""
    for dim in [2, 4, 8, 16, 32]:
        pmf = haar_fidelity_pmf(num_bins=20, dim=dim)
        assert pmf.shape == (20,)
        assert np.all(pmf >= 0)
        assert np.isclose(pmf.sum(), 1.0)


def test_haar_pmf_matches_cdf_formula():
    """Each bin must equal the difference of the CDF ``1 - (1-F)^(d-1)`` at its edges."""
    dim = 8
    num_bins = 20
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    expected = (1 - (1 - edges[1:]) ** (dim - 1)) - (1 - (1 - edges[:-1]) ** (dim - 1))
    actual = haar_fidelity_pmf(num_bins, dim)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("n_data", [2, 3])
def test_expressibility_smoke(n_data):
    """Smoke test: a small parameterized circuit should produce a finite, non-negative KL."""
    rng = np.random.default_rng(0)
    circuit = build_vanilla_circuit(n_data, depth=2)
    kl = compute_expressibility(circuit, num_fidelity=50, num_bins=20, rng=rng)
    assert np.isfinite(kl)
    assert kl >= 0.0
