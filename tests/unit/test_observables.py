"""Tests for the observables logic."""

from qlink_replication.core.observables import create_observable_buggy, create_observable_fixed
import qulacs
import numpy as np
import pytest


def test_fixed_observable_trace():
    """Verify the fixed observable correctly traces out the messenger."""
    n_tot = 3
    obs = create_observable_fixed(n_tot, "Q-LINK(Fixed)")

    # We evaluate it on a simple basis state |000>
    state = qulacs.QuantumState(n_tot)
    state.set_computational_basis(0)

    val = obs.get_expectation_value(state).real
    # 1/2 I - 1/4 (Z0 + Z1) evaluated on |000> is 1/2 - 1/4(1 + 1) = 0.0
    assert np.isclose(val, 0.0)


def _authors_literal_loop(probs: np.ndarray, n_tot: int, model: str) -> float:
    """Verbatim port of AARC-lab/DCAS_2026_QLINK src/qlink/metrics/cost_function.py."""
    num_qubits = n_tot
    s = 0.0
    if model != "Vanilla":
        num_qubits = num_qubits - 1
    for i in range(num_qubits):
        for j in range(0, 2 ** num_qubits):
            if format(j, f"0{num_qubits}b")[i] == "0":
                s += probs[j]
    return 1.0 - s / num_qubits


@pytest.mark.parametrize("n_tot", [3, 4, 5])
@pytest.mark.parametrize("model", ["Q-LINK(Fixed)", "Vanilla"])
def test_buggy_observable_matches_literal_loop(n_tot, model):
    """The buggy observable must reproduce the authors' Python loop bit-for-bit.

    The state is loaded directly into Qulacs: TC qubit ``i`` then maps to Qulacs qubit
    ``n_tot - 1 - i`` because TC indexes qubit 0 as MSB while Qulacs indexes qubit 0
    as LSB of the flat state-vector index. The observable uses the same mapping via
    ``q(tc_idx) = n_tot - 1 - tc_idx`` in ``create_observable_buggy``.
    """
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(2 ** n_tot) + 1j * rng.standard_normal(2 ** n_tot)
    psi /= np.linalg.norm(psi)

    state = qulacs.QuantumState(n_tot)
    state.load(psi.astype(np.complex128))

    obs = create_observable_buggy(n_tot, model)
    observable_value = obs.get_expectation_value(state).real

    probs_tc = np.abs(psi) ** 2
    loop_value = _authors_literal_loop(probs_tc, n_tot, model)

    assert np.isclose(observable_value, loop_value, atol=1e-10), (
        f"observable={observable_value:.12f} vs literal_loop={loop_value:.12f} "
        f"for n_tot={n_tot}, model={model}"
    )
