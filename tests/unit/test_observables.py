"""Tests for the observables logic."""

from qlink_replication.core.observables import create_observable_buggy, create_observable_fixed
import qulacs
import numpy as np

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
