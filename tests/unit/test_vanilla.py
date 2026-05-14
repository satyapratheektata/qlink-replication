"""Tests for the Vanilla architecture."""

from qlink_replication.architectures.vanilla import build_vanilla_circuit

def test_vanilla_gate_count():
    """Verify that Vanilla has exactly the expected number of parametric gates."""
    n_data = 3
    depth = 2
    # For n_data = 3, depth = 2:
    # Each depth layer has n_data * 3 parametric gates = 3 * 3 = 9.
    # Total parametric gates = 9 * 2 = 18.
    circuit = build_vanilla_circuit(n_data, depth)
    assert circuit.get_parameter_count() == 18
