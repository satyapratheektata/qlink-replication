"""Tests for the Q-LINK architecture."""

from qlink_replication.architectures.qlink import build_qlink_circuit

def test_qlink_qubit_count():
    """Verify that Q-LINK has n_data + 1 total qubits."""
    n_data = 3
    depth = 2
    circuit = build_qlink_circuit(n_data, depth)
    assert circuit.get_qubit_count() == n_data + 1

def test_qlink_parameter_count():
    """Verify that Q-LINK has the identical number of parameters as Vanilla."""
    n_data = 3
    depth = 2
    circuit = build_qlink_circuit(n_data, depth)
    assert circuit.get_parameter_count() == 18
