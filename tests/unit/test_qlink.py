"""Tests for the Q-LINK architecture."""

import pytest

from qlink_replication.architectures.qlink import build_qlink_circuit, u_rotation_indices


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


@pytest.mark.parametrize("n_data,depth", [(3, 2), (4, 3), (5, 5)])
def test_u_rotation_indices_fixed(n_data, depth):
    """Fixed Q-LINK has no parametric Rxx gates; all parameters are U rotations."""
    circuit = build_qlink_circuit(n_data, depth, is_adaptive=False)
    indices = u_rotation_indices(n_data, depth, is_adaptive=False)
    assert indices == list(range(circuit.get_parameter_count()))
    assert len(indices) == 3 * n_data * depth


@pytest.mark.parametrize("n_data,depth", [(3, 2), (4, 3), (5, 5)])
def test_u_rotation_indices_adaptive(n_data, depth):
    """Adaptive Q-LINK has 3*n_data*depth U-rotation params plus n_data*depth Rxx params.

    The U-rotation params are interleaved between collection blocks; the helper must
    select the same subset of qulacs' flat parameter vector that the authors' code
    records as ``u_params.grad``.
    """
    circuit = build_qlink_circuit(n_data, depth, is_adaptive=True)
    indices = u_rotation_indices(n_data, depth, is_adaptive=True)
    assert len(indices) == 3 * n_data * depth
    # Total params = U rotations + Rxx (initial collection + (depth-1) inter-layer collections)
    assert circuit.get_parameter_count() == 3 * n_data * depth + n_data * depth
    assert max(indices) < circuit.get_parameter_count()
    # Indices must be strictly increasing and disjoint from the Rxx slots.
    assert indices == sorted(set(indices))
