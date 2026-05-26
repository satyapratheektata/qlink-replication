"""Q-LINK architectural variants used in Part B (extensions).

Two knobs:

* ``basis``: interaction basis of the collection block — ``"XX"`` (default, the
  paper), ``"YY"``, or ``"ZZ"``. Each is implemented via the standard
  Hadamard/S-conjugation of the two-qubit ZZ rotation so the gate stays
  parameter-free with ``theta = pi/4``.
* ``mechanism``: ``"full"`` (default, both collection and distribution blocks),
  ``"coll_only"`` (no CNOT fan-out), ``"dist_only"`` (no Rxx collection).

The remaining structure — single-qubit R_zR_yR_x on data qubits, CZ chain across
the data register, depth-iteration — mirrors :func:`architectures.qlink.build_qlink_circuit`.

The TC<->Qulacs sign convention used by the rest of the package
(:mod:`core.optimizer`) carries through here: the fixed-collection block uses
``add_RZ_gate(target, -pi/4)`` so the resulting ZZ-conjugated rotation matches
TC's ``rxx(pi/4)`` gate-for-gate.
"""

from __future__ import annotations

import math
from typing import Literal

import qulacs


Basis = Literal["XX", "YY", "ZZ"]
Mechanism = Literal["full", "coll_only", "dist_only"]


def _apply_basis_rotation(circuit: qulacs.ParametricQuantumCircuit, target: int, basis: Basis) -> None:
    """Pre-rotate ``target`` so a subsequent ZZ-style rotation becomes XX/YY/ZZ.

    XX: Hadamard moves Z basis to X basis. YY: S followed by H gives Y basis.
    ZZ: identity.
    """
    if basis == "XX":
        circuit.add_H_gate(target)
    elif basis == "YY":
        circuit.add_S_gate(target)
        circuit.add_H_gate(target)
    elif basis == "ZZ":
        return
    else:  # pragma: no cover
        raise ValueError(f"unknown basis {basis!r}")


def _unapply_basis_rotation(circuit: qulacs.ParametricQuantumCircuit, target: int, basis: Basis) -> None:
    """Inverse of :func:`_apply_basis_rotation`."""
    if basis == "XX":
        circuit.add_H_gate(target)
    elif basis == "YY":
        circuit.add_H_gate(target)
        circuit.add_Sdag_gate(target)
    elif basis == "ZZ":
        return
    else:  # pragma: no cover
        raise ValueError(f"unknown basis {basis!r}")


def _add_collection_block(
    circuit: qulacs.ParametricQuantumCircuit,
    q,
    n_data: int,
    control_idx: int,
    basis: Basis,
) -> None:
    """One collection block: per data qubit, R_{basis,basis}(pi/4) with messenger."""
    for i in range(n_data):
        for target in (q(i), q(control_idx)):
            _apply_basis_rotation(circuit, target, basis)
        circuit.add_CNOT_gate(q(i), q(control_idx))
        circuit.add_RZ_gate(q(control_idx), -math.pi / 4)   # TC sign convention
        circuit.add_CNOT_gate(q(i), q(control_idx))
        for target in (q(i), q(control_idx)):
            _unapply_basis_rotation(circuit, target, basis)


def _add_distribution_block(
    circuit: qulacs.ParametricQuantumCircuit,
    q,
    n_data: int,
    control_idx: int,
) -> None:
    """CNOT fan-out from the messenger to each data qubit."""
    for i in range(n_data):
        circuit.add_CNOT_gate(q(control_idx), q(i))


def build_qlink_variant_circuit(
    n_data: int,
    depth: int,
    basis: Basis = "XX",
    mechanism: Mechanism = "full",
) -> qulacs.ParametricQuantumCircuit:
    """Build a Q-LINK variant for Part B extensions.

    For ``mechanism="full"`` and ``basis="XX"``, this matches
    :func:`architectures.qlink.build_qlink_circuit` (Fixed) gate-for-gate.
    """
    n_tot = n_data + 1
    circuit = qulacs.ParametricQuantumCircuit(n_tot)
    control_idx = n_data

    def q(tc_idx: int) -> int:
        return n_tot - 1 - tc_idx

    # Messenger preparation
    circuit.add_H_gate(q(control_idx))

    # Initial collection block (skipped in dist-only)
    if mechanism in {"full", "coll_only"}:
        _add_collection_block(circuit, q, n_data, control_idx, basis)

    for j in range(depth):
        # Operation block (always present)
        for i in range(n_data):
            circuit.add_parametric_RZ_gate(q(i), 0.0)
            circuit.add_parametric_RY_gate(q(i), 0.0)
            circuit.add_parametric_RX_gate(q(i), 0.0)
        for i in range(n_data - 1):
            circuit.add_CZ_gate(q(i), q(i + 1))

        # Distribution block (skipped in coll-only)
        if mechanism in {"full", "dist_only"}:
            _add_distribution_block(circuit, q, n_data, control_idx)

        # Inter-layer collection (skipped in dist-only, and after the last layer)
        if mechanism in {"full", "coll_only"} and j != depth - 1:
            _add_collection_block(circuit, q, n_data, control_idx, basis)

    return circuit


def variant_u_rotation_indices(n_data: int, depth: int) -> list[int]:
    """U-rotation parameter indices for the variant circuit.

    Because the variant always uses non-parametric collection and distribution
    blocks (Rxx fixed at pi/4 or absent), all parametric gates are U rotations.
    """
    return list(range(3 * n_data * depth))
