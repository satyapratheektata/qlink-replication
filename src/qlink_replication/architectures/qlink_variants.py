# Q-LINK variants for Part B: choose collection basis and ablate mechanism halves
import math                                                                  # pi for the fixed Rxx angle
import qulacs                                                                # parametric circuit
from ._basis import apply_basis_rotation, unapply_basis_rotation             # XX/YY/ZZ conjugation helpers


def variant_u_rotation_indices(n_data: int, depth: int) -> list[int]:        # variants only have u rotations
    return list(range(3 * n_data * depth))                                   # no parametric Rxx (fixed pi/4 or absent)


def _coll_block(circuit, q, n_data, ctrl, basis):                            # one collection block in given basis
    for i in range(n_data):                                                  # per data qubit
        for t in (q(i), q(ctrl)): apply_basis_rotation(circuit, t, basis)    # rotate into ZZ basis
        circuit.add_CNOT_gate(q(i), q(ctrl))                                 # ZZ rotation via CNOT-Rz-CNOT
        circuit.add_RZ_gate(q(ctrl), -math.pi / 4)                           # fixed theta = -pi/4 (TC sign)
        circuit.add_CNOT_gate(q(i), q(ctrl))                                 # close the ZZ rotation
        for t in (q(i), q(ctrl)): unapply_basis_rotation(circuit, t, basis)  # rotate out of ZZ basis


def _dist_block(circuit, q, n_data, ctrl):                                   # one distribution block
    for i in range(n_data): circuit.add_CNOT_gate(q(ctrl), q(i))             # CNOT fan-out from messenger


def build_qlink_variant_circuit(                                             # E2+E3 driver
    n_data: int, depth: int, basis: str = "XX", mechanism: str = "full",     # defaults reproduce build_qlink_circuit(Fixed)
) -> qulacs.ParametricQuantumCircuit:
    n_tot = n_data + 1                                                       # messenger qubit
    circuit = qulacs.ParametricQuantumCircuit(n_tot)                         # parametric circuit
    ctrl = n_data                                                            # TC messenger index
    def q(tc_idx: int) -> int: return n_tot - 1 - tc_idx                     # TC <-> qulacs mapping
    circuit.add_H_gate(q(ctrl))                                              # prepare messenger in |+>
    if mechanism in {"full", "coll_only"}: _coll_block(circuit, q, n_data, ctrl, basis)  # initial collection
    for j in range(depth):                                                   # main depth loop
        for i in range(n_data):                                              # operation: per-qubit Rz Ry Rx
            circuit.add_parametric_RZ_gate(q(i), 0.0)                        # placeholder
            circuit.add_parametric_RY_gate(q(i), 0.0)                        # placeholder
            circuit.add_parametric_RX_gate(q(i), 0.0)                        # placeholder
        for i in range(n_data - 1): circuit.add_CZ_gate(q(i), q(i + 1))      # CZ chain
        if mechanism in {"full", "dist_only"}: _dist_block(circuit, q, n_data, ctrl)  # distribution
        if mechanism in {"full", "coll_only"} and j != depth - 1:            # inter-layer collection
            _coll_block(circuit, q, n_data, ctrl, basis)                     # one per remaining layer boundary
    return circuit                                                           # ready
