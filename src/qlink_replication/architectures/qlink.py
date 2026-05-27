# Q-LINK architecture: messenger qubit + collection (Rxx) + operation (Rz Ry Rx + CZ) + distribution (CNOT)
import math                                                                  # pi for the fixed Rxx angle
import qulacs                                                                # parametric circuit primitives


def u_rotation_indices(n_data: int, depth: int, is_adaptive: bool) -> list[int]:  # u-only param mask
    indices: list[int] = []                                                  # collected u-rotation positions
    cursor = 0                                                               # walking pointer in insertion order
    if is_adaptive: cursor += n_data                                         # Adaptive: initial Rxx block consumes n_data params
    for j in range(depth):                                                   # one entry per layer
        indices.extend(range(cursor, cursor + 3 * n_data))                   # 3*n_data u-rotations (Rz, Ry, Rx)
        cursor += 3 * n_data                                                 # advance past them
        if is_adaptive and j != depth - 1: cursor += n_data                  # Adaptive: inter-layer Rxx between layers
    return indices                                                           # caller slices grad_tc[indices]


def build_qlink_circuit(n_data: int, depth: int, is_adaptive: bool = False) -> qulacs.ParametricQuantumCircuit:
    n_tot = n_data + 1                                                       # extra messenger qubit
    circuit = qulacs.ParametricQuantumCircuit(n_tot)                         # parametric so backprop works
    control_idx = n_data                                                     # messenger is TC qubit n_data
    def q(tc_idx: int) -> int: return n_tot - 1 - tc_idx                     # TC <-> qulacs mapping
    circuit.add_H_gate(q(control_idx))                                       # prepare messenger in |+>
    for i in range(n_data):                                                  # initial collection block
        for t in (q(i), q(control_idx)): circuit.add_H_gate(t)               # H pre-rotation for ZZ -> XX
        circuit.add_CNOT_gate(q(i), q(control_idx))                          # CNOT to entangle for ZZ rotation
        if is_adaptive: circuit.add_parametric_RZ_gate(q(control_idx), 0.0)  # Adaptive: trainable theta (set via params)
        else: circuit.add_RZ_gate(q(control_idx), -math.pi / 4)              # Fixed: -pi/4 (TC sign convention)
        circuit.add_CNOT_gate(q(i), q(control_idx))                          # CNOT to undo entanglement
        for t in (q(i), q(control_idx)): circuit.add_H_gate(t)               # H undo => Rxx(theta) net effect
    for j in range(depth):                                                   # main depth loop
        for i in range(n_data):                                              # operation block: per-qubit Rz Ry Rx
            circuit.add_parametric_RZ_gate(q(i), 0.0)                        # parametric Rz placeholder
            circuit.add_parametric_RY_gate(q(i), 0.0)                        # parametric Ry placeholder
            circuit.add_parametric_RX_gate(q(i), 0.0)                        # parametric Rx placeholder
        for i in range(n_data - 1): circuit.add_CZ_gate(q(i), q(i + 1))      # data-data CZ chain
        for i in range(n_data): circuit.add_CNOT_gate(q(control_idx), q(i))  # distribution block: messenger -> each data
        if j != depth - 1:                                                   # collection block between layers (skip after last)
            for i in range(n_data):                                          # one Rxx per data qubit
                for t in (q(i), q(control_idx)): circuit.add_H_gate(t)       # H pre-rotation
                circuit.add_CNOT_gate(q(i), q(control_idx))                  # CNOT
                if is_adaptive: circuit.add_parametric_RZ_gate(q(control_idx), 0.0)  # Adaptive: trainable
                else: circuit.add_RZ_gate(q(control_idx), -math.pi / 4)      # Fixed: -pi/4
                circuit.add_CNOT_gate(q(i), q(control_idx))                  # CNOT
                for t in (q(i), q(control_idx)): circuit.add_H_gate(t)       # H undo
    return circuit                                                           # ready to consume parameters
