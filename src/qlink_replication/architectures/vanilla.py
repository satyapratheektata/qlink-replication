# Vanilla architecture: depth layers of (Rz Ry Rx) per qubit + linear CZ chain
import qulacs                                                                # ParametricQuantumCircuit + gate factories


def u_rotation_indices(n_data: int, depth: int) -> list[int]:                # every param IS a u-rotation here
    return list(range(3 * n_data * depth))                                   # depth layers * n_data qubits * 3 axes


def build_vanilla_circuit(n_data: int, depth: int) -> qulacs.ParametricQuantumCircuit:
    n_tot = n_data                                                           # Vanilla has no messenger qubit
    circuit = qulacs.ParametricQuantumCircuit(n_tot)                         # parametric so backprop works
    def q(tc_idx: int) -> int: return n_tot - 1 - tc_idx                     # map TC big-endian to qulacs LSB
    for _ in range(depth):                                                   # depth iterations of one layer
        for i in range(n_data):                                              # per-qubit Rz/Ry/Rx (parametric)
            circuit.add_parametric_RZ_gate(q(i), 0.0)                        # placeholder; set_parameter overrides
            circuit.add_parametric_RY_gate(q(i), 0.0)                        # placeholder
            circuit.add_parametric_RX_gate(q(i), 0.0)                        # placeholder
        for i in range(n_data - 1):                                          # linear CZ chain on data qubits
            circuit.add_CZ_gate(q(i), q(i + 1))                              # nearest-neighbour entangler
    return circuit                                                           # ready for set_parameter + update_quantum_state
