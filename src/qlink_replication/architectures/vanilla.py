"""Vanilla quantum architecture construction."""

import qulacs


def u_rotation_indices(n_data: int, depth: int) -> list[int]:
    """All parameters in a Vanilla circuit are U-rotation gates, so this returns the
    full index range. Provided for API symmetry with :mod:`.qlink`."""
    return list(range(3 * n_data * depth))


def build_vanilla_circuit(n_data: int, depth: int) -> qulacs.ParametricQuantumCircuit:
    """Builds the Vanilla quantum architecture without messenger residual connections."""
    n_tot = n_data
    circuit = qulacs.ParametricQuantumCircuit(n_tot)
    
    def q(tc_idx):
        return n_tot - 1 - tc_idx
        
    for j in range(depth):
        for i in range(n_data):
            circuit.add_parametric_RZ_gate(q(i), 0.0)
            circuit.add_parametric_RY_gate(q(i), 0.0)
            circuit.add_parametric_RX_gate(q(i), 0.0)
            
        for i in range(n_data - 1):
            circuit.add_CZ_gate(q(i), q(i+1))
            
    return circuit
