"""Q-LINK quantum architecture construction."""

import math
import qulacs


def u_rotation_indices(n_data: int, depth: int, is_adaptive: bool) -> list[int]:
    """Return the parameter indices that correspond to U-rotation gates (Rz/Ry/Rx on
    data qubits), in the order they are added by :func:`build_qlink_circuit`.

    The authors' ``scripts/main.py`` only records ``u_params.grad`` for gradient-variance
    computation; this helper exposes the same subset of qulacs' flat parameter vector.

    For Fixed, this is simply [0, ..., 3*n_data*depth - 1] since no Rxx params exist.
    For Adaptive, the initial collection block contributes ``n_data`` Rxx params
    before the first U-rotation layer, and each subsequent collection block contributes
    another ``n_data`` between layers.
    """
    indices: list[int] = []
    cursor = 0
    if is_adaptive:
        cursor += n_data  # initial collection block (Rxx residuals)
    for j in range(depth):
        indices.extend(range(cursor, cursor + 3 * n_data))
        cursor += 3 * n_data
        if is_adaptive and j != depth - 1:
            cursor += n_data  # inter-layer collection block (Rxx residuals)
    return indices


def build_qlink_circuit(n_data: int, depth: int, is_adaptive: bool = False) -> qulacs.ParametricQuantumCircuit:
    """Builds the Q-LINK quantum architecture with messenger residual connections."""
    n_tot = n_data + 1
    circuit = qulacs.ParametricQuantumCircuit(n_tot)
    control_idx = n_data
    
    def q(tc_idx):
        return n_tot - 1 - tc_idx
        
    circuit.add_H_gate(q(control_idx))
    for i in range(n_data):
        for target in (q(i), q(control_idx)): circuit.add_H_gate(target)
        circuit.add_CNOT_gate(q(i), q(control_idx))
        if is_adaptive:
            circuit.add_parametric_RZ_gate(q(control_idx), 0.0)
        else:
            # Qulacs RZ uses exp(+i*angle/2*Z) while TC's rz/rxx uses exp(-i*theta/2);
            # negate the fixed angle so the resulting Rxx matches TC's Rxx(pi/4).
            circuit.add_RZ_gate(q(control_idx), -math.pi/4)
        circuit.add_CNOT_gate(q(i), q(control_idx))
        for target in (q(i), q(control_idx)): circuit.add_H_gate(target)
            
    for j in range(depth):
        for i in range(n_data):
            circuit.add_parametric_RZ_gate(q(i), 0.0)
            circuit.add_parametric_RY_gate(q(i), 0.0)
            circuit.add_parametric_RX_gate(q(i), 0.0)
            
        for i in range(n_data - 1):
            circuit.add_CZ_gate(q(i), q(i+1))
            
        for i in range(n_data):
            circuit.add_CNOT_gate(q(control_idx), q(i))
            
        if j != depth - 1:
            for i in range(n_data):
                for target in (q(i), q(control_idx)): circuit.add_H_gate(target)
                circuit.add_CNOT_gate(q(i), q(control_idx))
                if is_adaptive:
                    circuit.add_parametric_RZ_gate(q(control_idx), 0.0)
                else:
                    # Qulacs RZ uses exp(+i*angle/2*Z) while TC's rz/rxx uses exp(-i*theta/2);
                    # negate the fixed angle so the resulting Rxx matches TC's Rxx(pi/4).
                    circuit.add_RZ_gate(q(control_idx), -math.pi/4)
                circuit.add_CNOT_gate(q(i), q(control_idx))
                for target in (q(i), q(control_idx)): circuit.add_H_gate(target)
                
    return circuit
