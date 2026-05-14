"""Q-LINK quantum architecture construction."""

import math
import qulacs

def build_qlink_circuit(n_data: int, depth: int) -> qulacs.ParametricQuantumCircuit:
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
        circuit.add_RZ_gate(q(control_idx), math.pi/4)
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
                circuit.add_RZ_gate(q(control_idx), math.pi/4)
                circuit.add_CNOT_gate(q(i), q(control_idx))
                for target in (q(i), q(control_idx)): circuit.add_H_gate(target)
                
    return circuit
