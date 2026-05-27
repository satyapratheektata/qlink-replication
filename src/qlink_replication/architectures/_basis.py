# basis-conjugation helpers for E2 (XX vs YY vs ZZ collection)
import qulacs                                                                # H and S gates


def apply_basis_rotation(circuit: qulacs.ParametricQuantumCircuit, target: int, basis: str) -> None:
    if basis == "XX": circuit.add_H_gate(target)                             # H takes Z -> X basis
    elif basis == "YY":                                                      # S then H takes Z -> Y basis
        circuit.add_S_gate(target); circuit.add_H_gate(target)               # S diagonalises X up to phase
    elif basis == "ZZ": return                                               # no rotation needed
    else: raise ValueError(f"unknown basis {basis!r}")                       # fail loud on typos


def unapply_basis_rotation(circuit: qulacs.ParametricQuantumCircuit, target: int, basis: str) -> None:
    if basis == "XX": circuit.add_H_gate(target)                             # H is self-inverse
    elif basis == "YY":                                                      # inverse: H then S^dagger
        circuit.add_H_gate(target); circuit.add_Sdag_gate(target)            # mirror of apply order
    elif basis == "ZZ": return                                               # nothing to undo
    else: raise ValueError(f"unknown basis {basis!r}")                       # ditto
