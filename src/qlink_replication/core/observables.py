"""Cost function definitions, including buggy and fixed variations."""

import qulacs

def create_observable_fixed(n_tot: int, model: str) -> qulacs.Observable:
    """
    FIXED COST FUNCTION: 
    Marginal probability over data qubits being 0, tracing out the control qubit.
    Cost = 1/2 I - 1/(2n) sum Z_i.
    """
    n_data = n_tot if model == "Vanilla" else n_tot - 1
    obs = qulacs.Observable(n_tot)
    
    def q(tc_idx):
        return n_tot - 1 - tc_idx
        
    obs.add_operator(0.5, "")
    for i in range(n_data):
        obs.add_operator(-0.5 / n_data, f"Z {q(i)}")
    return obs

def create_observable_buggy(n_tot: int, model: str) -> qulacs.Observable:
    """
    BUGGY COST FUNCTION (Yi-Bhadani):
    Accidentally computes joint probability P(data=0 AND control=0).
    """
    n_data = n_tot if model == "Vanilla" else n_tot - 1
    obs = qulacs.Observable(n_tot)
    
    def q(tc_idx):
        return n_tot - 1 - tc_idx
        
    obs.add_operator(0.5, "")
    for i in range(n_data):
        obs.add_operator(-0.5 / n_data, f"Z {q(i)}")
        
    if model != "Vanilla":
        control_idx = n_data
        obs.add_operator(0.25, "")
        obs.add_operator(-0.25, f"Z {q(control_idx)}")
        for i in range(n_data):
            obs.add_operator(0.25 / n_data, f"Z {q(i)}")
            obs.add_operator(-0.25 / n_data, f"Z {q(control_idx)} Z {q(i)}")
            
    return obs
