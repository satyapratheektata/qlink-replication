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
    The cost function in AARC-lab/DCAS_2026_QLINK src/qlink/metrics/cost_function.py
    iterates ``j in range(0, 2**n_data)`` over a ``2**n_tot``-length probability vector.
    In TensorCircuit's big-endian ordering (qubit 0 = MSB, verified empirically), this
    restricts the sum to states where TC qubit 0 = |0>, i.e. it implicitly projects the
    first data qubit. Inside the truncated string, position i indexes TC qubits 1..n_tot-1,
    which excludes qubit 0 from the measured set and includes the messenger.

    Operator form (with q_k denoting TC qubit k):
        O_buggy = I - (1 / (4 n_data)) * sum_{k=1}^{n_tot-1} (I + Z_{q_0}) (I + Z_{q_k})

    For Vanilla, n_data == n_tot, so the loop bound matches the probability vector
    length and the bug does not manifest; return the corrected observable.
    """
    if model == "Vanilla":
        return create_observable_fixed(n_tot, model)

    n_data = n_tot - 1
    obs = qulacs.Observable(n_tot)

    def q(tc_idx):
        return n_tot - 1 - tc_idx

    # Expand sum_{k=1}^{n_tot-1} (I + Z_{q_0}) (I + Z_{q_k}) = sum_k (I + Z_{q_0} + Z_{q_k} + Z_{q_0} Z_{q_k}).
    # Number of k terms is n_tot - 1 = n_data, so the constant contribution is n_data.
    coef = 1.0 / (4 * n_data)
    obs.add_operator(1.0 - coef * n_data, "")            # 3/4 I
    obs.add_operator(-coef * n_data, f"Z {q(0)}")        # -1/4 Z_{q_0}
    for k in range(1, n_tot):
        obs.add_operator(-coef, f"Z {q(k)}")
        obs.add_operator(-coef, f"Z {q(0)} Z {q(k)}")
    return obs
