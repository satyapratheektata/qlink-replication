# cost-function operators: fixed (paper Eq. 2) and buggy (literal authors' loop)
import qulacs                                                                # qulacs Observable container


def create_observable_fixed(n_tot: int, model: str) -> qulacs.Observable:    # marginal-prob form of paper cost
    n_data = n_tot if model == "Vanilla" else n_tot - 1                      # data-qubit count for this model
    obs = qulacs.Observable(n_tot)                                           # allocate Hermitian operator
    def q(tc_idx: int) -> int: return n_tot - 1 - tc_idx                     # TC big-endian -> qulacs LSB mapping
    obs.add_operator(0.5, "")                                                # 1/2 * I term
    for i in range(n_data):                                                  # walk data qubits in TC indexing
        obs.add_operator(-0.5 / n_data, f"Z {q(i)}")                         # -1/(2n) * Z_i term
    return obs                                                               # caller measures expectation


def create_observable_buggy(n_tot: int, model: str) -> qulacs.Observable:    # operator form of authors' Python loop
    if model == "Vanilla":                                                   # Vanilla loop matches probs length
        return create_observable_fixed(n_tot, model)                         # so the "bug" never manifests
    n_data = n_tot - 1                                                       # Q-LINK case: n_data = n_tot - 1
    obs = qulacs.Observable(n_tot)                                           # fresh operator
    def q(tc_idx: int) -> int: return n_tot - 1 - tc_idx                     # same TC <-> qulacs mapping
    coef = 1.0 / (4 * n_data)                                                # prefactor for the projector expansion
    obs.add_operator(1.0 - coef * n_data, "")                                # constant = 3/4 I
    obs.add_operator(-coef * n_data, f"Z {q(0)}")                            # -1/4 Z_{q_0} (projector on first data qubit)
    for k in range(1, n_tot):                                                # sum over measured qubits k=1..n_tot-1
        obs.add_operator(-coef, f"Z {q(k)}")                                 # Z_{q_k} term from (I + Z_{q_k})
        obs.add_operator(-coef, f"Z {q(0)} Z {q(k)}")                        # cross term Z_{q_0} Z_{q_k}
    return obs                                                               # see test_buggy_observable_matches_literal_loop
