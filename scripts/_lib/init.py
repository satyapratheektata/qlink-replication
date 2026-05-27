# N(0, 0.1) parameter init matching authors' u_params + res_params layout
import numpy as np                                                           # random draws


def init_params(model: str, n_data: int, depth: int, num_params: int) -> np.ndarray:
    is_adaptive = model == "Q-LINK(Adaptive)"                                # only Adaptive has trainable Rxx
    p = np.zeros(num_params)                                                 # output buffer
    cursor = 0                                                               # walking pointer
    if is_adaptive:                                                          # initial collection block
        p[cursor:cursor + n_data] = np.random.randn(n_data) * 0.1            # n_data Rxx residual params
        cursor += n_data                                                     # advance
    for j in range(depth):                                                   # one entry per layer
        p[cursor:cursor + 3 * n_data] = np.random.randn(3 * n_data) * 0.1    # 3*n_data u-rotations
        cursor += 3 * n_data                                                 # advance
        if is_adaptive and j != depth - 1:                                   # inter-layer collection
            p[cursor:cursor + n_data] = np.random.randn(n_data) * 0.1        # more Rxx residual params
            cursor += n_data                                                 # advance
    return p                                                                 # TC-convention parameter vector
