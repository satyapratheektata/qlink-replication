# SGD + seeding defaults matching the paper's Section 4 protocol
DEFAULT_LEARNING_RATE = 0.1                                                  # paper: SGD lr 0.1
DEFAULT_MAX_ITERS = 1500                                                     # paper: max 1500 iters
DEFAULT_CONVERGENCE_THRESHOLD = 1e-3                                         # paper: stop at L < 1e-3
DEFAULT_NUM_SEEDS = 5                                                        # paper: 5 random seeds per cell
DEFAULT_RANDOM_SEED = 42                                                     # authors' global seed in main.py
