# Standalone Q-LINK Replication

This repository contains the standalone, fully modularized empirical replication of the Q-LINK architecture's trajectory variance advantage, as established in the 2026 DCAS paper.

## Architecture

This codebase is built for strict traceability, reproducibility, and rigorous CI/CD compatibility:
- **Strict Separation of Concerns**: Vanilla and Q-LINK topologies are physically decoupled in `src/qlink_replication/architectures/`. 
- **Granular Mathematics**: Observables (including the buggy Yi-Bhadani projection and our corrected marginalization) are fully isolated in `src/qlink_replication/core/observables.py`.
- **Pure Python `uv` Ecosystem**: All dependencies are instantly resolved via `uv`.

## Running the Code

1. Install `uv` if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Sync dependencies:
   ```bash
   uv sync
   ```
3. Run the CLI:
   ```bash
   uv run python src/qlink_replication/cli.py --qubits 5
   ```
   To run with the mathematically correct observable (tracing out the messenger qubit properly):
   ```bash
   uv run python src/qlink_replication/cli.py --qubits 5 --fix-cost
   ```

## Testing

Run the deterministic, isolated unit tests using `pytest`:
```bash
uv run pytest tests/unit/
```
These tests assert architectural gate counts and projective mathematical behavior, running automatically on every GitHub push via our CI workflow.
