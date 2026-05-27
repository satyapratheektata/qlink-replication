# add ../src to sys.path so scripts work without `uv run` or PYTHONPATH
import sys                                                                   # mutate import path
from pathlib import Path                                                     # OS-agnostic paths

_PKG_SRC = Path(__file__).resolve().parent.parent.parent / "src"             # repo-root/src
if _PKG_SRC.is_dir() and str(_PKG_SRC) not in sys.path:                      # idempotent guard
    sys.path.insert(0, str(_PKG_SRC))                                        # prepend so it wins over installed copies
