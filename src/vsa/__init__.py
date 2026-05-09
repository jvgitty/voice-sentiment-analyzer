__version__ = "0.1.1"

# Windows + Python 3.13 DLL-init workaround.
#
# On Windows + Py 3.13, importing librosa/audioread BEFORE pyarrow is
# initialised poisons the shared DLL state, and the first subsequent
# pyarrow load (pulled in transitively via NeMo / pandas) crashes the
# process with a fatal access violation (Windows status 0xC0000005,
# exit code -1073741819). No Python traceback — just an instant
# native-level segfault.
#
# tests/conftest.py already guards the pytest suite via the same
# pre-import. Putting it in the package __init__ extends that
# protection to every production entry point (the CLI, FastAPI handler,
# any third-party code that does ``from vsa import ...``).
#
# pandas is a transitive dep of NeMo so it's always present in this
# venv, but the try/except keeps light-weight installs from breaking.
try:  # noqa: SIM105 -- explicit try/except for clarity over contextlib.suppress
    import pandas as _pandas  # noqa: F401  -- imported for DLL-init side effects only.
except ImportError:
    pass
