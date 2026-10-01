# mypy: ignore-errors

# Reuse the fixtures and command-line options of spy/tests (CompilerTest needs
# --dump-c, --slow-tests, ...), so that the experiment can live outside of
# spy/tests. Run it with: pytest examples/exp_heap_class
from spy.tests.conftest import *  # noqa: F401,F403
