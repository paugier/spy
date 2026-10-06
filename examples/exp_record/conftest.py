# mypy: ignore-errors

# Reuse the fixtures and command-line options of spy/tests (CompilerTest needs
# --dump-c, --slow-tests, ...), so that the experiment can live outside of
# spy/tests. Run it with: pytest examples/exp_record
from spy.tests.conftest import *  # noqa: F401,F403


def pytest_configure(config):
    config.addinivalue_line("markers", "record: Point written with metaclass=Record")
    config.addinivalue_line("markers", "by-hand: the code that Record generates")
    config.addinivalue_line("markers", "simple: simplified Point that runs")
