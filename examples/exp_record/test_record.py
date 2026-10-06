# mypy: ignore-errors
"""
Heap-allocated classes, part 2: generate the code of part 1 with the metaclass
`Record` (stdlib/metaclasses/record.spy), applied to the user code in
point_record.spy and point_record_generated.spy.

This is how we would like to write it. It does NOT run today: the parser accepts
neither base classes nor keywords in a class statement, and the metaclass
machinery does not exist, see README.md.
"""

from pathlib import Path

import pytest

from spy.tests.support import CompilerTest

HERE = Path(__file__).parent
METACLASSES = HERE.parent.parent / "stdlib" / "metaclasses"
CHECKS = (HERE / "record_checks.spy").read_text()


@pytest.mark.skip(reason="metaclasses written in SPy not supported yet")
class TestRecord(CompilerTest):
    def compile_checks(self):
        # make `import point_record` and `import record` work (modules are flat:
        # there are no packages yet, so stdlib/metaclasses is a directory of the path)
        self.vm.path.append(str(HERE))
        self.vm.path.append(str(METACLASSES))
        return self.compile(CHECKS, modname="record_checks")

    def test_construct_and_method(self):
        mod = self.compile_checks()
        assert mod.construct_and_method() == 25

    def test_aliasing(self):
        mod = self.compile_checks()
        assert mod.aliasing() == 99

    def test_identity(self):
        mod = self.compile_checks()
        assert mod.identity() is True

    def test_generated_init(self):
        mod = self.compile_checks()
        assert mod.generated_init() == 34

    def test_generated_eq(self):
        mod = self.compile_checks()
        assert mod.generated_eq() is True

    def test_generated_repr(self):
        mod = self.compile_checks()
        assert mod.generated_repr() == "Vec(x=1, y=2)"
