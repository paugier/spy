# mypy: ignore-errors
"""
Heap-allocated classes, part 1: the code that `Record` is supposed to generate,
written by hand in point_by_hand.spy.

point_by_hand_checks.spy imports `Point` from point_by_hand and defines the
functions called by the tests below.
"""

from pathlib import Path

from spy.tests.support import CompilerTest

HERE = Path(__file__).parent
CHECKS = (HERE / "point_by_hand_checks.spy").read_text()


class TestRecordByHand(CompilerTest):
    def compile_checks(self):
        # make `import point_by_hand` work
        self.vm.path.append(str(HERE))
        return self.compile(CHECKS, modname="point_by_hand_checks")

    def test_construct_and_method(self):
        mod = self.compile_checks()
        assert mod.construct_and_method() == 25

    def test_getattr_setattr(self):
        mod = self.compile_checks()
        assert mod.getattr_setattr() == 1004

    def test_aliasing(self):
        mod = self.compile_checks()
        assert mod.aliasing() == 99

    def test_mutation_through_call(self):
        mod = self.compile_checks()
        assert mod.mutation_through_call() == 3

    def test_independent_instances(self):
        mod = self.compile_checks()
        assert mod.independent_instances() == 1

    def test_identity_via_ll(self):
        mod = self.compile_checks()
        assert mod.identity_same() is True
        assert mod.identity_different() is False
