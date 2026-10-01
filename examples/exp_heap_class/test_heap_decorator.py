# mypy: ignore-errors
"""
Heap-allocated classes, part 2: generate the code of part 1 with a `@heap`
decorator (heap_decorator.spy), applied to the user code in point_heap.spy.

This is how we would like to write it. It does NOT run today: the parser only
accepts `@struct` as a class decorator, see README.md.
"""

from pathlib import Path

import pytest

from spy.tests.support import CompilerTest

HERE = Path(__file__).parent
CHECKS = (HERE / "heap_decorator_checks.spy").read_text()


@pytest.mark.skip(reason="class decorators not supported yet (parser)")
class TestHeapDecorator(CompilerTest):
    def compile_checks(self):
        # make `import point_heap` and `import heap_decorator` work
        self.vm.path.append(str(HERE))
        return self.compile(CHECKS, modname="heap_decorator_checks")

    def test_construct_and_method(self):
        mod = self.compile_checks()
        assert mod.construct_and_method() == 25

    def test_aliasing(self):
        mod = self.compile_checks()
        assert mod.aliasing() == 99

    def test_identity(self):
        mod = self.compile_checks()
        assert mod.identity() is True
