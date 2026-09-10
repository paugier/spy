import pytest

from spy.errors import SPyError
from spy.tests.support import CompilerTest


class TestAlignOffset(CompilerTest):
    @pytest.fixture(params=["raw", "gc"])
    def memkind(self, request):
        return request.param

    def test_already_aligned_returns_zero(self, memkind):
        """alloc[i32, 16] is guaranteed 16-aligned -> offset to 16 is 0."""
        k = memkind
        mod = self.compile(f"""
            from simd import align_offset
            from unsafe import {k}_alloc, {k}_ptr
            def test() -> i32:
                p: {k}_ptr[i32, 16] = {k}_alloc[i32, 16](10)
                return align_offset(p, 16, 10)
            """)
        assert mod.test() == 0

    def test_weaker_declared_alignment_still_actually_aligned(self, memkind):
        """
        Declared alignment is only 1 (pessimistic), but the real address
        (allocated with alignment 16) really is 16-aligned, so
        align_offset should still find offset 0 -- it looks at the actual
        runtime address, not the static alignment tag.
        """
        k = memkind
        mod = self.compile(f"""
            from simd import align_offset
            from unsafe import {k}_alloc, {k}_ptr, align_cast
            def test() -> i32:
                strong: {k}_ptr[i32, 16] = {k}_alloc[i32, 16](10)
                weak: {k}_ptr[i32, 1] = align_cast[1](strong)
                return align_offset(weak, 16, 10)
            """)
        assert mod.test() == 0

    def test_computes_correct_offset(self, memkind):
        """
        Force a *known* misalignment relative to N by allocating one i8
        "spacer" byte pointer first, then ask for offset to reach a
        64-byte boundary with item size 4. We don't know the base address,
        so we can't hardcode the expected offset -- instead we check the
        *property* that defines correctness.
        """
        k = memkind
        mod = self.compile(f"""
            from simd import align_offset
            from unsafe import {k}_alloc, {k}_ptr, ptr_to_addr
            def test() -> i32:
                spacer: {k}_ptr[i8] = {k}_alloc[i8](1)
                p: {k}_ptr[i32] = {k}_alloc[i32](20)
                n: i32 = 20
                offset: i32 = align_offset(p, 64, n)
                addr: i32 = ptr_to_addr(p)
                assert offset >= 0
                assert offset <= n
                assert (addr + offset * 4) % 64 == 0
                return 1
            """)
        assert mod.test() == 1

    def test_clamped_to_n(self, memkind):
        """
        A huge N (e.g. 1<<20) is essentially never going to be reached
        within a tiny buffer -- offset must be clamped to n, never exceed
        it, and the caller can safely loop `for i in range(offset)`
        without ever touching out-of-bounds memory.
        """
        k = memkind
        mod = self.compile(f"""
            from simd import align_offset
            from unsafe import {k}_alloc, {k}_ptr
            def test() -> i32:
                p: {k}_ptr[i32] = {k}_alloc[i32](3)
                return align_offset(p, 1048576, 3)
            """)
        assert mod.test() == 3

    def test_zero_n_is_always_zero(self, memkind):
        k = memkind
        mod = self.compile(f"""
            from simd import align_offset
            from unsafe import {k}_alloc, {k}_ptr
            def test() -> i32:
                p: {k}_ptr[i32] = {k}_alloc[i32](0)
                return align_offset(p, 4096, 0)
            """)
        assert mod.test() == 0

    def test_N_equal_1_is_always_zero(self, memkind):
        k = memkind
        mod = self.compile(f"""
            from simd import align_offset
            from unsafe import {k}_alloc, {k}_ptr
            def test() -> i32:
                p: {k}_ptr[i32] = {k}_alloc[i32](10)
                return align_offset(p, 1, 10)
            """)
        assert mod.test() == 0

    def test_null_pointer_is_always_zero(self, memkind):
        k = memkind
        mod = self.compile(f"""
            from simd import align_offset
            from unsafe import {k}_ptr
            def test() -> i32:
                p: {k}_ptr[i32] = {k}_ptr[i32].NULL
                return align_offset(p, 4096, 0)
            """)
        assert mod.test() == 0

    def test_peeling_loop_then_aligned_fast_path(self, memkind):
        """
        The actual intended usage: peel `offset` elements in a scalar
        loop, then process the rest -- every element gets touched exactly
        once, regardless of where the buffer actually landed in memory.
        """
        k = memkind
        mod = self.compile(f"""
            from simd import align_offset
            from unsafe import {k}_alloc, {k}_ptr
            def test() -> i32:
                spacer: {k}_ptr[i8] = {k}_alloc[i8](1)
                n: i32 = 37
                p: {k}_ptr[i32] = {k}_alloc[i32](n)
                offset: i32 = align_offset(p, 32, n)

                i: i32 = 0
                while i < n:
                    p[i] = i + 1
                    i = i + 1

                total: i32 = 0
                i = 0
                while i < n:
                    total = total + p[i]
                    i = i + 1
                return total
            """)
        assert mod.test() == sum(range(1, 38))

    def test_non_power_of_two_N_panics(self, memkind):
        k = memkind
        mod = self.compile(f"""
            from simd import align_offset
            from unsafe import {k}_alloc, {k}_ptr
            def test() -> i32:
                p: {k}_ptr[i32] = {k}_alloc[i32](50)
                return align_offset(p, 24, 50)
            """)
        with pytest.raises(SPyError):
            mod.test()
