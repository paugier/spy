from spy.errors import SPyError
from spy.tests.support import CompilerTest, expect_errors, only_interp


class TestSIMD(CompilerTest):
    def test_sizes(self):
        src = """
        from _simd import SIMD

        def get_last[N](x: i32) -> i32:
            v = SIMD[i32, N].splat(x)
            return v[N - 1]

        s1 = get_last[1]
        s2 = get_last[2]
        s4 = get_last[4]
        s8 = get_last[8]
        """
        mod = self.compile(src)
        assert mod.s1(1) == 1
        assert mod.s2(2) == 2
        assert mod.s4(3) == 3
        assert mod.s8(7) == 7

    @only_interp
    def test_invalid_params(self):
        src = """
        from _simd import SIMD

        @struct
        class Point:
            x: i32
            y: i32

        def length_not_power_of_two() -> None:
            v = SIMD[f32, 3]

        def length_zero() -> None:
            v = SIMD[f32, 0]

        def length_negative() -> None:
            v = SIMD[f32, -2]

        def dtype_bool() -> None:
            v = SIMD[bool, 4]

        def dtype_str() -> None:
            v = SIMD[str, 4]

        def dtype_struct() -> None:
            v = SIMD[Point, 4]
        """

        def check_error(funcname: str, msg: str) -> None:
            self.compile_raises(src, funcname, expect_errors(msg))

        check_error(
            "length_not_power_of_two", "SIMD length must be a power of two, got 3"
        )
        check_error("length_zero", "SIMD length must be a positive power of two, got 0")
        check_error(
            "length_negative", "SIMD length must be a positive power of two, got -2"
        )
        check_error(
            "dtype_bool", "SIMD element type must be a numeric primitive, got `bool`"
        )
        check_error(
            "dtype_str", "SIMD element type must be a numeric primitive, got `str`"
        )
        check_error(
            "dtype_struct",
            "SIMD element type must be a numeric primitive, got `test::Point`",
        )

    def test_constructors(self):
        src = """
        from _simd import SIMD

        VEC = SIMD[i32, 4]

        def lanes(v: VEC) -> tuple[i32, i32, i32, i32]:
            return v[0], v[1], v[2], v[3]

        def zeros() -> tuple[i32, i32, i32, i32]:
            return lanes(VEC.zeros())

        def splat(x: i32) -> tuple[i32, i32, i32, i32]:
            return lanes(VEC.splat(x))

        def explicit(a: i32, b: i32, c: i32, d: i32) -> tuple[i32, i32, i32, i32]:
            return lanes(VEC(a, b, c, d))

        def convert(a: i32, b: i32, c: i32, d: i32) -> tuple[i32, i32, i32, i32]:
            v: VEC = [a, b, c, d]
            return lanes(v)

        def splat_is_explicit(x: i32) -> bool:
            v = VEC.splat(x)
            w = VEC(x, x, x, x)
            return lanes(v) == lanes(w)
        """
        mod = self.compile(src)
        assert mod.zeros() == (0, 0, 0, 0)
        assert mod.splat(42) == (42, 42, 42, 42)
        assert mod.explicit(1, 2, 3, 4) == (1, 2, 3, 4)
        assert mod.convert(5, 6, 7, 8) == (5, 6, 7, 8)
        assert mod.splat_is_explicit(9)

    def test_constructors_all_dtypes(self):
        src = """
        from _simd import SIMD

        def func[T](x: T) -> tuple[T, T, T, T]:
            z = SIMD[T, 4].zeros()
            s = SIMD[T, 4].splat(x)
            v = SIMD[T, 4](z[0], s[1], x, z[3])
            return v[0], v[1], v[2], v[3]

        f_i8 = func[i8]
        f_u8 = func[u8]
        f_i32 = func[i32]
        f_u32 = func[u32]
        f_i64 = func[i64]
        f_u64 = func[u64]
        f_f32 = func[f32]
        f_f64 = func[f64]
        """
        mod = self.compile(src)
        assert mod.f_i8(-5) == (0, -5, -5, 0)
        assert mod.f_u8(200) == (0, 200, 200, 0)
        assert mod.f_i32(-42) == (0, -42, -42, 0)
        assert mod.f_u32(42) == (0, 42, 42, 0)
        assert mod.f_i64(-42) == (0, -42, -42, 0)
        assert mod.f_u64(42) == (0, 42, 42, 0)
        assert mod.f_f32(1.5) == (0.0, 1.5, 1.5, 0.0)
        assert mod.f_f64(2.25) == (0.0, 2.25, 2.25, 0.0)

    def test_no_default_constructor(self):
        src = """
        from _simd import SIMD

        VEC = SIMD[i32, 4]

        def bad() -> None:
            v = VEC()
        """
        errors = expect_errors(
            "SIMD requires explicit initialization",
            (
                "use `SIMD[i32, 4].zeros()`, `SIMD[i32, 4].splat(x)` or pass 4 values",
                "VEC",
            ),
        )
        self.compile_raises(src, "bad", errors)

    def test_wrong_number_of_args(self):
        src = """
        from _simd import SIMD

        def bad() -> None:
            v = SIMD[i32, 4](1, 2)
        """
        errors = expect_errors(
            "`SIMD[i32, 4]` expects 4 values, got 2",
            ("this is `SIMD[i32, 4]`", "SIMD[i32, 4]"),
        )
        self.compile_raises(src, "bad", errors)

    def test_convert_from_list_and_tuple(self):
        # items are implicitly converted to the dtype: here i32 -> f64
        src = """
        from _simd import SIMD

        VEC = SIMD[f64, 2]

        def list_literal() -> tuple[f64, f64]:
            v: VEC = [1, 2]
            return v[0], v[1]

        def tuple_literal() -> tuple[f64, f64]:
            v: VEC = (1, 2)
            return v[0], v[1]

        def list_runtime(a: i32, b: i32) -> tuple[f64, f64]:
            v: VEC = [a, b]
            return v[0], v[1]

        def tuple_runtime(a: i32, b: f64) -> tuple[f64, f64]:
            v: VEC = (a, b)
            return v[0], v[1]
        """
        mod = self.compile(src)
        assert mod.list_literal() == (1.0, 2.0)
        assert mod.tuple_literal() == (1.0, 2.0)
        assert mod.list_runtime(1, 2) == (1.0, 2.0)
        assert mod.tuple_runtime(1, 2.5) == (1.0, 2.5)

    def test_convert_from_tuple_wrong_length(self):
        # the length of a tuple is part of its type: this is a compile-time error
        src = """
        from _simd import SIMD

        def bad() -> None:
            v: SIMD[f64, 2] = (1, 2, 3)
        """
        errors = expect_errors(
            "`SIMD[f64, 2]` expects 2 values, got 3",
            ("this is `tuple[i32, i32, i32]`", "(1, 2, 3)"),
        )
        self.compile_raises(src, "bad", errors)

    def test_convert_from_wrong_length(self):
        # the length of a list is known only at runtime
        src = """
        from _simd import SIMD

        def bad(n: i32) -> i32:
            lst = [1, 2, 3]
            v: SIMD[i32, 4] = lst
            return v[0]
        """
        mod = self.compile(src)
        with SPyError.raises("W_ValueError", match="expected 4 elements"):
            mod.bad(0)

    def test_runtime_index(self):
        src = """
        from _simd import SIMD

        def get_lane(idx: i32) -> f64:
            v = SIMD[f64, 4](10.0, 20.0, 30.0, 40.0)
            return v[idx]

        def sum_lanes(a: i32, b: i32, c: i32, d: i32) -> i32:
            v = SIMD[i32, 4](a, b, c, d)
            s: i32 = 0
            for i in range(4):
                s = s + v[i]
            return s
        """
        mod = self.compile(src)
        assert [mod.get_lane(i) for i in range(4)] == [10.0, 20.0, 30.0, 40.0]
        assert mod.sum_lanes(1, 2, 3, 4) == 10
        assert mod.sum_lanes(10, 20, 30, 40) == 100

    @only_interp
    def test_index_out_of_bounds(self):
        src = """
        from _simd import SIMD

        def bad(i: i32) -> f32:
            v = SIMD[f32, 4].splat(1.0)
            return v[i]
        """
        mod = self.compile(src)
        with SPyError.raises("W_PanicError", match="SIMD index out of bounds"):
            mod.bad(4)
        with SPyError.raises("W_PanicError", match="SIMD index out of bounds"):
            mod.bad(-1)

    def test_store_load(self):
        src = """
        from unsafe import gc_alloc, gc_ptr
        from _simd import SIMD

        VEC = SIMD[i32, 2]

        def roundtrip(a: i32, b: i32) -> tuple[i32, i32]:
            p: gc_ptr[VEC] = gc_alloc[VEC](1)
            p[0] = VEC(a, b)
            v = p[0]
            return v[0], v[1]

        def overwrite() -> tuple[i32, i32]:
            p: gc_ptr[VEC] = gc_alloc[VEC](1)
            p[0] = VEC(1, 2)
            p[0] = VEC(3, 4)
            v = p[0]
            return v[0], v[1]

        def multiple() -> i32:
            p: gc_ptr[VEC] = gc_alloc[VEC](3)
            p[0] = VEC(1, 2)
            p[1] = VEC(3, 4)
            p[2] = VEC(5, 6)
            s: i32 = 0
            for i in range(3):
                v = p[i]
                s = s + v[0] + v[1]
            return s
        """
        mod = self.compile(src)
        assert mod.roundtrip(7, 8) == (7, 8)
        assert mod.overwrite() == (3, 4)
        assert mod.multiple() == 21

    def test_store_load_all_dtypes(self):
        src = """
        from unsafe import gc_alloc, gc_ptr
        from _simd import SIMD

        def func[T](x: T) -> tuple[T, T]:
            p: gc_ptr[SIMD[T, 2]] = gc_alloc[SIMD[T, 2]](1)
            p[0] = SIMD[T, 2].splat(x)
            v = p[0]
            return v[0], v[1]

        rt_i8 = func[i8]
        rt_u8 = func[u8]
        rt_i32 = func[i32]
        rt_u32 = func[u32]
        rt_i64 = func[i64]
        rt_u64 = func[u64]
        rt_f32 = func[f32]
        rt_f64 = func[f64]
        """
        mod = self.compile(src)
        assert mod.rt_i8(-5) == (-5, -5)
        assert mod.rt_u8(200) == (200, 200)
        assert mod.rt_i32(-42) == (-42, -42)
        assert mod.rt_u32(42) == (42, 42)
        assert mod.rt_i64(-42) == (-42, -42)
        assert mod.rt_u64(42) == (42, 42)
        assert mod.rt_f32(1.5) == (1.5, 1.5)
        assert mod.rt_f64(2.25) == (2.25, 2.25)

    def test_pass_and_return_by_value(self):
        src = """
        from _simd import SIMD

        VEC = SIMD[f32, 4]

        def identity(v: VEC) -> VEC:
            return v

        def entry() -> tuple[f32, f32, f32, f32]:
            v = VEC(1.0, 2.0, 3.0, 4.0)
            w = identity(v)
            return w[0], w[1], w[2], w[3]
        """
        mod = self.compile(src)
        assert mod.entry() == (1.0, 2.0, 3.0, 4.0)

    def test_arithmetic_all_dtypes(self):
        src = """
        from _simd import SIMD

        def func[T](x: T, y: T) -> tuple[T, T, T]:
            a = SIMD[T, 2](x, y)
            b = SIMD[T, 2](y, x)
            s = a + b
            d = a - b
            m = a * b
            return s[1], d[0], m[0]

        f_i8 = func[i8]
        f_u8 = func[u8]
        f_i32 = func[i32]
        f_u32 = func[u32]
        f_i64 = func[i64]
        f_u64 = func[u64]
        f_f32 = func[f32]
        f_f64 = func[f64]
        """
        mod = self.compile(src)
        assert mod.f_i8(7, 3) == (10, 4, 21)
        assert mod.f_u8(7, 3) == (10, 4, 21)
        assert mod.f_i32(7, 3) == (10, 4, 21)
        assert mod.f_u32(7, 3) == (10, 4, 21)
        assert mod.f_i64(7, 3) == (10, 4, 21)
        assert mod.f_u64(7, 3) == (10, 4, 21)
        assert mod.f_f32(1.5, 0.5) == (2.0, 1.0, 0.75)
        assert mod.f_f64(1.5, 0.5) == (2.0, 1.0, 0.75)

    def test_integer_wraparound(self):
        src = """
        from _simd import SIMD

        def func[T](x: T, y: T) -> tuple[T, T, T]:
            a = SIMD[T, 2].splat(x)
            b = SIMD[T, 2].splat(y)
            return (a + b)[0], (a - b)[0], (a * b)[0]

        f_i8 = func[i8]
        f_u8 = func[u8]
        f_i32 = func[i32]
        f_u32 = func[u32]
        """
        mod = self.compile(src)
        assert mod.f_i8(100, 100) == (-56, 0, 16)
        assert mod.f_u8(100, 200) == (44, 156, 32)
        assert mod.f_i32(2147483647, 2) == (-2147483647, 2147483645, -2)
        assert mod.f_u32(0, 1) == (1, 4294967295, 0)

    def test_float_division(self):
        src = """
        from _simd import SIMD

        def func[T](x: T, y: T) -> tuple[T, T]:
            q = SIMD[T, 2](x, y) / SIMD[T, 2](y, x)
            return q[0], q[1]

        f_f32 = func[f32]
        f_f64 = func[f64]
        """
        mod = self.compile(src)
        assert mod.f_f32(6.0, 3.0) == (2.0, 0.5)
        assert mod.f_f64(6.0, 3.0) == (2.0, 0.5)

    def test_integer_division_rejected(self):
        src = """
        from _simd import SIMD

        def bad() -> None:
            v = SIMD[i32, 4].splat(1)
            w = v / v
        """
        errors = expect_errors("cannot do `SIMD[i32, 4]` / `SIMD[i32, 4]`")
        self.compile_raises(src, "bad", errors)

    def test_comparison(self):
        src = """
        from _simd import SIMD

        def lt[T, M](x: T, y: T) -> tuple[M, M, M, M]:
            a = SIMD[T, 4](x, y, y, x)
            b = SIMD[T, 4](y, y, x, x)
            m = a < b
            return m[0], m[1], m[2], m[3]

        def le[T, M](x: T, y: T) -> tuple[M, M, M, M]:
            a = SIMD[T, 4](x, y, y, x)
            b = SIMD[T, 4](y, y, x, x)
            m = a <= b
            return m[0], m[1], m[2], m[3]

        def gt[T, M](x: T, y: T) -> tuple[M, M, M, M]:
            a = SIMD[T, 4](x, y, y, x)
            b = SIMD[T, 4](y, y, x, x)
            m = a > b
            return m[0], m[1], m[2], m[3]

        def ge[T, M](x: T, y: T) -> tuple[M, M, M, M]:
            a = SIMD[T, 4](x, y, y, x)
            b = SIMD[T, 4](y, y, x, x)
            m = a >= b
            return m[0], m[1], m[2], m[3]

        def eq[T, M](x: T, y: T) -> tuple[M, M, M, M]:
            a = SIMD[T, 4](x, y, y, x)
            b = SIMD[T, 4](y, y, x, x)
            m = a == b
            return m[0], m[1], m[2], m[3]

        def ne[T, M](x: T, y: T) -> tuple[M, M, M, M]:
            a = SIMD[T, 4](x, y, y, x)
            b = SIMD[T, 4](y, y, x, x)
            m = a != b
            return m[0], m[1], m[2], m[3]

        lt_i32 = lt[i32, i32]
        le_i32 = le[i32, i32]
        gt_i32 = gt[i32, i32]
        ge_i32 = ge[i32, i32]
        eq_i32 = eq[i32, i32]
        ne_i32 = ne[i32, i32]

        lt_i8 = lt[i8, i8]
        lt_u8 = lt[u8, i8]
        lt_u32 = lt[u32, i32]
        lt_f32 = lt[f32, i32]
        lt_i64 = lt[i64, i64]
        lt_u64 = lt[u64, i64]
        lt_f64 = lt[f64, i64]
        """
        mod = self.compile(src)
        assert mod.lt_i32(1, 2) == (-1, 0, 0, 0)
        assert mod.le_i32(1, 2) == (-1, -1, 0, -1)
        assert mod.gt_i32(1, 2) == (0, 0, -1, 0)
        assert mod.ge_i32(1, 2) == (0, -1, -1, -1)
        assert mod.eq_i32(1, 2) == (0, -1, 0, -1)
        assert mod.ne_i32(1, 2) == (-1, 0, -1, 0)

        assert mod.lt_i8(-1, 2) == (-1, 0, 0, 0)
        assert mod.lt_u8(200, 250) == (-1, 0, 0, 0)  # unsigned comparison
        assert mod.lt_u32(1, 4000000000) == (-1, 0, 0, 0)
        assert mod.lt_f32(1.5, 2.5) == (-1, 0, 0, 0)
        assert mod.lt_i64(1, 2) == (-1, 0, 0, 0)
        assert mod.lt_u64(1, 2) == (-1, 0, 0, 0)
        assert mod.lt_f64(1.5, 2.5) == (-1, 0, 0, 0)

    def test_select(self):
        src = """
        from _simd import SIMD

        def func[T](x: T, y: T) -> tuple[T, T, T, T]:
            a = SIMD[T, 4].splat(x)
            b = SIMD[T, 4].splat(y)
            m = SIMD[T, 4](x, y, y, x) < SIMD[T, 4](y, y, x, x)
            r = m.select(a, b)
            return r[0], r[1], r[2], r[3]

        # any integer vector of the same lane width can be a mask
        def handmade_mask() -> tuple[f32, f32, f32, f32]:
            m = SIMD[i32, 4](-1, 0, -1, 0)
            r = m.select(SIMD[f32, 4].splat(1.0), SIMD[f32, 4].splat(2.0))
            return r[0], r[1], r[2], r[3]

        f_i8 = func[i8]
        f_u8 = func[u8]
        f_i32 = func[i32]
        f_u32 = func[u32]
        f_i64 = func[i64]
        f_u64 = func[u64]
        f_f32 = func[f32]
        f_f64 = func[f64]
        """
        mod = self.compile(src)
        assert mod.f_i8(1, 2) == (1, 2, 2, 2)
        assert mod.f_u8(1, 2) == (1, 2, 2, 2)
        assert mod.f_i32(1, 2) == (1, 2, 2, 2)
        assert mod.f_u32(1, 2) == (1, 2, 2, 2)
        assert mod.f_i64(1, 2) == (1, 2, 2, 2)
        assert mod.f_u64(1, 2) == (1, 2, 2, 2)
        assert mod.f_f32(1.5, 2.5) == (1.5, 2.5, 2.5, 2.5)
        assert mod.f_f64(1.5, 2.5) == (1.5, 2.5, 2.5, 2.5)
        assert mod.handmade_mask() == (1.0, 2.0, 1.0, 2.0)

    @only_interp
    def test_select_invalid_mask(self):
        src = """
        from _simd import SIMD

        def bad_size() -> None:
            m = SIMD[i32, 2].zeros()
            m.select(SIMD[f32, 4].zeros(), SIMD[f32, 4].zeros())

        def bad_width() -> None:
            m = SIMD[i8, 4].zeros()
            m.select(SIMD[f32, 4].zeros(), SIMD[f32, 4].zeros())

        def bad_float() -> None:
            m = SIMD[f32, 4].zeros()
            m.select(SIMD[f32, 4].zeros(), SIMD[f32, 4].zeros())
        """

        def check_error(funcname: str, mask_t: str) -> None:
            msg = (
                "expected an integer SIMD with length 4 and 4-byte lanes "
                f"(e.g. `SIMD[i32, 4]`), got `{mask_t}`"
            )
            errors = expect_errors("invalid mask for `select`", (msg, "m"))
            self.compile_raises(src, funcname, errors)

        check_error("bad_size", "SIMD[i32, 2]")
        check_error("bad_width", "SIMD[i8, 4]")
        check_error("bad_float", "SIMD[f32, 4]")
