# Experiment: `Record`, a heap-allocated class

Goal: `class Point(metaclass=Record)` defines two structs, `Point::Data` (the fields) and
`Point` (a wrapper holding `__ll__: gc_ptr[Point::Data]`), see ROADMAP.md ("support for
heap-allocated `class`es"). `Record` is a metaclass written in SPy, in
`stdlib/metaclasses/record.spy`: there is no decorator, a class statement has a
metaclass (see `metaclasses.md`).

Files:

- `point_by_hand.spy`: the generated code, written by hand, with a `main()` demo. Run
  with `spy examples/exp_record/point_by_hand.spy`.
- `point_by_hand_checks.spy`: imports `Point` from `point_by_hand` and defines the
  functions called by the tests.
- `test_record_by_hand.py`: tests for part 1 (passing).
- `point_record.spy`: the user code, `class Point(metaclass=Record)` with a written
  `__init__` (does not run yet).
- `point_record_generated.spy`: `class Vec(metaclass=Record)` with only fields: `__init__`,
  `__eq__`, `__repr__` are generated (does not run yet).
- `record_checks.spy`: imports `Point` and `Vec`, functions called by the Record tests.
- `test_record.py`: tests for part 2 (skipped).
- `conftest.py`: reuses the fixtures/options of `spy/tests`.

The metaclass itself is `stdlib/metaclasses/record.spy`. The other metaclasses
(`ClassMetaClass`, `DataClass`, `Protocol`, `Variant`) are in the same
directory, with a README that lists the invented API and what the experiment found.

Run the tests with `pytest examples/exp_record`.

Changes needed in the compiler:

- `spy/vm/modules/operator/attrop.py` (included): if a custom `__getattribute__` returns
  `OpSpec.NULL`, fall back to the default lookup. Without it, reading `self.__ll__`
  inside the hook recurses forever.

Still missing for `Record` (part 2), in the order of `metaclasses.md`, section 7:

1. `P1`-`P6`: starred expressions in blue code, `.keys()`/`.values()`/`.items()` on
   `interp_dict`, `*args: tuple[*types]` and `f(*args)` in red code, unrolled loops,
   `isinstance`.
2. `M1.1`-`M1.6`: `w_meta` and `w_base`, a generic declaration, bases and `metaclass=`
   in the parser, a metaclass that inherits from `StructType`, `StructType` and
   `ClassBody` visible from SPy.
3. `M1.7`: default values in fields (not needed by these two examples).
4. keywords in a class statement, if `implements=` is kept (see
   `stdlib/metaclasses/README.md`, finding 5), and an `__is__` hook: `is` is currently
   rejected on value types.
5. modules in a directory: the test adds `stdlib/metaclasses` to `vm.path`, because
   there are no packages yet.
