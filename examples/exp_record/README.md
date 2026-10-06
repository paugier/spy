# Experiment: `Record`, a heap-allocated class

Goal: `class Point(metaclass=Record)` defines two structs, `Point::Data` (the fields) and
`Point` (a wrapper holding `__ll__: gc_ptr[Point::Data]`). `Record` is a metaclass written
in SPy, in `stdlib/metaclasses/record.spy`: there is no decorator, a class statement has a
metaclass (see `metaclasses.md`).

Three implementations of the same `Point` type (two `i32` fields `x` and `y`, and a
method `norm2`):

- `record` (`point_record.spy`): the user code, `class Point(metaclass=Record)`. Does
  not run: the parser rejects the keyword in the class statement.
- `by-hand` (`point_by_hand.spy`): exactly the code that `Record` should generate.
  Runs, except `is` (no `__is__` hook yet).
- `simple` (`point_simple.spy`): simplified so that the tests pass, with a `main()`
  demo (`spy examples/exp_record/point_simple.spy`). Runs, except `is`.

`test_point.py` runs the same tests on the three implementations: each test imports
`Point` from `module_name`, a fixture parametrized with the three module names (each
one carries its marker). `conftest.py` reuses the fixtures/options of `spy/tests` and
registers the markers.

```sh
pytest test_point.py                 # the three implementations
pytest test_point.py -m record
pytest test_point.py -m by-hand
pytest test_point.py -m simple
pytest test_point.py -m "simple and interp"
```

Add `-m "simple and interp"` (or `doppler`, `C`) to select a backend. The tests that
are known to fail are `xfail`: all of them for `record`, and `test_identity` for the
other two. Use `--runxfail` to see the real errors.

The metaclass itself is `stdlib/metaclasses/record.spy`. The other metaclasses
(`ClassMetaClass`, `DataClass`, `Protocol`, `Variant`) are in the same
directory, with a README that lists the invented API and what the experiment found.

Changes needed in the compiler:

- `spy/vm/modules/operator/attrop.py` (included): if a custom `__getattribute__` returns
  `OpSpec.NULL`, fall back to the default lookup. Without it, reading `self.__ll__`
  inside the hook recurses forever.
