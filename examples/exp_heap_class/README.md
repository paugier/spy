# Experiment: heap-allocated classes

Goal: `@heap class Point` generates two structs, `Point_data` (the fields) and
`Point` (a wrapper holding `__ll__: gc_ptr[Point_data]`), see ROADMAP.md
("support for heap-allocated `class`es").

Files:

- `point_by_hand.spy`: the generated code, written by hand, with a `main()`
  demo. Run with `spy examples/exp_heap_class/point_by_hand.spy`.
- `point_by_hand_checks.spy`: imports `Point` from `point_by_hand` and
  defines the functions called by the tests.
- `test_heap_by_hand.py`: tests for part 1 (passing).
- `heap_decorator.spy`: the `@heap` decorator definition (does not run yet).
- `point_heap.spy`: the user code, a `@heap class Point` (does not run yet).
- `heap_decorator_checks.spy`: imports `Point` from `point_heap`, functions
  called by the decorator tests.
- `test_heap_decorator.py`: tests for part 2 (skipped).
- `conftest.py`: reuses the fixtures/options of `spy/tests`.

Run the tests with `pytest examples/exp_heap_class`.

Changes needed in the compiler:

- `spy/vm/modules/operator/attrop.py` (included): if a custom
  `__getattribute__` returns `OpSpec.NULL`, fall back to the default lookup.
  Without it, reading `self.__ll__` inside the hook recurses forever.

Still missing for the `@heap` decorator (part 2):

1. class decorators other than `@struct` (parser: "class decorators not
    supported yet")
2. `cls.__annotations__` (fields) and `cls.__methods__` (method templates,
    whose `self` is not bound to a class yet), see the dataclass note
3. `(*methods.keys(),)` and variadic `__new__` (`*m_args`,
    `OpSpec(impl, [*m_args])`, spylang/spy#710), plus `self.__init__(*args)`
    in red code
4. an `__is__` hook: `is` is currently rejected on value types
5. method injection, see below

## Proposed syntax to inject the user methods

The decorator has to copy the user-written methods (`__init__`, `norm2`, ...)
into the generated class. The simplest proposal is a second reserved local in
the class body, by analogy with `__extra_fields__`:

```python
@struct
class _heap:
    __ll__: gc_ptr[_data]
    __extra_methods__ = cls.__methods__

    # generated metafunctions: __getattribute__, __setattr__, __new__, ...
```

Today `__extra_fields__` is a special local of `ClassFrame`: it is declared
with type `interp_dict[str, type]`, so that its assignment is typechecked, and
`ClassFrame.run` moves it into the class body, where the struct machinery
turns it into fields. `__extra_methods__` would work the same way:

  - it is declared as an `interp_dict[str, <method>]` and typechecked at the
    assignment;
  - its entries are merged into the class namespace when the class is created,
    as if the `def`s were written in the body;
  - a name defined both explicitly and in `__extra_methods__` is an error.

Compared to the fields, there is one subtlety. The values of
`cls.__methods__` are method *templates*: the `self` of the user's
`def norm2(self)` is not bound to any class yet. When they are merged,
each template is instantiated in the new class scope, so that `self` is
`_heap` and not the user class. Copying the functions as they are would keep
`self` typed as the user class.

What this needs:

  - `cls.__methods__`: an ordered `{name: method template}` dict, next to
    `cls.__annotations__`;
  - `__extra_methods__` handling in `ClassFrame` and in the struct class
    creation (a few lines, mirroring `__extra_fields__`).

What this does not need: any new statement in class bodies, `unroll_range`, or
the blue `for` loops that Antonio is working on. So it is independent from
them.

The limit is that it is all or nothing: every user method is copied as is.
The decorator cannot skip, rename or wrap a method (for example to add a
check before `__init__`). A more general alternative, an unrolled `for` loop
in the class body, is left to discuss.
