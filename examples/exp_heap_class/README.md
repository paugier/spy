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

`heap_decorator.spy` injects the methods with an unrolled loop in the body of
the generated class:

    for i in unroll_range(N):
        locals()[method_names[i]] = methods[method_names[i]]

This needs three things:

- `For` in a class body. Today: "`For` not supported inside a classdef"
  (ClassFrame only allows VarDef, AssignLocal, If, Pass, FuncDef). It would
  be restricted to `unroll_range`, so that it is fully resolved at blue time.
- `unroll_range`, from the dataclass note (not in the repo yet; Antonio is
  working on unroll loops).
- a way to bind a name computed at blue time. `locals()[name] = v` is the
  Python spelling; since `name` is blue in each unrolled iteration, ClassFrame
  can declare the local statically.

Assigning a method template re-evaluates its `def` in the new class scope, so
`self` becomes the generated class and not the user class. A plain copy of the
function would keep `self` typed as the user class.

Alternative without a loop, by analogy with `__extra_fields__`:
`__extra_methods__ = methods`. Less general, but no new statement in class
bodies.
