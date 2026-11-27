import builtins
import readline
import sys
import textwrap
from io import StringIO

import pytest
from typer.testing import CliRunner

import spy.cli
from spy.cli import app
from spy.errors import SPyError
from spy.repl import (
    DEFAULT_PRELUDE,
    ReplSession,
    _make_completer,
    is_complete,
    run_repl,
)
from spy.vm.vm import SPyVM

PRELUDE = """
from math import sqrt

def square(x: i32) -> i32:
    return x * x

def main() -> None:
    print("hello from main")
"""

# NOTE: creating a SPyVM and importing the modules needed by `range` and list
# literals is what makes these tests slow. Tests which need them are merged into
# few big ones, which share one session.


@pytest.fixture
def make_session(tmp_path):
    """
    Make a session based on a FILE. With `run_main=[...]`, it is a `-x` session:
    main() of the file is run first, and the session starts in its frame.
    """

    def make(src=PRELUDE, name="mymod", **kwargs):
        f = tmp_path / f"{name}.spy"
        f.write_text(textwrap.dedent(src))
        vm = SPyVM()
        vm.path.append(str(tmp_path))
        return ReplSession(vm, f, use_spyc=False, **kwargs)

    return make


def bare_session(*, module_scope=False):
    """
    A session without FILE: it starts with the default prelude only.
    """
    return ReplSession(SPyVM(), None, module_scope=module_scope, use_spyc=False)


def run_lines(session, lines):
    """
    Run `run_repl` on `session`, feeding it `lines` (an exception class in `lines`
    is raised by the "prompt"); return (output, prompts).
    """
    it = iter(lines)
    prompts = []

    def read_line(prompt):
        prompts.append(prompt)
        try:
            item = next(it)
        except StopIteration:
            raise EOFError
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item
        return item

    out = StringIO()
    run_repl(session, read_line=read_line, out=out, use_colors=False)
    return out.getvalue(), prompts


class TestMainScope:
    def test_basics(self, make_session, capsys):
        s = make_session()
        assert s.run("1 + 2") == "3"
        assert s.run("print(42)") is None  # None is not displayed
        assert capsys.readouterr().out == "42\n"
        assert s.run("x = 3") is None
        assert s.run("y = x + 1") is None
        assert s.run("y") == "4"
        # the module scope (the FILE) is visible
        assert s.run("square(x)") == "9"
        assert s.run("sqrt(16.0)") == "4.0"
        # the repl main must not clash with a `main` defined in the FILE
        s.run("main()")
        assert capsys.readouterr().out == "hello from main\n"
        # the second assignment turns `x` from const to var: this is why we
        # re-analyze everything at each input
        s.run("x = 5")
        assert s.run("x") == "5"

    def test_loops_and_lists(self, make_session):
        s = make_session()
        s.run("total = 0\nfor i in range(4):\n    total += i")
        assert s.run("total") == "6"
        # a `for` is compiled to more than one stmt: check that we execute them all
        s.run("for j in range(2):\n    total += 10")
        assert s.run("total") == "26"
        # a list literal needs the implicit import of the corresponding module
        s.run("w = [1, 2, 3]")
        assert s.run("len(w)") == "3"

    def test_errors_and_rollback(self, make_session):
        s = make_session()
        s.run("x = 1")
        with pytest.raises(SPyError) as exc:
            s.run("undefined_name + 1")
        assert exc.value.etype == "W_NameError"
        out = exc.value.format(use_colors=False)
        # line numbers are relative to the inputs, which are all concatenated:
        # this is the second one
        assert "<repl>:2" in out
        assert "undefined_name + 1" in out
        assert s.run("x") == "1"

        # a runtime error undoes the assignments done by the failed input
        with pytest.raises(SPyError) as exc:
            s.run("x = 2\nraise ValueError('boom')")
        assert exc.value.etype == "W_ValueError"
        assert s.run("x") == "1"
        # the failed input is not part of the session: we can use its names again
        s.run("x = 7")
        assert s.run("x") == "7"

    def test_module_name_already_in_use(self, make_session):
        with pytest.raises(SPyError) as exc:
            make_session("", name="time")
        assert exc.value.etype == "W_ImportError"
        assert "rename" in exc.value.w_exc.message

    def test_is_complete(self):
        assert is_complete("x = 1")
        assert is_complete("1 + 2")
        assert is_complete("for i in range(3):\n    print(i)\n")
        # not valid Python: SPy must be the one reporting the error
        assert is_complete("x = = 1")

        assert not is_complete("for i in range(3):")
        assert not is_complete("for i in range(3):\n    print(i)")
        assert not is_complete("if x:\n    pass\nelse:")
        assert not is_complete("foo(1,")


class TestTracebacks:
    """
    Tracebacks and error messages about the inputs show `<repl>`, not the FILE.
    """

    def test_file(self, make_session):
        s = make_session("def boom() -> None:\n    raise ValueError('boom')\n")
        with pytest.raises(SPyError) as exc:
            s.run("boom()")
        out = exc.value.format(use_colors=False)
        # the frame of the input shows <repl>, with a line number relative to
        # the inputs, and the corresponding source line
        assert "  * <repl> at <repl>:1\n" in out
        assert "boom()" in out
        # the frame of boom() still points to the FILE
        assert f"  * mymod::boom at {s.filename}:" in out

    def test_no_file(self):
        s = bare_session()
        with pytest.raises(SPyError) as exc:
            s.run("raise ValueError('boom')")
        out = exc.value.format(use_colors=False)
        assert "  * <repl> at <repl>:" in out
        assert "raise ValueError('boom')" in out

    def test_no_file_line_numbers_are_relative_to_the_inputs(self):
        s = bare_session()
        with pytest.raises(SPyError) as exc:
            s.run("raise ValueError('boom')")
        assert "  * <repl> at <repl>:1\n" in exc.value.format(use_colors=False)


class TestRunRepl:
    def test_loop(self):
        s = bare_session()
        lines = [
            "x = 3",
            "x + 1",
            "",
            "while x < 5:",
            "    x += 1",
            "",
            "x",
            "z",  # an error does not stop the loop
            "if x:",
            KeyboardInterrupt,  # neither does ^C, which also drops the pending input
            "1 + 1",
            "exit",
            "99",  # never executed
        ]
        out, prompts = run_lines(s, lines)
        assert out.startswith("4\n5\n")
        assert "NameError: name `z` is not defined" in out
        assert "\nKeyboardInterrupt\n" in out
        assert out.rstrip().endswith("2")
        assert "99" not in out
        assert (
            prompts
            == [">>> "] * 4 + ["... "] * 2 + [">>> "] * 3 + ["... "] + [">>> "] * 2
        )

    def test_interrupt_during_execution(self, monkeypatch):
        # ^C while an input is running does not kill the REPL: the input is
        # dropped and the session stays alive
        s = bare_session()
        orig_run = s.run

        def run(src):
            if src.startswith("while True:"):
                raise KeyboardInterrupt
            return orig_run(src)

        monkeypatch.setattr(s, "run", run)
        out, _ = run_lines(s, ["x = 1", "while True:", "    pass", "", "x", "exit"])
        assert "\nKeyboardInterrupt\n" in out
        assert out.rstrip().endswith("1")

    def test_internal_error_during_execution(self, monkeypatch):
        # a non-SPyError exception during the execution of an input (an
        # internal bug) is shown as a traceback, and does not kill the REPL
        s = bare_session()
        orig_run = s.run

        def run(src):
            if src == "bug":
                raise RecursionError("maximum recursion depth exceeded")
            return orig_run(src)

        monkeypatch.setattr(s, "run", run)
        out, _ = run_lines(s, ["1 + 1", "bug", "2 + 2", "exit"])
        assert out.startswith("2\n")
        assert "Traceback (most recent call last)" in out
        assert "RecursionError: maximum recursion depth exceeded" in out
        assert out.rstrip().endswith("4")

    def test_question_mark_help(self, make_session):
        s = make_session()
        out, _ = run_lines(s, ["?square", "square?", "?", "?nosuch", "1 + 1"])
        assert out.count("def square(i32) -> i32") == 2
        assert "usage: ?NAME or NAME?" in out
        assert "NameError: name `nosuch` is not defined" in out
        assert out.rstrip().endswith("2")

    def test_question_mark_only_for_names(self, make_session):
        s = make_session()
        out, _ = run_lines(s, ["x = 1  # why?", "x", "1 + 1?", "square?", "exit"])
        # `x = 1  # why?` ends with `?` but is executed as normal code
        assert out.startswith("1\n")
        # `1 + 1?` is not a name either: SPy reports the syntax error
        assert "ParseError" in out
        # `square?` is still a help request
        assert "def square(i32) -> i32" in out

    def test_history_file(self, make_session, tmp_path, monkeypatch):
        readline.clear_history()
        hf = tmp_path / "ispy_history"
        lines = iter(["1 + 1", "x = 3", "exit"])
        monkeypatch.setattr(builtins, "input", lambda prompt="": next(lines))
        s = make_session()
        out = StringIO()
        run_repl(s, out=out, use_colors=False, history_file=hf)
        assert out.getvalue() == "2\n"
        # the history was saved to the file, and can be read back
        readline.clear_history()
        readline.read_history_file(str(hf))
        n = readline.get_current_history_length()
        items = [readline.get_history_item(i) for i in range(1, n + 1)]
        assert items == ["1 + 1", "x = 3", "exit"]

        # saving the history is best-effort: an unwritable file is not an error
        lines = iter(["exit"])
        run_repl(s, out=out, use_colors=False, history_file=tmp_path / "no" / "hf")


class TestModuleScope:
    def test_workflow(self):
        s = bare_session(module_scope=True)
        assert s.mode == "module"
        assert s.filename == "<repl:prelude>"
        assert s.prelude == DEFAULT_PRELUDE
        # the prelude is not imported yet: only builtins are visible
        assert "print" in s.complete("pri")

        s.add_module_input("import math")
        s.add_module_input("def area(r: f64) -> f64:\n    return math.pi * r * r")
        s.add_module_input("def twice(x: i32) -> i32:\n    return x + x")
        s.add_module_input("def f(x: i32) -> i32:\n    return twice(x) + 1")
        s.add_module_input("answer = 42")
        assert s.mode == "module"  # nothing ran yet

        # an input which fails the check is not appended to the prelude
        with pytest.raises(SPyError) as exc:
            s.add_module_input("print(1)")
        assert exc.value.etype == "W_ParseError"
        assert "print(1)" not in s.prelude
        with pytest.raises(SPyError) as exc:
            s.add_module_input("answer = 43")
        assert exc.value.etype == "W_ScopeError"
        assert "answer = 43" not in s.prelude

        s.enter_main()
        assert s.mode == "main"
        assert s.run("f(2)") == "5"
        assert s.run("area(2.0)") == "12.566370614359172"
        assert s.run("answer") == "42"
        # names imported by the default prelude can be used without importing them
        assert s.run("as_red(42)") == "42"
        assert s.run("interp_list[i32]") == "<spy type '__spy__::interp_list[i32]'>"

    def test_file_as_prelude(self, make_session):
        s = make_session(module_scope=True)
        assert s.mode == "module"
        assert "def square" in s.prelude
        assert s.prelude.startswith(DEFAULT_PRELUDE)
        s.enter_main()
        assert s.mode == "main"
        assert s.run("square(7)") == "49"

    def test_loop(self):
        s = bare_session(module_scope=True)
        lines = [
            "def square(x: i32) -> i32:",
            "    return x * x",
            "",
            "print(1)",  # an error does not stop the loop
            "%main",
            "square(4)",
        ]
        out, prompts = run_lines(s, lines)
        assert "ParseError" in out
        assert "16" in out
        assert prompts == ["module>>> ", "... ", "... ", "module>>> ", "module>>> "] + [
            ">>> ",
            ">>> ",
        ]
        assert s.mode == "main"

    def test_commands(self, tmp_path):
        good = tmp_path / "good.spy"
        good.write_text("def cube(x: i32) -> i32:\n    return x * x * x\n")
        bad = tmp_path / "bad.spy"
        bad.write_text("print(1)\n")
        s = bare_session(module_scope=True)
        lines = [
            f"%load {good}",
            "%load",
            "%load /does/not/exist.spy",
            f"%load {bad}",
            "import does_not_exist",
            "%main",  # fails, we stay in module scope
            "%reset",  # forgets everything, including `cube`
            "%main",
            "cube(2)",
        ]
        out, prompts = run_lines(s, lines)
        assert "usage: %load FILE" in out
        assert "cannot load `/does/not/exist.spy`" in out
        assert "ParseError" in out
        assert "ImportError" in out
        assert "NameError: name `cube` is not defined" in out
        assert prompts[-2:] == [">>> ", ">>> "]
        assert prompts[-3] == "module>>> "
        assert s.mode == "main"

    def test_commands_in_main_scope(self, make_session):
        s = make_session()
        out, _ = run_lines(s, ["%main", "%load foo.spy", "%frobnicate", "1 + 1"])
        assert "`%main` is only available in module scope" in out
        assert "`%load` is only available in module scope" in out
        assert "unknown command `%frobnicate`" in out
        assert out.rstrip().endswith("2")


class TestReset:
    def test_reset(self, make_session):
        s = make_session()
        assert s.modname == "mymod"
        s.run("x = 3")
        s.reset()
        assert s.mode == "module"
        assert s.filename == "<repl:prelude>"
        assert s.prelude == DEFAULT_PRELUDE
        assert s.inputs == []
        assert s.n_done == 0
        assert s.frame is None
        # `mymod` is already imported and cannot be redefined: a fresh module
        # name is used
        assert s.modname == "repl"
        s.add_module_input("def double(x: i32) -> i32:\n    return 2 * x")
        s.enter_main()
        assert s.run("double(21)") == "42"
        with pytest.raises(SPyError) as exc:
            s.run("x")  # forgotten
        assert exc.value.etype == "W_NameError"
        s.reset()
        assert s.modname == "repl_2"

    def test_reset_command(self, make_session):
        s = make_session()
        lines = [
            "x = 1",
            "%reset",
            "def f() -> i32:",
            "    return 7",
            "",
            "%main",
            "f()",
        ]
        out, prompts = run_lines(s, lines)
        assert out.startswith("7\n")
        assert prompts == (
            [">>> "] * 2 + ["module>>> ", "... ", "... ", "module>>> "] + [">>> "] * 2
        )
        assert s.mode == "main"

    def test_reset_in_module_scope(self):
        s = bare_session(module_scope=True)
        s.add_module_input("def square(x: i32) -> i32:\n    return x * x")
        s.reset()
        assert s.mode == "module"
        assert s.prelude == DEFAULT_PRELUDE
        # the module was never imported, so the name `repl` is still available
        assert s.modname == "repl"
        s.add_module_input("def cube(x: i32) -> i32:\n    return x * x * x")
        s.enter_main()
        assert s.run("cube(3)") == "27"
        with pytest.raises(SPyError) as exc:
            s.run("square(3)")
        assert exc.value.etype == "W_NameError"


class TestExecuteMode:
    """
    -x: run main() first, then start the session in its frame.
    """

    def test_main_frame(self, make_session, capsys):
        s = make_session(
            """
            def square(x: i32) -> i32:
                return x * x

            def main(argv: list[str]) -> i32:
                n = len(argv)
                x = 3
                total = 0
                for i in range(4):
                    total += i
                for a in argv:
                    print(a)
                return 42
            """,
            name="xmod",
            run_main=["hello", "world"],
        )
        # main() ran, with argv = [FILE] + ARGS
        assert capsys.readouterr().out.splitlines() == [s.filename, "hello", "world"]
        assert s.mode == "main"
        assert s.main_error is None
        # an exit code does not end the session
        assert s.main_exit_code == 42

        # a traceback shows <repl> also in this mode
        with pytest.raises(SPyError) as exc:
            s.run("raise ValueError('boom')")
        assert "  * <repl> at <repl>:1\n" in exc.value.format(use_colors=False)

        # the locals of main() and the definitions of the FILE are visible
        assert s.run("n") == "3"
        assert s.run("x") == "3"
        assert s.run("total") == "6"
        assert s.run("square(6)") == "36"
        # the inputs run in the frame of main()
        s.run("x = 10")
        assert s.run("x + total") == "16"
        # a `for` compiles to several stmts: we execute them all, without
        # re-executing the body of main()
        s.run("for j in range(2):\n    total += 10")
        assert s.run("total") == "26"
        # a failed input is undone
        with pytest.raises(SPyError) as exc:
            s.run("x = 2\nraise ValueError('boom')")
        assert exc.value.etype == "W_ValueError"
        assert s.run("x") == "10"

    def test_main_raises(self, make_session):
        s = make_session(
            """
            def main() -> None:
                x = 1
                raise ValueError('boom')
            """,
            name="xraise",
            run_main=[],
        )
        assert s.mode == "main"
        assert s.main_error.etype == "W_ValueError"
        # the error is shown, but the session starts anyway, in the frame of main()
        out, _ = run_lines(s, ["x", "1 + 1"])
        assert "ValueError" in out
        assert "boom" in out
        assert out.rstrip().endswith("1\n2")

    def test_single_line_main(self, make_session):
        s = make_session("def main() -> None: x = 3\n", name="oneline", run_main=[])
        assert s.run("x") == "3"
        s.run("x = 4")
        assert s.run("x") == "4"

    def test_main_must_be_indented_with_4_spaces(self, make_session):
        # with -x, the inputs are injected into the body of main with a
        # 4-space indent: a main indented differently is rejected at startup
        for i, src in enumerate(
            ["def main() -> None:\n  x = 1\n", "def main() -> None:\n\tx = 1\n"]
        ):
            with pytest.raises(SPyError) as exc:
                make_session(src, name=f"badindent{i}", run_main=[])
            assert exc.value.etype == "W_TypeError"
            assert "indented with 4 spaces" in exc.value.w_exc.message

    def test_no_main(self, make_session):
        with pytest.raises(SPyError) as exc:
            make_session("x = 1\n", name="nomain", run_main=[])
        assert exc.value.etype == "W_TypeError"
        assert "main" in exc.value.w_exc.message

    def test_reset(self, make_session):
        s = make_session("def main() -> None:\n    x = 1\n", run_main=[])
        s.reset()
        assert s.run_main is None
        assert s.main_error is None
        assert s.mode == "module"
        s.enter_main()
        # we are back to the normal repl main: `x` is not defined anymore
        with pytest.raises(SPyError) as exc:
            s.run("x")
        assert exc.value.etype == "W_NameError"


class TestCompleteAndInspect:
    def test_complete_and_inspect(self, make_session, capsys):
        s = make_session(
            """
            import math

            def square(x: i32) -> i32:
                "return the square of x"
                return x * x

            def side_effect() -> i32:
                print("SIDE EFFECT")
                return 1
            """,
            name="cmod",
        )
        s.run("x = 3")
        s.run("w = [1, 2, 3]")

        # plain names: locals, globals of the module, builtins
        assert s.complete("squa") == ["square"]
        assert s.complete("x") == ["x"]
        assert "print" in s.complete("pri")
        # attributes
        assert s.complete("math.sq") == ["math.sqrt"]
        assert "math.pi" in s.complete("math.p")
        assert s.complete("w.ap") == ["w.append"]
        # names starting with `_` are proposed only if explicitly requested
        for prefix in ("math.", "w."):
            names = s.complete(prefix)
            assert names
            assert all(not n.rpartition(".")[2].startswith("_") for n in names)

        # inspect
        out = s.inspect("square")
        assert "def square(i32) -> i32" in out
        assert "return the square of x" in out
        assert s.inspect("math") == "module math"
        assert s.inspect("math.sqrt").startswith("def math.sqrt(")
        assert s.inspect("i32") == "type i32"
        assert s.inspect("x") == "type: i32"
        assert s.inspect("w.append").startswith("def w.append(")
        with pytest.raises(SPyError) as exc:
            s.inspect("math.nosuch")
        assert exc.value.etype == "W_TypeError"
        assert "has no attribute `nosuch`" in exc.value.w_exc.message

        # completing and inspecting never evaluate calls, so no side effects
        with pytest.raises(SPyError) as exc:
            s.complete("side_effect().")
        assert exc.value.etype == "W_TypeError"
        assert "cannot evaluate `side_effect()`" in exc.value.w_exc.message
        assert capsys.readouterr().out == ""

        # the readline completer
        completer = _make_completer(s)
        assert completer("squa", 0) == "square"
        assert completer("squa", 1) is None
        assert completer("nosuch.", 0) is None  # errors are swallowed

    def test_complete_edge_cases(self, make_session, monkeypatch):
        s = make_session()
        # nothing before the dot: no completions, and no crash
        assert s.complete(".x") == []
        assert s.complete(".") == []
        completer = _make_completer(s)
        assert completer(".x", 0) is None

        # the readline completer swallows any error, not just SPyError
        def boom(prefix):
            raise IndexError("boom")

        monkeypatch.setattr(s, "complete", boom)
        assert completer("x", 0) is None


class TestCli:
    def test_repl_without_file(self):
        # `ispy` without a file starts in module scope, with the default prelude
        res = CliRunner().invoke(
            app,
            ["repl"],
            input="def cube(x: i32) -> i32:\n    return x * x * x\n\n%main\ncube(3)\n",
        )
        assert res.exit_code == 0, res.output
        assert "27" in res.stdout

    @pytest.mark.parametrize(
        "args, input",
        [
            (["repl"], "x = 6\nsquare(x)\n"),
            (["repl", "-m"], "%main\nsquare(6)\n"),
        ],
    )
    def test_repl_file(self, tmp_path, args, input):
        f = tmp_path / "mymod.spy"
        f.write_text(textwrap.dedent(PRELUDE))
        res = CliRunner().invoke(app, [*args, str(f)], input=input)
        assert res.exit_code == 0, res.output
        assert "36" in res.stdout

    def test_repl_x(self, tmp_path):
        f = tmp_path / "mymod.spy"
        f.write_text(
            "def main(argv: list[str]) -> None:\n"
            "    n = len(argv)\n"
            "    for a in argv:\n"
            "        print(a)\n"
        )
        res = CliRunner().invoke(app, ["repl", "-x", str(f), "hello"], input="n\n")
        assert res.exit_code == 0, res.output
        lines = res.stdout.splitlines()
        assert lines[0] == str(f)
        assert lines[1] == "hello"
        assert lines[2] == ">>> 2"  # the repr of `n`, after the prompt

    def test_repl_usage_errors(self, tmp_path):
        res = CliRunner().invoke(app, ["repl", "-x"], input="")
        assert res.exit_code != 0
        assert "requires a FILE" in res.output

        f = tmp_path / "mymod.spy"
        f.write_text(textwrap.dedent(PRELUDE))
        res = CliRunner().invoke(app, ["repl", str(f), "extra"], input="")
        assert res.exit_code != 0
        assert "-x/--execute" in res.output

    def test_repl_x_main_must_be_indented_with_4_spaces(self, tmp_path):
        f = tmp_path / "mymod.spy"
        f.write_text("def main() -> None:\n  x = 1\n")
        res = CliRunner().invoke(app, ["repl", "-x", str(f)], input="")
        assert res.exit_code != 0
        assert "indented with 4 spaces" in res.output

    def test_execute_interactive(self, tmp_path):
        # `spy -i FILE [ARGS...]` is the same as `ispy -x FILE [ARGS...]`
        f = tmp_path / "mymod.spy"
        f.write_text("def main(argv: list[str]) -> None:\n    n = len(argv)\n")
        res = CliRunner().invoke(app, ["-i", str(f), "a", "b"], input="n\n")
        assert res.exit_code == 0, res.output
        assert ">>> 3\n" in res.stdout

    def test_ispy_entry_point(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            spy.cli, "pyproject_entry_point", lambda: calls.append(list(sys.argv))
        )
        monkeypatch.setattr(sys, "argv", ["ispy", "foo.spy"])
        spy.cli.ispy_entry_point()
        assert sys.argv == ["ispy", "repl", "foo.spy"]
        assert calls == [["ispy", "repl", "foo.spy"]]
