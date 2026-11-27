import sys
import textwrap
from io import StringIO

import pytest
from typer.testing import CliRunner

import spy.cli
from spy.cli import app
from spy.errors import SPyError
from spy.repl import DEFAULT_PRELUDE, ReplSession, is_complete, run_repl
from spy.vm.vm import SPyVM

PRELUDE = """
from math import sqrt

def square(x: i32) -> i32:
    return x * x

def main() -> None:
    print("hello from main")
"""


@pytest.fixture
def make_session(tmp_path):
    def make(prelude=PRELUDE, name="mymod", module_scope=False):
        f = tmp_path / f"{name}.spy"
        f.write_text(textwrap.dedent(prelude))
        vm = SPyVM()
        vm.path.append(str(tmp_path))
        return ReplSession(vm, f, module_scope=module_scope, use_spyc=False)

    return make


@pytest.fixture
def make_x_session(tmp_path):
    """
    Like make_session, but for -x mode: main() of the file is run first, and
    the session starts in its frame.
    """

    def make(src, args=None, name="xmod"):
        f = tmp_path / f"{name}.spy"
        f.write_text(textwrap.dedent(src))
        vm = SPyVM()
        vm.path.append(str(tmp_path))
        return ReplSession(vm, f, use_spyc=False, run_main=args or [])

    return make


def run_lines(session: ReplSession, lines: list[str]) -> tuple[str, list[str]]:
    """
    Run `run_repl` on `session`, feeding it `lines`; return (output, prompts).
    """
    it = iter(lines)
    prompts: list[str] = []

    def read_line(prompt: str) -> str:
        prompts.append(prompt)
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    out = StringIO()
    run_repl(session, read_line=read_line, out=out, use_colors=False)
    return out.getvalue(), prompts


class TestRepl:
    # ReplSession, main scope

    def test_expression(self, make_session):
        s = make_session()
        assert s.run("1 + 2") == "3"

    def test_none_is_not_displayed(self, make_session, capsys):
        s = make_session()
        assert s.run("print(42)") is None
        assert capsys.readouterr().out == "42\n"

    def test_locals_persist(self, make_session):
        s = make_session()
        assert s.run("x = 3") is None
        assert s.run("y = x + 1") is None
        assert s.run("y") == "4"

    def test_module_scope_is_visible(self, make_session):
        s = make_session()
        s.run("x = 5")
        assert s.run("square(x)") == "25"
        assert s.run("sqrt(16.0)") == "4.0"

    def test_call_main_of_the_file(self, make_session, capsys):
        # the repl main must not clash with a `main` defined in the file
        s = make_session()
        s.run("main()")
        assert capsys.readouterr().out == "hello from main\n"

    def test_reassign(self, make_session):
        # the second assignment turns `x` from const to var: this is why we
        # re-analyze everything at each input
        s = make_session()
        s.run("x = 1")
        s.run("x = 2")
        assert s.run("x") == "2"

    def test_multiple_statements_and_loops(self, make_session):
        s = make_session()
        s.run("total = 0\nfor i in range(4):\n    total += i")
        assert s.run("total") == "6"
        # a `for` is compiled to more than one stmt: check that we execute them all
        s.run("for j in range(2):\n    total += 10")
        assert s.run("total") == "26"

    def test_list_needs_implicit_import(self, make_session):
        s = make_session()
        s.run("w = [1, 2, 3]")
        assert s.run("len(w)") == "3"

    def test_empty_prelude(self, make_session):
        s = make_session(prelude="", name="empty")
        assert s.run("1 + 1") == "2"

    def test_static_error_and_recovery(self, make_session):
        s = make_session()
        s.run("x = 1")
        with pytest.raises(SPyError) as exc:
            s.run("z")
        assert exc.value.etype == "W_NameError"
        assert s.run("x") == "1"

    def test_runtime_error_rolls_back_locals(self, make_session):
        s = make_session()
        s.run("x = 1")
        with pytest.raises(SPyError) as exc:
            s.run("x = 2\nraise ValueError('boom')")
        assert exc.value.etype == "W_ValueError"
        assert s.run("x") == "1"
        # the failed input is not part of the session: we can use its names again
        s.run("x = 7")
        assert s.run("x") == "7"

    def test_error_shows_source_line(self, make_session):
        s = make_session()
        with pytest.raises(SPyError) as exc:
            s.run("undefined_name + 1")
        assert "undefined_name + 1" in exc.value.format(use_colors=False)

    def test_error_in_prelude_imports(self, make_session):
        with pytest.raises(SPyError) as exc:
            make_session(prelude="import does_not_exist\n")
        assert exc.value.etype == "W_ImportError"

    def test_module_name_already_in_use(self, make_session):
        with pytest.raises(SPyError) as exc:
            make_session(prelude="", name="time")
        assert exc.value.etype == "W_ImportError"
        assert "rename" in exc.value.w_exc.message

    # is_complete

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

    # run_repl, main scope

    def test_loop(self, make_session):
        s = make_session()
        lines = [
            "x = 3",
            "x + 1",
            "",
            "for i in range(2):",
            "    x += i",
            "",
            "x",
        ]
        out, prompts = run_lines(s, lines)
        assert out == "4\n4\n\n"
        assert prompts == [
            ">>> ",
            ">>> ",
            ">>> ",
            ">>> ",
            "... ",
            "... ",
            ">>> ",
            ">>> ",
        ]

    def test_error_does_not_stop_the_loop(self, make_session):
        s = make_session()
        out, _ = run_lines(s, ["z", "1 + 1"])
        assert "NameError: name `z` is not defined" in out
        assert out.rstrip().endswith("2")

    def test_exit(self, make_session):
        s = make_session()
        out, _ = run_lines(s, ["1", "exit", "2"])
        assert out == "1\n"

    # CLI

    def test_repl_command(self, tmp_path):
        f = tmp_path / "mymod.spy"
        f.write_text(textwrap.dedent(PRELUDE))
        res = CliRunner().invoke(
            app, ["repl", str(f)], input="x = 6\nsquare(x)\nexit\n"
        )
        assert res.exit_code == 0, res.output
        assert "36" in res.stdout

    def test_repl_command_without_file(self):
        # `ispy` without a file starts in module scope, with the default prelude
        res = CliRunner().invoke(
            app,
            ["repl"],
            input="def cube(x: i32) -> i32:\n    return x * x * x\n\n%main\ncube(3)\nexit\n",
        )
        assert res.exit_code == 0, res.output
        assert "27" in res.stdout

    def test_repl_command_module_scope_flag(self, tmp_path):
        f = tmp_path / "mymod.spy"
        f.write_text(textwrap.dedent(PRELUDE))
        res = CliRunner().invoke(
            app, ["repl", "-m", str(f)], input="%main\nsquare(6)\nexit\n"
        )
        assert res.exit_code == 0, res.output
        assert "36" in res.stdout

    def test_ispy_entry_point(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            spy.cli, "pyproject_entry_point", lambda: calls.append(list(sys.argv))
        )
        monkeypatch.setattr(sys, "argv", ["ispy", "foo.spy"])
        spy.cli.ispy_entry_point()
        assert sys.argv == ["ispy", "repl", "foo.spy"]
        assert calls == [["ispy", "repl", "foo.spy"]]

    # ReplSession, module scope

    def test_no_file_starts_with_default_prelude(self):
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        assert s.mode == "module"
        assert s.filename == "<repl>"
        assert s.prelude == DEFAULT_PRELUDE

    def test_file_as_prelude(self, make_session):
        s = make_session(module_scope=True)
        assert s.mode == "module"
        assert "def square" in s.prelude
        # the default prelude comes first
        assert s.prelude.startswith(DEFAULT_PRELUDE)
        s.enter_main()
        assert s.mode == "main"
        assert s.run("square(7)") == "49"

    def test_add_module_input_then_main(self):
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        s.add_module_input("def square(x: i32) -> i32:\n    return x * x")
        assert s.mode == "module"  # nothing ran yet
        s.enter_main()
        assert s.run("square(3)") == "9"

    def test_module_input_can_use_previous_inputs(self):
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        s.add_module_input("import math")
        s.add_module_input("def area(r: f64) -> f64:\n    return math.pi * r * r")
        s.add_module_input("def twice(x: i32) -> i32:\n    return x + x")
        s.add_module_input("def f(x: i32) -> i32:\n    return twice(x) + 1")
        s.enter_main()
        assert s.run("f(2)") == "5"
        assert s.run("area(2.0)") == "12.566370614359172"

    def test_module_input_is_checked(self):
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        with pytest.raises(SPyError) as exc:
            s.add_module_input("print(1)")
        assert exc.value.etype == "W_ParseError"
        # the failed input was not appended to the prelude
        assert "print(1)" not in s.prelude

    def test_module_input_scope_error_leaves_prelude_unchanged(self):
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        s.add_module_input("answer = 42")
        with pytest.raises(SPyError) as exc:
            s.add_module_input("answer = 43")
        assert exc.value.etype == "W_ScopeError"
        assert "answer = 43" not in s.prelude
        s.enter_main()
        assert s.run("answer") == "42"

    def test_default_prelude_is_available_in_main_scope(self):
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        s.enter_main()
        # names imported by the default prelude can be used without importing them
        assert s.run("as_red(42)") == "42"
        assert s.run("interp_list[i32]") == "<spy type '__spy__::interp_list[i32]'>"

    def test_enter_main_import_error(self):
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        s.add_module_input("import does_not_exist")
        with pytest.raises(SPyError) as exc:
            s.enter_main()
        assert exc.value.etype == "W_ImportError"
        # the session is still in module scope
        assert s.mode == "module"

    def test_load(self, tmp_path):
        f = tmp_path / "extra.spy"
        f.write_text("def cube(x: i32) -> i32:\n    return x * x * x\n")
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        s.load(str(f))
        assert "def cube" in s.prelude
        s.enter_main()
        assert s.run("cube(3)") == "27"

    def test_load_missing_file(self):
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        with pytest.raises(OSError):
            s.load("/does/not/exist.spy")

    # run_repl, module scope and commands

    def test_module_scope_prompts(self):
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        lines = [
            "def square(x: i32) -> i32:",
            "    return x * x",
            "",
            "%main",
            "square(4)",
        ]
        out, prompts = run_lines(s, lines)
        assert out == "16\n\n"
        assert prompts == [
            "module>>> ",
            "... ",
            "... ",
            "module>>> ",
            ">>> ",
            ">>> ",
        ]
        assert s.mode == "main"

    def test_module_scope_error_does_not_stop_the_loop(self):
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        out, _ = run_lines(s, ["print(1)", "x = 1", "%main", "x"])
        assert "ParseError" in out
        assert out.rstrip().endswith("1")

    def test_load_command(self, tmp_path):
        f = tmp_path / "extra.spy"
        f.write_text("def cube(x: i32) -> i32:\n    return x * x * x\n")
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        out, _ = run_lines(s, [f"%load {f}", "%main", "cube(2)"])
        assert out == "8\n\n"

    def test_load_command_missing_file(self):
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        out, _ = run_lines(s, ["%load /does/not/exist.spy", "%main", "1 + 1"])
        assert "cannot load `/does/not/exist.spy`" in out
        assert out.rstrip().endswith("2")

    def test_load_command_usage(self):
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        out, _ = run_lines(s, ["%load", "%main", "1 + 1"])
        assert "usage: %load FILE" in out

    def test_commands_are_not_available_in_main_scope(self, make_session):
        s = make_session()
        out, _ = run_lines(s, ["%main", "%load foo.spy", "1 + 1"])
        assert "`%main` is only available in module scope" in out
        assert "`%load` is only available in module scope" in out
        assert out.rstrip().endswith("2")

    def test_unknown_command(self, make_session):
        s = make_session()
        out, _ = run_lines(s, ["%frobnicate", "1 + 1"])
        assert "unknown command `%frobnicate`" in out
        assert out.rstrip().endswith("2")

    # %reset

    def test_reset(self, make_session):
        s = make_session()
        assert s.modname == "mymod"
        s.run("x = 3")
        s.reset()
        assert s.mode == "module"
        assert s.filename == "<repl>"
        assert s.prelude == DEFAULT_PRELUDE
        assert s.inputs == []
        assert s.n_done == 0
        assert s.frame is None
        # `mymod` is already imported and cannot be redefined: a fresh module
        # name is used
        assert s.modname == "repl"
        s.enter_main()
        assert s.run("1 + 1") == "2"
        s.reset()
        assert s.modname == "repl_2"
        s.enter_main()
        assert s.run("2 + 2") == "4"

    def test_reset_forgets_inputs(self, make_session):
        s = make_session()
        s.run("x = 3")
        s.reset()
        s.add_module_input("def double(x: i32) -> i32:\n    return 2 * x")
        s.enter_main()
        assert s.run("double(21)") == "42"
        with pytest.raises(SPyError) as exc:
            s.run("x")
        assert exc.value.etype == "W_NameError"

    def test_reset_in_module_scope(self):
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        s.add_module_input("def square(x: i32) -> i32:\n    return x * x")
        s.reset()
        assert s.mode == "module"
        assert s.prelude == DEFAULT_PRELUDE
        assert "square" not in s.prelude
        # the module was never imported, so the name `repl` is still available
        assert s.modname == "repl"
        s.add_module_input("def cube(x: i32) -> i32:\n    return x * x * x")
        s.enter_main()
        assert s.run("cube(3)") == "27"
        with pytest.raises(SPyError) as exc:
            s.run("square(3)")
        assert exc.value.etype == "W_NameError"

    def test_reset_command_in_main_scope(self, make_session):
        s = make_session()
        lines = [
            "x = 1",
            "%reset",
            "def f() -> i32:",
            "    return 7",
            "",
            "%main",
            "f()",
            "x",
        ]
        out, prompts = run_lines(s, lines)
        assert "7" in out
        assert "NameError" in out
        assert prompts == [
            ">>> ",
            ">>> ",
            "module>>> ",
            "... ",
            "... ",
            "module>>> ",
            ">>> ",
            ">>> ",
            ">>> ",
        ]
        assert s.mode == "main"

    def test_reset_command_in_module_scope(self):
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        s.add_module_input("def square(x: i32) -> i32:\n    return x * x")
        out, prompts = run_lines(s, ["%reset", "%main", "1 + 1"])
        assert out.rstrip().endswith("2")
        assert "NameError" not in out
        assert prompts[0] == "module>>> "
        assert prompts[1] == "module>>> "
        assert prompts[2] == ">>> "

    # -x: run main() first, then start the session in its frame

    def test_x_runs_main_first(self, make_x_session, capsys):
        s = make_x_session("""
            def main() -> None:
                print("hello from main")
        """)
        assert capsys.readouterr().out == "hello from main\n"
        assert s.mode == "main"
        assert s.main_error is None
        assert s.main_exit_code is None

    def test_x_locals_of_main_are_visible(self, make_x_session):
        s = make_x_session("""
            def main() -> None:
                x = 3
                total = 0
                for i in range(4):
                    total += i
        """)
        assert s.run("x") == "3"
        assert s.run("total") == "6"

    def test_x_inputs_run_in_main_frame(self, make_x_session):
        s = make_x_session("""
            def main() -> None:
                x = 3
                total = 0
                for i in range(4):
                    total += i
        """)
        s.run("x = 10")
        assert s.run("x") == "10"
        assert s.run("x + total") == "16"
        # a `for` in an input compiles to several stmts: check that we execute
        # them all, without re-executing main's body
        s.run("for j in range(2):\n    total += 10")
        assert s.run("total") == "26"

    def test_x_file_definitions_are_visible(self, make_x_session):
        s = make_x_session("""
            def square(x: i32) -> i32:
                return x * x

            def main() -> None:
                pass
        """)
        assert s.run("square(6)") == "36"

    def test_x_argv(self, make_x_session, capsys):
        s = make_x_session(
            """
            def main(argv: list[str]) -> None:
                n = len(argv)
                for a in argv:
                    print(a)
            """,
            args=["hello", "world"],
        )
        assert capsys.readouterr().out.splitlines() == [s.filename, "hello", "world"]
        assert s.run("n") == "3"  # FILE + 2 args

    def test_x_main_raises_but_session_still_starts(self, make_x_session):
        s = make_x_session("""
            def main() -> None:
                x = 1
                raise ValueError('boom')
        """)
        assert s.mode == "main"
        assert s.main_error is not None
        assert s.main_error.etype == "W_ValueError"
        # the locals set before the raise are available at the prompt
        assert s.run("x") == "1"

    def test_x_main_error_is_printed(self, make_x_session):
        s = make_x_session(
            """
            def main() -> None:
                raise ValueError('boom')
            """,
            name="xraise",
        )
        out, _ = run_lines(s, ["1 + 1"])
        assert "ValueError" in out
        assert "boom" in out
        assert out.rstrip().endswith("2")

    def test_x_exit_code_does_not_end_the_session(self, make_x_session):
        s = make_x_session("""
            def main() -> i32:
                x = 5
                return 42
        """)
        assert s.main_exit_code == 42
        assert s.mode == "main"
        assert s.run("x") == "5"
        assert s.run("x * 2") == "10"

    def test_x_input_error_rolls_back_locals(self, make_x_session):
        s = make_x_session("""
            def main() -> None:
                x = 1
        """)
        with pytest.raises(SPyError) as exc:
            s.run("x = 2\nraise ValueError('boom')")
        assert exc.value.etype == "W_ValueError"
        assert s.run("x") == "1"

    def test_x_single_line_main(self, make_x_session):
        s = make_x_session("""
            def main() -> None: x = 3
        """)
        assert s.run("x") == "3"
        s.run("x = 4")
        assert s.run("x") == "4"

    def test_x_no_main(self, tmp_path):
        f = tmp_path / "nomain.spy"
        f.write_text("x = 1\n")
        vm = SPyVM()
        vm.path.append(str(tmp_path))
        with pytest.raises(SPyError) as exc:
            ReplSession(vm, f, use_spyc=False, run_main=[])
        assert exc.value.etype == "W_TypeError"
        assert "main" in exc.value.w_exc.message

    def test_x_wrong_main_signature(self, tmp_path):
        f = tmp_path / "badsig.spy"
        f.write_text("def main(x: i32) -> None:\n    pass\n")
        vm = SPyVM()
        vm.path.append(str(tmp_path))
        with pytest.raises(SPyError) as exc:
            ReplSession(vm, f, use_spyc=False, run_main=[])
        assert exc.value.etype == "W_TypeError"

    def test_x_reset_clears_run_main(self, make_x_session):
        s = make_x_session("""
            def main() -> None:
                x = 1
        """)
        s.reset()
        assert s.run_main is None
        assert s.main_error is None
        assert s.mode == "module"
        s.enter_main()
        # we are back to the normal repl main: `x` is not defined anymore
        with pytest.raises(SPyError) as exc:
            s.run("x")
        assert exc.value.etype == "W_NameError"

    # CLI: -x and -i

    def test_repl_command_x(self, tmp_path):
        f = tmp_path / "mymod.spy"
        f.write_text(
            textwrap.dedent(
                """
                def main(argv: list[str]) -> None:
                    n = len(argv)
                    for a in argv:
                        print(a)
                """
            )
        )
        res = CliRunner().invoke(
            app, ["repl", "-x", str(f), "hello"], input="n\nexit\n"
        )
        assert res.exit_code == 0, res.output
        lines = res.stdout.splitlines()
        assert lines[0] == str(f)
        assert lines[1] == "hello"
        assert lines[2] == ">>> 2"  # the repr of `n`, after the prompt

    def test_repl_command_x_requires_file(self):
        res = CliRunner().invoke(app, ["repl", "-x"], input="")
        assert res.exit_code != 0
        assert "requires a FILE" in res.output

    def test_repl_command_argv_without_x(self, tmp_path):
        f = tmp_path / "mymod.spy"
        f.write_text(textwrap.dedent(PRELUDE))
        res = CliRunner().invoke(app, ["repl", str(f), "extra"], input="exit\n")
        assert res.exit_code != 0
        assert "-x/--execute" in res.output

    def test_execute_command_interactive(self, tmp_path):
        # `spy -i FILE [ARGS...]` is the same as `ispy -x FILE [ARGS...]`
        f = tmp_path / "mymod.spy"
        f.write_text(
            textwrap.dedent(
                """
                def main() -> None:
                    x = 21
                """
            )
        )
        res = CliRunner().invoke(app, ["-i", str(f)], input="x * 2\nexit\n")
        assert res.exit_code == 0, res.output
        assert "42" in res.stdout

    def test_execute_command_interactive_with_args(self, tmp_path):
        f = tmp_path / "mymod.spy"
        f.write_text(
            textwrap.dedent(
                """
                def main(argv: list[str]) -> None:
                    n = len(argv)
                """
            )
        )
        res = CliRunner().invoke(app, ["-i", str(f), "a", "b"], input="n\nexit\n")
        assert res.exit_code == 0, res.output
        assert ">>> 3\n" in res.stdout

    # history file

    def test_history_file(self, make_session, tmp_path, monkeypatch):
        import builtins
        import readline

        readline.clear_history()
        hf = tmp_path / "ispy_history"
        lines = iter(["1 + 1", "x = 3", "exit"])
        monkeypatch.setattr(builtins, "input", lambda prompt="": next(lines))
        s = make_session()
        out = StringIO()
        run_repl(s, out=out, use_colors=False, history_file=hf)
        assert out.getvalue() == "2\n"
        # the history was saved to the file, and can be read back
        assert hf.exists()
        readline.clear_history()
        readline.read_history_file(str(hf))
        n = readline.get_current_history_length()
        items = [readline.get_history_item(i) for i in range(1, n + 1)]
        assert items == ["1 + 1", "x = 3", "exit"]

    # tracebacks show <repl>

    def test_traceback_shows_repl(self, tmp_path):
        f = tmp_path / "mymod.spy"
        f.write_text(
            textwrap.dedent(
                """
                def boom() -> None:
                    raise ValueError('boom')
                """
            )
        )
        vm = SPyVM()
        vm.path.append(str(tmp_path))
        s = ReplSession(vm, f, use_spyc=False)
        with pytest.raises(SPyError) as exc:
            s.run("boom()")
        out = exc.value.format(use_colors=False)
        # the frame of the input shows <repl>, with a line number relative to
        # the inputs, and the corresponding source line
        assert "  * <repl> at <repl>:1\n" in out
        assert "boom()" in out
        # the frame of boom() still points to the FILE
        assert f"  * mymod::boom at {f}:" in out

    def test_traceback_shows_repl_no_file(self):
        s = ReplSession(SPyVM(), None, use_spyc=False)
        with pytest.raises(SPyError) as exc:
            s.run("raise ValueError('boom')")
        out = exc.value.format(use_colors=False)
        assert "  * <repl> at <repl>:" in out
        assert "raise ValueError('boom')" in out

    def test_traceback_shows_repl_x_mode(self, make_x_session):
        s = make_x_session(
            """
            def main() -> None:
                x = 1
            """
        )
        with pytest.raises(SPyError) as exc:
            s.run("raise ValueError('boom')")
        out = exc.value.format(use_colors=False)
        assert "  * <repl> at <repl>:1\n" in out

    def test_static_error_shows_repl(self, make_session):
        s = make_session()
        with pytest.raises(SPyError) as exc:
            s.run("undefined_name + 1")
        out = exc.value.format(use_colors=False)
        assert "<repl>:1" in out
        assert "undefined_name + 1" in out

    def test_error_in_loop_shows_repl(self, make_session):
        s = make_session()
        out, _ = run_lines(s, ["raise ValueError('boom')", "1 + 1"])
        assert "<repl>" in out
        assert out.rstrip().endswith("2")

    # complete() and inspect()

    def test_complete_plain_names(self, make_session):
        s = make_session()
        assert s.complete("squa") == ["square"]
        assert s.complete("mai") == ["main"]
        assert "print" in s.complete("pri")
        s.run("x = 3")
        assert s.complete("x") == ["x"]

    def test_complete_attribute_chain(self, tmp_path):
        f = tmp_path / "m.spy"
        f.write_text("import math\n")
        vm = SPyVM()
        vm.path.append(str(tmp_path))
        s = ReplSession(vm, f, use_spyc=False)
        assert s.complete("math.sq") == ["math.sqrt"]
        assert s.complete("math.sqrt") == ["math.sqrt"]
        assert "math.pi" in s.complete("math.p")
        # names starting with `_` are proposed only if explicitly requested
        assert all(
            not c.rpartition(".")[2].startswith("_") for c in s.complete("math.")
        )

    def test_complete_instance_attributes(self, make_session):
        s = make_session()
        s.run("w = [1, 2, 3]")
        assert s.complete("w.ap") == ["w.append"]
        assert "w.append" in s.complete("w.")

    def test_complete_in_module_scope(self):
        s = ReplSession(SPyVM(), None, module_scope=True, use_spyc=False)
        # the prelude is not imported yet: only builtins are visible
        assert "print" in s.complete("pri")
        with pytest.raises(SPyError) as exc:
            s.inspect("nosuch")
        assert exc.value.etype == "W_NameError"

    def test_complete_never_evaluates_calls(self, tmp_path, capsys):
        f = tmp_path / "m.spy"
        f.write_text(
            textwrap.dedent(
                """
                def side_effect() -> i32:
                    print('SIDE EFFECT')
                    return 1
                """
            )
        )
        vm = SPyVM()
        vm.path.append(str(tmp_path))
        s = ReplSession(vm, f, use_spyc=False)
        assert s.complete("side_eff") == ["side_effect"]
        with pytest.raises(SPyError) as exc:
            s.complete("side_effect().")
        assert exc.value.etype == "W_TypeError"
        # nothing was executed
        assert capsys.readouterr().out == ""

    def test_inspect_function(self, tmp_path):
        f = tmp_path / "m.spy"
        f.write_text(
            textwrap.dedent(
                """
                def square(x: i32) -> i32:
                    "return the square of x"
                    return x * x
                """
            )
        )
        vm = SPyVM()
        vm.path.append(str(tmp_path))
        s = ReplSession(vm, f, use_spyc=False)
        out = s.inspect("square")
        assert "def square(i32) -> i32" in out
        assert "return the square of x" in out

    def test_inspect_module_type_and_value(self, tmp_path):
        f = tmp_path / "m.spy"
        f.write_text("import math\n")
        vm = SPyVM()
        vm.path.append(str(tmp_path))
        s = ReplSession(vm, f, use_spyc=False)
        assert s.inspect("math") == "module math"
        assert s.inspect("math.sqrt").startswith("def math.sqrt(")
        assert s.inspect("i32") == "type i32"
        s.run("x = 3")
        assert s.inspect("x") == "type: i32"

    def test_inspect_rejects_calls(self, make_session):
        s = make_session()
        with pytest.raises(SPyError) as exc:
            s.inspect("square(3)")
        assert exc.value.etype == "W_TypeError"
        assert "cannot evaluate `square(3)`" in exc.value.w_exc.message

    def test_readline_completer(self, make_session):
        from spy.repl import _make_completer

        s = make_session()
        completer = _make_completer(s)
        assert completer("squa", 0) == "square"
        assert completer("squa", 1) is None

    # ? help in the loop

    def test_question_mark_help(self, make_session):
        s = make_session()
        out, _ = run_lines(s, ["?square", "square?", "?", "?nosuch", "1 + 1"])
        assert out.count("def square(i32) -> i32") == 2
        assert "usage: ?NAME or NAME?" in out
        assert "NameError: name `nosuch` is not defined" in out
        assert out.rstrip().endswith("2")
