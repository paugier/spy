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
