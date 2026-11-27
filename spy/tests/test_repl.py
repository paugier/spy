import textwrap
from io import StringIO

import pytest
from typer.testing import CliRunner

from spy.cli import app
from spy.errors import SPyError
from spy.repl import ReplSession, is_complete, run_repl
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
    def make(prelude=PRELUDE, name="mymod"):
        f = tmp_path / f"{name}.spy"
        f.write_text(textwrap.dedent(prelude))
        vm = SPyVM()
        vm.path.append(str(tmp_path))
        return ReplSession(vm, f, use_spyc=False)

    return make


class TestRepl:
    # ReplSession

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

    # run_repl

    def run_lines(self, session, lines):
        it = iter(lines)
        self.prompts = []

        def read_line(prompt):
            self.prompts.append(prompt)
            try:
                return next(it)
            except StopIteration:
                raise EOFError

        out = StringIO()
        run_repl(session, read_line=read_line, out=out, use_colors=False)
        return out.getvalue()

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
        out = self.run_lines(s, lines)
        assert out == "4\n4\n\n"
        assert self.prompts == [
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
        out = self.run_lines(s, ["z", "1 + 1"])
        assert "NameError: name `z` is not defined" in out
        assert out.rstrip().endswith("2")

    def test_exit(self, make_session):
        s = make_session()
        out = self.run_lines(s, ["1", "exit", "2"])
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