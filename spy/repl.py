"""
The SPy REPL (`spy repl FILE`, alias `ispy FILE`).

SPy has two very different kinds of scope: the module scope and the function
scope. A REPL has to respect that, so a session is split in two steps:

  1. `FILE` holds the module-scope code. It is imported once, as a normal
     module.

  2. The user then types statements at the beginning of an empty function (the
     "repl main"), living in the same module as FILE: everything defined in
     FILE is visible, and the variables created interactively are locals of
     that function, which stay alive for the whole session.

How it works: one `ASTFrame` for the repl main is created at the beginning and
lives as long as the session. We don't touch ScopeAnalyzer: for each input, we
run the normal parse/scope-analysis/astcompile pipeline on the *whole* source
(FILE, plus the repl main whose body is made of all the inputs accepted so far,
plus the new input). This is simple and gives the correct scoping for free.
Then we execute in the live frame only the statements which are new.
"""

import codeop
import copy
import linecache
import sys
import textwrap
from pathlib import Path
from typing import IO, Callable, Optional

import py.path

from spy import ast
from spy.analyze.importing import ImportAnalyzer
from spy.analyze.scope import ScopeAnalyzer
from spy.astcompile import astcompile
from spy.errors import SPyError
from spy.location import Loc
from spy.parser import Parser
from spy.util import OrderedSet
from spy.vm.astframe import ASTFrame
from spy.vm.b import B
from spy.vm.function import W_ASTFunc
from spy.vm.vm import SPyVM

# Name of the function whose frame is used by the session. FILE is free to define its
# own `main`, which the user can then call from the REPL.
REPL_MAIN = "__repl_main__"

EXIT_COMMANDS = ("exit", "exit()", "quit", "quit()")


class _ReplImporter(ImportAnalyzer):
    """
    Import FILE as a module, but with the repl main appended to it.

    By overriding astcompile_one we get, for free, the handling of everything which
    FILE imports (explicit and implicit imports, import order, errors, .spyc cache of
    the dependencies, etc.)
    """

    def __init__(self, session: "ReplSession", use_spyc: bool) -> None:
        super().__init__(session.vm, session.modname, use_spyc=use_spyc)
        self.session = session

    def astcompile_one(self, modname: str, spyfile: py.path.local) -> ast.Module:
        if modname == self.session.modname:
            return self.session.compile([])
        return super().astcompile_one(modname, spyfile)


class ReplSession:
    """
    A REPL session: see the module docstring.

    This class does no I/O, except for the `print`s done by the SPy code which is
    executed: see `run_repl` for the interactive loop.
    """

    vm: SPyVM
    filename: str
    modname: str
    prelude: str  # source code of FILE
    use_spyc: bool
    inputs: list[str]  # inputs accepted so far
    n_done: int  # number of compiled stmts, in the repl main, already executed
    frame: ASTFrame

    def __init__(self, vm: SPyVM, filename: Path, *, use_spyc: bool = True) -> None:
        self.vm = vm
        self.filename = str(filename)
        self.modname = filename.stem
        self.prelude = filename.read_text()
        self.use_spyc = use_spyc
        self.inputs = []
        self.n_done = 0

        if self.modname in vm.modules_w:
            # e.g. a file called `math.spy`: ImportAnalyzer would consider it as
            # already imported and never compile it
            raise SPyError(
                "W_ImportError",
                f"cannot start the REPL on `{filename.name}`: "
                f"a module called `{self.modname}` already exists, please rename it",
            )
        importer = _ReplImporter(self, use_spyc)
        importer.astcompile_all()
        importer.import_all()

        w_mod = vm.modules_w[self.modname]
        w_main = w_mod.getattr_maybe(vm, REPL_MAIN)
        assert isinstance(w_main, W_ASTFunc)
        self.frame = ASTFrame(vm, w_main, [])
        self.frame.declare_arguments()

    def compile(self, inputs: list[str]) -> ast.Module:
        """
        Parse, analyze and astcompile FILE + the repl main containing `inputs`.
        """
        src = self.prelude
        if not src.endswith("\n"):
            src += "\n"
        src += f"\n\ndef {REPL_MAIN}() -> None:\n"
        if inputs:
            src += textwrap.indent("\n".join(inputs), "    ") + "\n"
        else:
            src += "    pass\n"

        # we want the source lines of both FILE and of the inputs to be shown in the
        # error messages: they are all in `src`. An entry without mtime is never
        # invalidated by linecache.checkcache
        linecache.cache[self.filename] = (
            len(src),
            None,
            src.splitlines(keepends=True),
            self.filename,
        )
        parsed = Parser(src, self.filename).parse()
        assert parsed.scoping_rules in ("strict", "pythonic")
        sa = ScopeAnalyzer(self.modname, parsed)
        sa.analyze()
        return astcompile(parsed, scope_analyzer=sa)

    def _load_imports(self, mod: ast.Module) -> None:
        """
        Load the modules needed by an input but not loaded yet (e.g. `_list` for a
        list literal). This is the same "imports come first" rule which applies to
        normal modules: each input is analyzed as a whole before being executed.
        """
        importer = ImportAnalyzer(self.vm, self.modname, use_spyc=self.use_spyc)
        importer.queue.clear()
        importer.deps[self.modname] = OrderedSet()
        importer.cur_modname = self.modname
        for implicit in mod.implicit_imports:
            importer.record_import(self.modname, implicit, node=None)
        importer.visit(mod)
        importer.cur_modname = None
        if importer.queue:
            importer.astcompile_all()
            importer.import_all()

    @staticmethod
    def _get_repl_main(mod: ast.Module) -> ast.FuncDef:
        for decl in mod.decls:
            if isinstance(decl, ast.GlobalFuncDef) and decl.funcdef.name == REPL_MAIN:
                return decl.funcdef
        assert False, "cannot find the repl main"

    def run(self, src: str) -> Optional[str]:
        """
        Execute one input, which can contain multiple statements.

        If the input ends with an expression whose value is not None, return its
        repr. Raise SPyError in case of errors.

        An input is atomic as far as the local variables are concerned: if it fails,
        the variables are restored to what they were before. Side effects on the
        objects are not undone (e.g. a list which was appended to).
        """
        mod = self.compile(self.inputs + [src])
        self._load_imports(mod)
        funcdef = self._get_repl_main(mod)
        new_stmts = funcdef.body.body[self.n_done :]

        frame = self.frame
        old_frameinfo = frame.frameinfo
        old_locals = {name: copy.copy(lv) for name, lv in frame.locals.items()}
        frame.frameinfo = funcdef.frameinfo
        for slot_name in funcdef.frameinfo._symbols:
            if slot_name not in frame.locals:
                frame.reset_local(slot_name)
        try:
            wam = frame.run_interactive(funcdef.body.replace(body=new_stmts))
        except BaseException:
            frame.frameinfo = old_frameinfo
            frame.locals = old_locals
            raise

        # commit
        frame.funcdef = funcdef
        self.inputs.append(src)
        self.n_done = len(funcdef.body.body)

        if wam is None or wam.w_val is B.w_None:
            return None
        wam_s = self.vm.repr_wam(wam, loc=Loc.here())
        return self.vm.unwrap_str(wam_s.w_val)


def is_complete(src: str) -> bool:
    """
    Tell whether `src` is a complete input, or whether the REPL should ask for more
    lines (e.g. the body of a `for`). It's the same logic as the standard
    `code.InteractiveConsole`: SPy syntax is Python syntax, so we ask Python.
    """
    try:
        return codeop.compile_command(src, symbol="single") is not None
    except (SyntaxError, ValueError, OverflowError):
        # we let SPy report the real error
        return True


def run_repl(
    session: ReplSession,
    *,
    read_line: Optional[Callable[[str], str]] = None,
    out: Optional[IO[str]] = None,
    use_colors: bool = True,
) -> None:
    """
    The interactive loop. `read_line` and `out` can be overridden for testing.
    """
    if out is None:
        out = sys.stdout
    if read_line is None:
        try:
            import readline  # noqa: F401  # enables line editing/history in input()
        except ImportError:
            pass
        read_line = input

    buf: list[str] = []
    while True:
        try:
            line = read_line("... " if buf else ">>> ")
        except EOFError:
            print(file=out)
            return
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt", file=out)
            buf.clear()
            continue

        if not buf and line.strip() in EXIT_COMMANDS:
            return
        buf.append(line)
        src = "\n".join(buf)
        if not src.strip():
            buf.clear()
            continue
        if not is_complete(src):
            continue

        buf.clear()
        try:
            res = session.run(src)
        except SPyError as e:
            print(e.format(use_colors=use_colors), file=out)
        else:
            if res is not None:
                print(res, file=out)
