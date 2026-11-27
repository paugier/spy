"""
The SPy REPL (`ispy`, alias `spy repl`).

SPy has two very different kinds of scope: the module scope and the function
scope. A REPL has to respect that, so a session has two modes:

  1. module scope (prompt `module>>>`): the user types module-scope code:
     imports, `def`, `class` and global assignments. Nothing runs there: each
     input is checked (parsed and scope-analyzed together with the prelude)
     and appended to the *prelude*, i.e. the module-scope source code of the
     session. `%main` freezes the prelude and switches to main scope.

  2. main scope (prompt `>>>`): the user types statements and expressions,
     which run in the live frame of an empty function, the "repl main", living
     in the same module as the prelude: everything defined in the prelude is
     visible, and the variables created interactively are locals of that
     function, which stay alive for the whole session.

Every session starts with the DEFAULT_PRELUDE, a few imports that SPy code
commonly needs.

How main scope works: one `ASTFrame` for the repl main is created when the
session switches to main scope and lives as long as the session. We don't
touch ScopeAnalyzer: for each input, we run the normal
parse/scope-analysis/astcompile pipeline on the *whole* source (the prelude,
plus the repl main whose body is made of all the inputs accepted so far, plus
the new input). This is simple and gives the correct scoping for free. Then we
execute in the live frame only the statements which are new.
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

# Module-scope source code which every session starts with, so that the prompt is
# usable without ceremony. Literals (list, tuple, dict, ...) do not need an import:
# the REPL loads the corresponding module implicitly for each input, like for normal
# modules.
DEFAULT_PRELUDE = (
    "from __spy__ import COLOR, as_red, interp_list, interp_dict, interp_tuple\n"
)

EXIT_COMMANDS = ("exit", "exit()", "quit", "quit()")

MODULE_PROMPT = "module>>> "
MAIN_PROMPT = ">>> "
CONT_PROMPT = "... "


def _ensure_nl(src: str) -> str:
    if src and not src.endswith("\n"):
        return src + "\n"
    return src


class _ReplImporter(ImportAnalyzer):
    """
    Import the prelude as a module, but with the repl main appended to it.

    By overriding astcompile_one we get, for free, the handling of everything which
    the prelude imports (explicit and implicit imports, import order, errors, .spyc
    cache of the dependencies, etc.)
    """

    def __init__(self, session: "ReplSession", use_spyc: bool) -> None:
        super().__init__(session.vm, session.modname, use_spyc=use_spyc)
        self.session = session

    def astcompile_one(self, modname: str, spyfile: py.path.local) -> ast.Module:
        if modname == self.session.modname:
            return self.session.compile([])
        return super().astcompile_one(modname, spyfile)

    def find_file_on_path(self, modname: str) -> Optional[py.path.local]:
        res = super().find_file_on_path(modname)
        if res is None and modname == self.session.modname:
            # the root module of the session might not exist as a file (e.g. `ispy`
            # without arguments): its source is the prelude, which astcompile_one
            # compiles directly
            res = py.path.local(f"{modname}.spy")
        return res


class ReplSession:
    """
    A REPL session: see the module docstring.

    This class does no I/O, except for the `print`s done by the SPy code which is
    executed: see `run_repl` for the interactive loop.
    """

    vm: SPyVM
    filename: str
    modname: str
    prelude: str  # module-scope source code of the session
    mode: str  # "module" or "main"
    use_spyc: bool
    inputs: list[str]  # main-scope inputs accepted so far
    n_done: int  # number of compiled stmts, in the repl main, already executed
    frame: Optional[ASTFrame]  # the frame of the repl main, in main scope only

    def __init__(
        self,
        vm: SPyVM,
        filename: Optional[Path],
        *,
        module_scope: bool = False,
        use_spyc: bool = True,
    ) -> None:
        self.vm = vm
        if filename is not None:
            self.filename = str(filename)
            self.modname = filename.stem
            file_src = filename.read_text()
        else:
            self.filename = "<repl>"
            self.modname = "repl"
            file_src = ""
        self.use_spyc = use_spyc
        self.prelude = DEFAULT_PRELUDE + _ensure_nl(file_src)
        self.mode = "module"
        self.inputs = []
        self.n_done = 0
        self.frame = None

        if self.modname in vm.modules_w:
            # e.g. a file called `math.spy`: ImportAnalyzer would consider it as
            # already imported and never compile it
            raise SPyError(
                "W_ImportError",
                f"cannot start the REPL on `{self.filename}`: "
                f"a module called `{self.modname}` already exists, please rename it",
            )
        if not module_scope:
            self.enter_main()

    def _module_src(self, extra: str = "") -> str:
        """
        The whole module-scope source: the prelude, plus `extra`.
        """
        src = self.prelude
        if extra:
            src = _ensure_nl(src) + _ensure_nl(extra)
        return src

    def _scope_analyze(self, src: str) -> tuple[ast.Module, ScopeAnalyzer]:
        """
        Parse and scope-analyze `src`, which is the whole module source (prelude +
        inputs). Register it in linecache so that error messages can show the
        source lines.
        """
        # An entry without mtime is never invalidated by linecache.checkcache
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
        return parsed, sa

    def compile(self, inputs: list[str]) -> ast.Module:
        """
        Parse, analyze and astcompile the prelude + the repl main containing `inputs`.
        """
        if inputs:
            body = textwrap.indent("\n".join(inputs), "    ") + "\n"
        else:
            body = "    pass\n"
        src = self._module_src(f"\n\ndef {REPL_MAIN}() -> None:\n" + body)
        parsed, sa = self._scope_analyze(src)
        return astcompile(parsed, scope_analyzer=sa)

    def add_module_input(self, src: str) -> None:
        """
        Check a module-scope input and append it to the prelude.

        The input is parsed and scope-analyzed together with the prelude, so that
        syntax and scoping errors are reported immediately. Nothing is executed:
        the prelude is only imported by enter_main(). If the check fails, the
        prelude is left unchanged.
        """
        assert self.mode == "module"
        checked = self._module_src(src)
        self._scope_analyze(checked)
        self.prelude = checked

    def load(self, filename: str) -> None:
        """
        `%load FILE`: append the content of FILE to the prelude.
        """
        self.add_module_input(py.path.local(filename).read_text("utf-8"))

    def enter_main(self) -> None:
        """
        Freeze the prelude and switch to main scope: import the prelude as a normal
        module and create the frame of the repl main, which then stays alive for
        the rest of the session.
        """
        assert self.mode == "module"
        importer = _ReplImporter(self, self.use_spyc)
        importer.astcompile_all()
        importer.import_all()

        w_mod = self.vm.modules_w[self.modname]
        w_main = w_mod.getattr_maybe(self.vm, REPL_MAIN)
        assert isinstance(w_main, W_ASTFunc)
        self.frame = ASTFrame(self.vm, w_main, [])
        self.frame.declare_arguments()
        self.mode = "main"

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
        Execute one main-scope input, which can contain multiple statements.

        If the input ends with an expression whose value is not None, return its
        repr. Raise SPyError in case of errors.

        An input is atomic as far as the local variables are concerned: if it fails,
        the variables are restored to what they were before. Side effects on the
        objects are not undone (e.g. a list which was appended to).
        """
        assert self.mode == "main" and self.frame is not None
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


def run_command(
    session: ReplSession,
    line: str,
    *,
    out: IO[str],
    use_colors: bool,
) -> None:
    """
    Execute a REPL command (a line starting with `%`), e.g. `%main` or `%load FILE`.
    """
    parts = line.split()
    cmd, args = parts[0], parts[1:]
    if cmd in ("%main", "%load") and session.mode == "main":
        print(f"`{cmd}` is only available in module scope", file=out)
    elif cmd == "%main":
        try:
            session.enter_main()
        except SPyError as e:
            print(e.format(use_colors=use_colors), file=out)
    elif cmd == "%load":
        if len(args) != 1:
            print("usage: %load FILE", file=out)
        else:
            try:
                session.load(args[0])
            except SPyError as e:
                print(e.format(use_colors=use_colors), file=out)
            except OSError as e:
                print(f"cannot load `{args[0]}`: {e.strerror}", file=out)
    else:
        print(f"unknown command `{cmd}`", file=out)


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
            if buf:
                prompt = CONT_PROMPT
            elif session.mode == "module":
                prompt = MODULE_PROMPT
            else:
                prompt = MAIN_PROMPT
            line = read_line(prompt)
        except EOFError:
            print(file=out)
            return
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt", file=out)
            buf.clear()
            continue

        if not buf and line.strip() in EXIT_COMMANDS:
            return
        if not buf and line.startswith("%"):
            run_command(session, line.strip(), out=out, use_colors=use_colors)
            continue
        buf.append(line)
        src = "\n".join(buf)
        if not src.strip():
            buf.clear()
            continue
        if not is_complete(src):
            continue

        buf.clear()
        try:
            if session.mode == "module":
                session.add_module_input(src)
                res = None
            else:
                res = session.run(src)
        except SPyError as e:
            print(e.format(use_colors=use_colors), file=out)
        else:
            if res is not None:
                print(res, file=out)
