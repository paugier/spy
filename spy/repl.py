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
commonly needs. `%reset` forgets everything and restarts in module scope with
the default prelude.

With `-x` (see ReplSession's `run_main`), the session starts in the live frame
of the `main` function of FILE, after running it: the local variables of
`main` are available at the prompt, as in a debugger after a post-mortem.

How main scope works: one `ASTFrame` for the repl main is created when the
session switches to main scope and lives as long as the session. We don't
touch ScopeAnalyzer: for each input, we run the normal
parse/scope-analysis/astcompile pipeline on the *whole* source (the prelude,
plus the repl main whose body is made of all the inputs accepted so far, plus
the new input). This is simple and gives the correct scoping for free. Then we
execute in the live frame only the statements which are new. In `-x` mode the
inputs are appended to the body of `main` instead, and executed in its frame.

When the session is based on a FILE, the locs of the inputs are rewritten to
point to the pseudo-file `<repl>`, with line numbers relative to the inputs
themselves: tracebacks and error messages about the inputs show `<repl>`,
not the FILE and the line numbers of the combined source.

The interactive loop (run_repl) uses readline when available: line editing,
a history file (HISTORY_FILE, like ~/.python_history for python) and tab
completion (see ReplSession.complete). Lines starting with `?` (or ending
with `?`) print help about a name: see ReplSession.inspect.
"""

import codeop
import copy
import dataclasses
import linecache
import sys
import textwrap
from pathlib import Path
from typing import IO, Any, Callable, Optional

import py.path

from spy import ast
from spy.analyze.importing import ImportAnalyzer
from spy.analyze.scope import ScopeAnalyzer
from spy.astcompile import astcompile
from spy.errors import SPyError
from spy.location import Loc
from spy.parser import Parser
from spy.util import OrderedSet, record_src_in_linecache
from spy.vm.astframe import ASTFrame
from spy.vm.b import B
from spy.vm.function import W_ASTFunc, W_BuiltinFunc, W_Func
from spy.vm.module import W_Module
from spy.vm.object import W_Object, W_Type
from spy.vm.vm import SPyVM

# Name of the function whose frame is used by the session. FILE is free to define its
# own `main`, which the user can then call from the REPL.
REPL_MAIN = "__repl_main__"

# Pseudo-filename shown in tracebacks and error messages for the inputs typed
# at the prompt (as opposed to the prelude, which keeps its real filename).
REPL_FILENAME = "<repl>"

# Default history file of `ispy`.
HISTORY_FILE = Path.home() / ".ispy_history"

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


def _matches(name: str, prefix: str) -> bool:
    """
    Whether `name` is a completion candidate for `prefix`. Names starting with
    `_` are proposed only if the user explicitly typed a leading `_`.
    """
    if not name.startswith(prefix):
        return False
    return prefix.startswith("_") or not name.startswith("_")


def _shifted_loc(loc: Loc, filename: str, line_offset: int) -> Loc:
    return Loc(
        filename,
        loc.line_start - line_offset,
        loc.line_end - line_offset,
        loc.col_start,
        loc.col_end,
        loc.colorize,
    )


def _rewrite_locs(node: ast.Node, filename: str, line_offset: int) -> ast.Node:
    """
    Return a copy of `node` and its descendants in which every loc points to
    `filename`, with line numbers shifted by `line_offset`.
    """
    changes: dict[str, Any] = {"loc": _shifted_loc(node.loc, filename, line_offset)}
    for f in dataclasses.fields(node):
        if f.name == "loc":
            continue
        value = getattr(node, f.name)
        if isinstance(value, ast.Node):
            changes[f.name] = _rewrite_locs(value, filename, line_offset)
        elif isinstance(value, list):
            changes[f.name] = [
                _rewrite_locs(item, filename, line_offset)
                if isinstance(item, ast.Node)
                else item
                for item in value
            ]
    return node.replace(**changes)


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
    frame: Optional[ASTFrame]  # the live frame, in main scope only
    run_main: Optional[
        list[str]
    ]  # with -x: ARGS passed to main(), FILE comes first in argv
    main_error: Optional[SPyError]  # with -x: the error raised by main(), if any
    main_exit_code: Optional[int]  # with -x: the exit code returned by main(), if any

    def __init__(
        self,
        vm: SPyVM,
        filename: Optional[Path],
        *,
        module_scope: bool = False,
        use_spyc: bool = True,
        run_main: Optional[list[str]] = None,
    ) -> None:
        self.vm = vm
        if filename is not None:
            self.filename = str(filename)
            self.modname = filename.stem
            file_src = filename.read_text()
        else:
            self.filename = REPL_FILENAME
            self.modname = "repl"
            file_src = ""
        self.use_spyc = use_spyc
        self.run_main = run_main
        self.main_error = None
        self.main_exit_code = None
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
        if run_main is not None:
            # -x implies main scope: main() is run by enter_main()
            module_scope = False
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

    def _parse(self, src: str) -> ast.Module:
        """
        Parse `src`, which is the whole module source (prelude + inputs).
        Register it in linecache so that error messages can show the source
        lines.
        """
        # An entry without mtime is never invalidated by linecache.checkcache
        linecache.cache[self.filename] = (
            len(src),
            None,
            src.splitlines(keepends=True),
            self.filename,
        )
        return Parser(src, self.filename).parse()

    def _scope_analyze(self, src: str) -> tuple[ast.Module, ScopeAnalyzer]:
        """
        Parse and scope-analyze `src`, which is the whole module source (prelude +
        inputs).
        """
        parsed = self._parse(src)
        assert parsed.scoping_rules in ("strict", "pythonic")
        sa = ScopeAnalyzer(self.modname, parsed)
        sa.analyze()
        return parsed, sa

    def compile(self, inputs: list[str]) -> ast.Module:
        """
        Parse, analyze and astcompile the prelude + the repl main containing `inputs`
        (or, in -x mode, the prelude with `inputs` appended to the body of `main`).

        When the session is based on a FILE, the locs of the inputs are rewritten
        to point to the pseudo-file `<repl>` (see _localize_inputs), so that
        tracebacks and error messages about the inputs show `<repl>` instead of
        the FILE and the line numbers of the combined source.
        """
        if self.run_main is not None:
            src, offset, n_orig = self._main_src(inputs)
        else:
            src, offset, n_orig = self._repl_main_src(inputs)
        parsed = self._parse(src)
        if inputs and self.filename != REPL_FILENAME:
            parsed = self._localize_inputs(parsed, offset, n_orig)
            self._record_repl_src(inputs)
        assert parsed.scoping_rules in ("strict", "pythonic")
        sa = ScopeAnalyzer(self.modname, parsed)
        sa.analyze()
        return astcompile(parsed, scope_analyzer=sa)

    def _inputs_body(self, inputs: list[str]) -> str:
        """
        The source of the inputs, indented to fit in the body of a function.
        """
        return "".join(textwrap.indent(_ensure_nl(src), "    ") for src in inputs)

    def _repl_main_src(self, inputs: list[str]) -> tuple[str, int, int]:
        """
        The whole module source in main mode: the prelude, plus the repl main
        whose body is made of `inputs`. Returns (src, offset, n_orig): `offset`
        is the number of lines of `src` which precede the inputs (input line 1
        is at line offset+1 of `src`), and n_orig is 0, because the whole body
        of the repl main comes from the inputs.
        """
        if inputs:
            body = self._inputs_body(inputs)
        else:
            body = "    pass\n"
        head = self._module_src(f"\n\ndef {REPL_MAIN}() -> None:\n")
        return head + body, head.count("\n"), 0

    def _main_src(self, inputs: list[str]) -> tuple[str, int, int]:
        """
        The whole module source in -x mode: the prelude, with `inputs` appended
        to the body of `main`, so that they are scope-analyzed (and then
        executed) as part of `main`, in its frame. Returns (src, offset, n_orig):
        `offset` is the number of lines of `src` which precede the appended
        inputs, and n_orig is the number of stmts of the original body of `main`.
        """
        src = self._module_src()
        if not inputs:
            return src, 0, 0
        parsed = Parser(src, self.filename).parse()
        funcdef: Optional[ast.FuncDef] = None
        for decl in parsed.decls:
            if isinstance(decl, ast.GlobalFuncDef) and decl.funcdef.name == "main":
                funcdef = decl.funcdef
                break
        if funcdef is None:
            raise SPyError("W_TypeError", f"`{self.filename}` does not define `main`")
        n_orig = len(funcdef.body.body)
        lines = src.splitlines(keepends=True)
        injected = self._inputs_body(inputs)
        body = funcdef.body.body
        first = body[0] if body else None
        if first is not None and first.loc.line_start == funcdef.loc.line_start:
            # `def main(): <stmt>` on a single line: move the body to its own
            # line, to make room for the inputs
            ln = funcdef.loc.line_start  # 1-based
            line = lines[ln - 1]
            col = first.loc.col_start
            mid = line[:col] + "\n" + textwrap.indent(line[col:], "    ") + injected
            # the injected inputs start at line ln+2
            return "".join(lines[: ln - 1]) + mid + "".join(lines[ln:]), ln + 1, n_orig
        # insert right after the last line of `main` (lines are 1-based and
        # line_end is inclusive, so the insertion index is line_end itself)
        end = funcdef.loc.line_end
        prefix = "".join(lines[:end])
        if not prefix.endswith("\n"):
            prefix += "\n"
        # the injected inputs start at line end+1
        return prefix + injected + "".join(lines[end:]), end, n_orig

    def _localize_inputs(self, mod: ast.Module, offset: int, n_orig: int) -> ast.Module:
        """
        Rewrite the locs of the inputs (the body of the repl main, or the stmts
        appended to `main` in -x mode) to point to the pseudo-file `<repl>`,
        with line numbers relative to the inputs themselves.
        """
        name = "main" if self.run_main is not None else REPL_MAIN
        decls: list[ast.Decl] = []
        for decl in mod.decls:
            if isinstance(decl, ast.GlobalFuncDef) and decl.funcdef.name == name:
                funcdef = decl.funcdef
                body = funcdef.body.body
                new_stmts = body[:n_orig] + [
                    _rewrite_locs(stmt, REPL_FILENAME, offset) for stmt in body[n_orig:]
                ]
                new_funcdef = funcdef.replace(body=funcdef.body.replace(body=new_stmts))
                decl = decl.replace(funcdef=new_funcdef)
            decls.append(decl)
        return mod.replace(decls=decls)

    def _record_repl_src(self, inputs: list[str]) -> None:
        """
        Register the source of the inputs in linecache under `<repl>`, so that
        error messages can show the corresponding source lines.
        """
        src = self._inputs_body(inputs)
        linecache.cache[REPL_FILENAME] = (
            len(src),
            None,
            src.splitlines(keepends=True),
            REPL_FILENAME,
        )

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
        module and create the frame which then stays alive for the rest of the
        session: the frame of the repl main, or of `main` in -x mode (after
        running it).
        """
        assert self.mode == "module"
        importer = _ReplImporter(self, self.use_spyc)
        importer.astcompile_all()
        importer.import_all()

        w_mod = self.vm.modules_w[self.modname]
        if self.run_main is not None:
            self._run_main_and_enter(w_mod)
            return
        w_main = w_mod.getattr_maybe(self.vm, REPL_MAIN)
        assert isinstance(w_main, W_ASTFunc)
        self.frame = ASTFrame(self.vm, w_main, [])
        self.frame.declare_arguments()
        self.mode = "main"

    def _run_main_and_enter(self, w_mod: W_Module) -> None:
        """
        -x mode: run `main` of the file first, with argv = [FILE] + ARGS, and
        start the session in its live frame: the local variables of `main` are
        available at the prompt, as in a debugger after a post-mortem. If
        `main` raises, the session still starts in its frame. A returned exit
        code does not end the session.
        """
        w_main = w_mod.getattr_maybe(self.vm, "main")
        if not isinstance(w_main, W_ASTFunc):
            raise SPyError(
                "W_TypeError", f"`{self.filename}` does not define a `main` function"
            )
        w_restype, has_args = self.vm.typecheck_main(w_main)
        run_main = self.run_main
        assert run_main is not None
        argv = [self.filename] + run_main
        args_w: list[W_Object] = []
        if has_args:
            args_w = [self.vm.wrap_list(B.w_str, argv)]
        frame = ASTFrame(self.vm, w_main, args_w)
        # after running the original body of main, the inputs are the only new
        # stmts: even if main raised in the middle, we don't re-execute the
        # stmts which follow the raise point
        self.n_done = len(frame.funcdef.body.body)
        try:
            w_res = frame.run(args_w)
        except SPyError as e:
            self.main_error = e
        else:
            if w_restype == B.w_i32:
                self.main_exit_code = self.vm.unwrap_i32(w_res)
        self.frame = frame
        self.mode = "main"

    def reset(self) -> None:
        """
        `%reset`: forget everything and restart in module scope, with the
        default prelude. A fresh module name is used, because a module which
        has already been imported cannot be redefined.
        """
        self.filename = REPL_FILENAME
        self.modname = self._fresh_modname()
        self.prelude = DEFAULT_PRELUDE
        self.mode = "module"
        self.inputs = []
        self.n_done = 0
        self.frame = None
        self.run_main = None
        self.main_error = None
        self.main_exit_code = None

    def _fresh_modname(self) -> str:
        i = 0
        while True:
            i += 1
            modname = "repl" if i == 1 else f"repl_{i}"
            if modname not in self.vm.modules_w:
                return modname

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

    def _get_frame_funcdef(self, mod: ast.Module) -> ast.FuncDef:
        """
        The funcdef executed in the live frame of the session: the repl main,
        or `main` of the file in -x mode.
        """
        name = "main" if self.run_main is not None else REPL_MAIN
        for decl in mod.decls:
            if isinstance(decl, ast.GlobalFuncDef) and decl.funcdef.name == name:
                return decl.funcdef
        assert False, f"cannot find `{name}`"

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
        funcdef = self._get_frame_funcdef(mod)
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

    # === tab completion and ? help ===

    def complete(self, prefix: str) -> list[str]:
        """
        Return the completions for `prefix`: plain names visible in the session
        (locals of the live frame, globals of the session module, builtins), or,
        if `prefix` contains a dot, the attributes of the object designated by
        the part before the last dot, e.g. `math.sq` -> `['math.sqrt']`.

        Only plain names and attribute chains are evaluated (see _eval_chain),
        so completing never has side effects.
        """
        if "." in prefix:
            base, _, attr_prefix = prefix.rpartition(".")
            w_obj = self._eval_chain(base)
            return sorted(
                f"{base}.{attr}"
                for attr in self._dir(w_obj)
                if _matches(attr, attr_prefix)
            )
        return sorted(name for name in self._visible_names() if _matches(name, prefix))

    def inspect(self, name: str) -> str:
        """
        `?name`: return a help string about `name`, which must be a plain name
        or an attribute chain (e.g. `?math.sqrt`).
        """
        w_obj = self._eval_chain(name)
        vm = self.vm
        if isinstance(w_obj, W_Func):
            w_ft = w_obj.w_functype
            if w_ft.kind == "metafunc":
                # the w_functype of a metafunc is its interp-level signature,
                # which is not what the user wants to see
                sig = f"def {name}(...)"
            else:
                # NOTE: use w_ft.params and not all_params(), which is infinite
                # in case of varargs
                params = ", ".join(
                    ("*" if p.kind == "var_positional" else "")
                    + p.w_T.fqn.human_name(vm)
                    for p in w_ft.params
                )
                ret = w_ft.w_restype.fqn.human_name(vm)
                sig = f"def {name}({params}) -> {ret}"
            lines = [sig, str(w_obj.fqn)]
            docstring: Optional[str] = None
            if isinstance(w_obj, W_ASTFunc):
                docstring = w_obj.funcdef.docstring
            elif isinstance(w_obj, W_BuiltinFunc):
                docstring = w_obj._pyfunc.__doc__
            if docstring:
                lines += ["", docstring.strip()]
            return "\n".join(lines)
        elif isinstance(w_obj, W_Module):
            return f"module {w_obj.name}"
        elif isinstance(w_obj, W_Type):
            return f"type {w_obj.fqn.human_name(vm)}"
        else:
            w_T = vm.dynamic_type(w_obj)
            return f"type: {w_T.fqn.human_name(vm)}"

    def _visible_names(self) -> set[str]:
        names: set[str] = set()
        frame = self.frame
        if frame is not None:
            for slot_name, lv in frame.locals.items():
                if lv.w_val is None:
                    continue
                sym = frame.frameinfo.lookup_maybe(slot_name)
                names.add(sym.src_name if sym is not None else slot_name)
        for modname in (self.modname, "builtins"):
            w_mod = self.vm.modules_w.get(modname)
            if w_mod is not None:
                names.update(w_mod.keys())
        return names

    def _eval_chain(self, src: str) -> W_Object:
        """
        Evaluate `src`, which must be a plain name or an attribute chain such as
        `math.sqrt`. Anything else (function calls, subscripts, operators, ...)
        is rejected: this guarantees that completing and inspecting never have
        side effects.
        """
        filename = record_src_in_linecache(src, name="repl-eval")
        stmt = Parser(src, filename).parse_single_stmt()
        node: Optional[ast.Expr] = (
            stmt.value if isinstance(stmt, ast.StmtExpr) else None
        )
        attrs: list[str] = []
        while node is not None and isinstance(node, ast.GetAttr):
            attrs.append(node.attr.value)
            node = node.value
        if not isinstance(node, ast.Name):
            raise SPyError.simple(
                "W_TypeError",
                f"cannot evaluate `{src}`",
                "only plain names and attribute chains are allowed",
                stmt.loc,
            )
        w_obj = self._lookup_name(node.id)
        if w_obj is None:
            raise SPyError("W_NameError", f"name `{node.id}` is not defined")
        for attr in reversed(attrs):
            w_attr = self._getattr_maybe(w_obj, attr)
            if w_attr is None:
                t = self.vm.dynamic_type(w_obj).fqn.human_name(self.vm)
                raise SPyError("W_TypeError", f"type `{t}` has no attribute `{attr}`")
            w_obj = w_attr
        return w_obj

    def _lookup_name(self, name: str) -> Optional[W_Object]:
        """
        Resolve a plain name roughly like the repl main would: locals of the
        live frame first, then the globals of the session module, then builtins.
        """
        frame = self.frame
        if frame is not None:
            for slot_name, lv in frame.locals.items():
                if lv.w_val is None:
                    continue
                sym = frame.frameinfo.lookup_maybe(slot_name)
                if sym is not None and sym.src_name == name:
                    return lv.w_val
        for modname in (self.modname, "builtins"):
            w_mod = self.vm.modules_w.get(modname)
            if w_mod is not None:
                w_obj = w_mod.getattr_maybe(self.vm, name)
                if w_obj is not None:
                    return w_obj
        return None

    def _getattr_maybe(self, w_obj: W_Object, attr: str) -> Optional[W_Object]:
        """
        Attribute lookup which never executes any code: modules are plain dicts,
        and for other objects we look into the dict of their type.
        """
        if isinstance(w_obj, W_Module):
            return w_obj.getattr_maybe(self.vm, attr)
        w_T = self.vm.dynamic_type(w_obj)
        w_attr = w_T.lookup(self.vm, attr)
        if w_attr is None and isinstance(w_obj, W_Type):
            w_attr = w_obj.lookup(self.vm, attr)
        return w_attr

    def _dir(self, w_obj: W_Object) -> set[str]:
        names = self.vm.dynamic_type(w_obj).spy_dir(self.vm)
        names.update(w_obj.spy_dir(self.vm))
        return names


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
    elif cmd == "%reset":
        session.reset()
    else:
        print(f"unknown command `{cmd}`", file=out)


def run_help(session: ReplSession, arg: str, *, out: IO[str]) -> None:
    """
    `?NAME` (or `NAME?`): print help about NAME. See ReplSession.inspect.
    """
    arg = arg.strip()
    if not arg:
        print("usage: ?NAME or NAME?", file=out)
        return
    try:
        print(session.inspect(arg), file=out)
    except SPyError as e:
        print(f"{e.etype[2:]}: {e.w_exc.message}", file=out)


def _make_completer(session: ReplSession) -> Callable[[str, int], Optional[str]]:
    """
    A readline completer based on session.complete.
    """
    matches: list[str] = []

    def completer(text: str, state: int) -> Optional[str]:
        nonlocal matches
        if state == 0:
            try:
                matches = session.complete(text)
            except SPyError:
                matches = []
        if 0 <= state < len(matches):
            return matches[state]
        return None

    return completer


def _read_history_file(readline: Any, history_file: Path) -> None:
    try:
        readline.read_history_file(str(history_file))
    except OSError:
        pass  # no history yet


def _write_history_file(readline: Any, history_file: Path) -> None:
    try:
        readline.write_history_file(str(history_file))
    except OSError:
        pass  # e.g. a read-only home directory: history is best-effort


def run_repl(
    session: ReplSession,
    *,
    read_line: Optional[Callable[[str], str]] = None,
    out: Optional[IO[str]] = None,
    use_colors: bool = True,
    history_file: Optional[Path] = HISTORY_FILE,
) -> None:
    """
    The interactive loop. `read_line` and `out` can be overridden for testing.

    When `read_line` is not given, the real `input()` is used, with readline
    enabled: line editing, tab completion (see ReplSession.complete) and a
    history file (`history_file`, pass None to disable it).
    """
    if out is None:
        out = sys.stdout
    readline_mod: Any = None
    if read_line is None:
        try:
            import readline  # enables line editing/history in input()
        except ImportError:
            pass
        else:
            readline_mod = readline
            # cap the history, so that the file does not grow without bounds
            readline_mod.set_history_length(1000)
            if history_file is not None:
                _read_history_file(readline_mod, history_file)
            # '.' must NOT be a word delimiter: `math.sq` is completed as a whole
            readline_mod.set_completer_delims(" \t\n()[]{}'\",:;")
            readline_mod.set_completer(_make_completer(session))
            # readline_mod.backend was introduced in Python 3.13
            backend = getattr(readline_mod, "backend", None)
            is_libedit = (
                backend == "editline"
                if backend is not None
                else "libedit" in (readline_mod.__doc__ or "")
            )
            if is_libedit:
                readline_mod.parse_and_bind("bind ^I rl_complete")
            else:
                readline_mod.parse_and_bind("tab: complete")
        read_line = input

    if session.main_error is not None:
        # with -x, main() raised: show the error, but the session still starts
        # in its frame
        print(session.main_error.format(use_colors=use_colors), file=out)

    buf: list[str] = []
    try:
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

            if readline_mod is not None and line.strip():
                readline_mod.add_history(line)

            if not buf and line.strip() in EXIT_COMMANDS:
                return
            if not buf and line.startswith("?"):
                run_help(session, line[1:], out=out)
                continue
            if (
                not buf
                and not line.startswith(("%", "?"))
                and line.rstrip().endswith("?")
            ):
                run_help(session, line.rstrip()[:-1], out=out)
                continue
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
    finally:
        if readline_mod is not None and history_file is not None:
            _write_history_file(readline_mod, history_file)
