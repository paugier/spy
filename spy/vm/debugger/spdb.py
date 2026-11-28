"""
The SPy debugger ("spy pdb")
"""

import cmd
import pdb
import sys
from typing import IO, TYPE_CHECKING, Annotated, Any, Literal, Optional

from spy import ast
from spy.analyze.sym import Scope
from spy.astcompile import astcompile_interactive
from spy.doppler import DopplerFrame
from spy.errfmt import ErrorFormatter
from spy.errors import SPyError
from spy.location import Loc
from spy.parser import Parser
from spy.textbuilder import ColorFormatter
from spy.util import record_src_in_linecache
from spy.vm.astframe import AbstractFrame, ASTFrame
from spy.vm.b import BUILTINS
from spy.vm.classframe import ClassFrame
from spy.vm.debugger.introspect import Introspector, is_help_request
from spy.vm.debugger.longlist import print_longlist
from spy.vm.exc import TBEntry, W_Traceback
from spy.vm.modframe import ModFrame
from spy.vm.modules.operator import OP
from spy.vm.opspec import W_MetaArg
from spy.vm.w import W_Object

if TYPE_CHECKING:
    from spy.vm.vm import SPyVM

FILE = Optional[IO[str]]


def readline_line_buffer() -> str:
    try:
        import readline
    except ImportError:
        return ""
    return readline.get_line_buffer()


@BUILTINS.builtin_func
def w_breakpoint(vm: "SPyVM") -> None:
    spdb = make_spdb(vm)
    spdb.interaction()


def make_spdb(vm: "SPyVM", **kwargs: Any) -> "SPdb":
    # generate a fake traceback
    pyframe = sys._getframe().f_back.f_back  # type: ignore
    assert pyframe is not None
    w_tb = W_Traceback.from_py_frame(pyframe)
    spdb = SPdb(vm, w_tb, **kwargs)
    return spdb


def print_wam(
    vm: "SPyVM", wam_arg: W_MetaArg, *, file: FILE = None, use_colors: bool = True
) -> None:
    if file is None:
        file = sys.stdout
    w_T = vm.dynamic_type(wam_arg.w_val)
    w_static_T = wam_arg.w_static_T
    wam_s = vm.repr_wam(wam_arg, loc=Loc.here())
    s = vm.unwrap_str(wam_s.w_val)
    #
    color = ColorFormatter(use_colors=use_colors)
    T = w_T.fqn.human_name(vm)
    static_T = w_static_T.fqn.human_name(vm)
    if w_T is w_static_T:
        print(color.set("green", "type:"), T, file=file)
    else:
        print(color.set("green", "static type: "), static_T, file=file)
        print(color.set("green", "dynamic type:"), T, file=file)
    print(s, file=file)


class SPdb(cmd.Cmd):
    prompt = "(spdb🥸) "

    def __init__(
        self,
        vm: "SPyVM",
        w_tb: W_Traceback,
        *,
        stdin: FILE = None,
        stdout: FILE = None,
        use_colors: bool = True,
    ) -> None:
        super().__init__(stdin=stdin, stdout=stdout)
        if stdin is not None:
            self.use_rawinput = False
        self.use_colors = use_colors
        self.vm = vm
        self.w_tb = w_tb
        self.curindex = -1  # currently selected frame
        self.completion_matches: list[str] = []

    def interaction(self) -> None:
        print("--- entering applevel debugger ---", file=self.stdout)
        last = len(self.w_tb.entries) - 1
        self.select_frame(last)
        self.cmdloop()

    def post_mortem(self) -> None:
        print("--- entering applevel debugger (post-mortem) ---", file=self.stdout)
        last = len(self.w_tb.entries) - 1
        self.select_frame(last)
        try:
            self.cmdloop()
        except SPyError as e:
            if e.etype == "W_SPdbQuit":
                return
            raise

    def select_frame(self, i: int) -> None:
        assert 0 <= i < len(self.w_tb.entries)
        if self.curindex != i:
            self.curindex = i
            self.print_frame_info(i)

    def get_curframe(self) -> TBEntry:
        return self.w_tb.entries[self.curindex]

    def print_frame_info(self, i: int) -> None:
        f = self.w_tb.entries[i]
        errfmt = ErrorFormatter(use_colors=self.use_colors)
        errfmt.emit_frameinfo(f, index=i)
        print(errfmt.build(), end="", file=self.stdout)

    def error(self, msg: str) -> None:
        print("***", msg, file=self.stdout)

    def default(self, arg: str) -> None:
        return self.do_print(arg)

    def do_quit(self, arg: str) -> None:
        raise SPyError("W_SPdbQuit", "")

    do_q = do_quit

    def do_EOF(self, arg: str) -> bool:
        print("", file=self.stdout)
        return True

    def do_continue(self, arg: str) -> bool:
        return True

    do_c = do_continue
    do_cont = do_continue

    def do_where(self, arg: str) -> None:
        """w(here)

        Print a stack trace, with the most recent frame at the bottom.
        """
        errfmt = ErrorFormatter(use_colors=self.use_colors)
        for i, f in enumerate(self.w_tb.entries):
            errfmt.emit_frameinfo(f, index=i, is_current=(i == self.curindex))
        print(errfmt.build(), end="", file=self.stdout)

    do_w = do_where
    do_bt = do_where

    def do_up(self, arg: str) -> None:
        """u(p) [count]

        Move the current frame count (default one) levels up in the
        stack trace (to an older frame).
        """
        if self.curindex == 0:
            self.error("Oldest frame")
            return
        try:
            count = int(arg or 1)
        except ValueError:
            self.error("Invalid frame count (%s)" % arg)
            return
        if count < 0:
            newframe = 0
        else:
            newframe = max(0, self.curindex - count)
        self.select_frame(newframe)

    do_u = do_up

    def do_down(self, arg: str) -> None:
        """d(own) [count]

        Move the current frame count (default one) levels down in the
        stack trace (to a newer frame).
        """
        if self.curindex + 1 == len(self.w_tb.entries):
            self.error("Newest frame")
            return
        try:
            count = int(arg or 1)
        except ValueError:
            self.error("Invalid frame count (%s)" % arg)
            return
        if count < 0:
            newframe = len(self.w_tb.entries) - 1
        else:
            newframe = min(len(self.w_tb.entries) - 1, self.curindex + count)
        self.select_frame(newframe)

    do_d = do_down

    def do_longlist(self, arg: str) -> None:
        """l | ll | list | longlist

        List the whole source code for the current function or frame.
        """
        f = self.get_curframe()
        if f.kind in ("astframe", "dopplerframe"):
            spyframe = f.spyframe
            assert isinstance(spyframe, ASTFrame)
            # Get the function location and current location
            func_loc = spyframe.loc
            cur_loc = f.loc
            print_longlist(
                func_loc, cur_loc, use_colors=self.use_colors, file=self.stdout
            )
        else:
            print(f"longlist: unsupported kind: {f.kind}")

    do_list = do_longlist
    do_l = do_longlist
    do_ll = do_longlist

    def do_print(self, arg: str) -> None:
        try:
            # eval "arg" in the current frame
            filename = record_src_in_linecache(arg, name="spdb-eval")
            parser = Parser(arg, filename)
            stmt = parser.parse_single_stmt()
            if not isinstance(stmt, ast.StmtExpr):
                clsname = stmt.__class__.__name__
                raise SPyError.simple(
                    "W_WIP",
                    f"not supported by SPdb: {clsname}",
                    "this is not supported",
                    stmt.loc,
                )

            f = self.get_curframe()
            # the parser produces a "parsed"-stage expression, but the frame can only
            # evaluate astcompiled nodes: run the astcompile pass on the fly, resolving
            # names against the scope which is/was active in th frame
            expr = astcompile_interactive(stmt.value, f.scope)
            wam = f.spyframe.eval_expr(expr)
            print_wam(self.vm, wam, file=self.stdout, use_colors=self.use_colors)

        except SPyError as e:
            etype = e.etype[2:]
            message = e.w_exc.message
            self.error(f"{etype}: {message}")

    do_p = do_print

    # === tab completion and ? help ===
    #
    # Same behavior as in the REPL (see spy.vm.debugger.introspect), but on the
    # currently selected frame.

    def introspector(self) -> Introspector:
        f = self.get_curframe()
        return Introspector(self.vm, f.spyframe, f.fqn.modname)

    def onecmd(self, line: str) -> bool:
        # `?name` and `name?` must be intercepted before cmd.Cmd, which would turn
        # `?` into `help`. We don't define a `do_inspect` command, so that it does not
        # hide a variable called `inspect`.
        stripped = line.strip()
        if stripped.startswith("?") or is_help_request(stripped):
            name = stripped.lstrip("?").rstrip("?").strip()
            self.print_help(name)
            return False
        return super().onecmd(line)

    def print_help(self, name: str) -> None:
        if not name:
            print("usage: ?NAME or NAME?", file=self.stdout)
            return
        try:
            print(self.introspector().inspect(name), file=self.stdout)
        except SPyError as e:
            self.error(f"{e.etype[2:]}: {e.w_exc.message}")

    def complete(self, text: str, state: int) -> Optional[str]:  # type: ignore[override]
        """
        Called by readline: complete the commands (at the beginning of the line) and
        the names visible in the current frame.
        """
        if state == 0:
            try:
                names = self.introspector().complete(text)
            except Exception:
                # readline must never die because of a completion error
                names = []
            line = readline_line_buffer()
            if line.lstrip() == text:  # we are completing the first word
                cmds = {
                    n[3:]
                    for n in self.get_names()
                    if n.startswith("do_") and n != "do_EOF"
                }
                names = sorted(set(names) | {c for c in cmds if c.startswith(text)})
            self.completion_matches = names
        if 0 <= state < len(self.completion_matches):
            return self.completion_matches[state]
        return None
