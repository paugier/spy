"""
Name completion and `?` help, shared by the SPy REPL (spy.repl) and the SPy
debugger (spy.vm.debugger.spdb).

An `Introspector` looks at the names visible from a frame: its locals, then the
globals of the module the frame belongs to, then the builtins. It never executes any
user code: only plain names and attribute chains are evaluated (see `_eval_chain`),
so completing and inspecting never have side effects.
"""

from typing import TYPE_CHECKING, Optional

from spy import ast
from spy.errors import SPyError
from spy.parser import Parser
from spy.util import record_src_in_linecache
from spy.vm.astframe import AbstractFrame
from spy.vm.function import W_ASTFunc, W_BuiltinFunc, W_Func
from spy.vm.module import W_Module
from spy.vm.object import W_Object, W_Type

if TYPE_CHECKING:
    from spy.vm.vm import SPyVM


def _matches(name: str, prefix: str) -> bool:
    """
    Whether `name` is a completion candidate for `prefix`. Names starting with
    `_` are proposed only if the user explicitly typed a leading `_`.
    """
    if not name.startswith(prefix):
        return False
    return prefix.startswith("_") or not name.startswith("_")


def is_help_request(line: str) -> bool:
    """
    Whether `line` is a `NAME?` help request: it must end with `?`, and the
    part before the `?` must be a plain name or an attribute chain, so that
    e.g. `x = 1  # why?` is executed as normal code, not treated as help.
    """
    rline = line.rstrip()
    if not rline.endswith("?"):
        return False
    rest = rline[:-1].strip()
    return bool(rest) and all(part.isidentifier() for part in rest.split("."))


class Introspector:
    """
    `frame` is the frame whose names are visible (None if there is no live frame,
    e.g. in the module scope of the REPL); `modname` is the module whose globals are
    visible.
    """

    vm: "SPyVM"
    frame: Optional[AbstractFrame]
    modname: str

    def __init__(
        self, vm: "SPyVM", frame: Optional[AbstractFrame], modname: str
    ) -> None:
        self.vm = vm
        self.frame = frame
        self.modname = modname

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
            if not base:
                return []  # e.g. ".x": there is nothing before the dot
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
        if not src.strip():
            raise SPyError("W_TypeError", f"cannot evaluate `{src}`")
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
