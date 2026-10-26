"""
PROTOTYPE: prebuild some stdlib modules into a wasm32-wasi .a archive, so that
the interpreter can call the compiled versions instead of interpreting them.

The archive is meant to be linked together with libspy.a into the VM's WASM
instance (see spy.libspy.get_LLMOD), so that interpreted and compiled code
share the same linear memory.

The manifest below is hand-written for now. The long-term plan is to derive
it from the redshifted modules.
"""

import hashlib
import shlex
import subprocess
import sys
from dataclasses import dataclass

import py.path

import spy
from spy.build.wasm_bundle import BUNDLE_CACHE_DIR, _zig_version

STDLIB = spy.ROOT.join("..", "stdlib")
LIBSPY_INCLUDE = spy.ROOT.join("libspy", "include")

# SPy type name -> C type, for the generated export shims
C_TYPES = {
    "None": "void",
    "u32": "uint32_t",
    "i32": "int32_t",
    "f64": "double",
    "f32": "float",
    "bool": "bool",
}


@dataclass
class NativeFunc:
    fqn: str  # e.g. "random::seed"
    c_symbol: str  # the function emitted by the C backend
    params: list[str]  # SPy type names
    restype: str

    @property
    def export_name(self) -> str:
        return "native__" + self.fqn.replace("::", "__")


@dataclass
class NativeCell:
    fqn: str  # e.g. "_random::_ptr_global_random"
    c_symbol: str  # the C global emitted by the C backend
    header: str

    @property
    def export_name(self) -> str:
        return "native_cell__" + self.fqn.replace("::", "__")


FUNCS = [
    NativeFunc("random::seed", "spy_random$seed", ["u32"], "None"),
    NativeFunc("random::random", "spy_random$random", [], "f64"),
    NativeFunc("random::uniform", "spy_random$uniform", ["f64", "f64"], "f64"),
]
CELLS = [
    NativeCell(
        "_random::_ptr_global_random", "spy__random$_ptr_global_random", "_random.h"
    ),
]
MODULES = ["random"]

STUB = "from random import seed, random, uniform\n\ndef main() -> None:\n    pass\n"


def _gen_shim() -> str:
    lines = [
        '#include "spy.h"',
        '#include "random.h"',
        '#include "_random.h"',
        "",
    ]
    for f in FUNCS:
        ret = C_TYPES[f.restype]
        params = ", ".join(f"{C_TYPES[t]} a{i}" for i, t in enumerate(f.params))
        args = ", ".join(f"a{i}" for i in range(len(f.params)))
        ret_kw = "" if f.restype == "None" else "return "
        lines.append(
            f"{ret} WASM_EXPORT({f.export_name})({params or 'void'}) "
            f"{{ {ret_kw}{f.c_symbol}({args}); }}"
        )
    for c in CELLS:
        # a data symbol cannot be exported portably, so export its address
        lines.append(
            f"void *WASM_EXPORT({c.export_name})(void) {{ return &{c.c_symbol}; }}"
        )
    return "\n".join(lines) + "\n"


def _cache_key(shim: str) -> str:
    h = hashlib.sha256()
    for f in sorted(STDLIB.listdir("*.spy")):
        h.update(f.read_binary())
    for f in sorted(LIBSPY_INCLUDE.visit("*.h")):
        h.update(f.read_binary())
    h.update(shim.encode())
    h.update(_zig_version().encode())
    h.update(repr((FUNCS, CELLS)).encode())
    return h.hexdigest()


def _cflags() -> list[str]:
    out = subprocess.run(
        [
            sys.executable, "-m", "spy.build.flags", "--cflags",
            "--target=wasi", "--build-type=debug", "--output-kind=testlib",
        ],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return shlex.split(out)


def build_archive(out_dir: py.path.local, shim: str) -> py.path.local:
    # 1. generate C for a program importing the modules
    stub = out_dir.join("stub.spy")
    stub.write(STUB)
    gen = out_dir.join("gen")
    subprocess.run(
        [
            sys.executable, "-m", "spy", "build", "--no-compile",
            "-t", "wasi", "-b", str(gen), str(stub),
        ],
        check=True, capture_output=True,
    )  # fmt: skip
    src = gen.join("src")
    shim_c = src.join("native_shim.c")
    shim_c.write(shim)

    # 2. compile everything but the stub's own module (which has main())
    cflags = _cflags() + [f"-I{src}"]
    cfiles = [c for c in sorted(src.listdir("*.c")) if c.basename != "stub.c"]
    objs = []
    procs = []
    for c in cfiles:
        o = c.new(ext=".o")
        objs.append(o)
        cmd = [
            sys.executable,
            "-m",
            "ziglang",
            "cc",
            *cflags,
            "-c",
            str(c),
            "-o",
            str(o),
        ]
        procs.append((c, subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)))
    for c, p in procs:
        _, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f"cannot compile {c}:\n{err}")

    # 3. archive
    archive = out_dir.join("libspy_stdlib.a")
    subprocess.run(
        [sys.executable, "-m", "ziglang", "ar", "rcs", str(archive)]
        + [str(o) for o in objs],
        check=True,
    )
    return archive


def get_archive() -> py.path.local:
    """
    Return the (cached) archive with the prebuilt stdlib modules.
    """
    shim = _gen_shim()
    out_dir = BUNDLE_CACHE_DIR.join("native-stdlib", _cache_key(shim))
    archive = out_dir.join("libspy_stdlib.a")
    if not archive.check(file=True):
        out_dir.ensure(dir=True)
        build_archive(out_dir, shim)
    return archive
