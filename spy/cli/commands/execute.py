from dataclasses import dataclass
from typing import Annotated

from typer import Option

from spy.analyze.importing import ImportAnalyzer
from spy.cli._runners import execute_spy_main, init_vm
from spy.cli.commands.shared_args import (
    Base_Args,
    Filename_Required_Args,
    _execute_options,
)
from spy.repl import ReplSession, run_repl


@dataclass
class _interactive_flag:
    interactive: Annotated[
        bool,
        Option(
            "-i",
            "--interactive",
            help="After running main(), start an interactive session (REPL)",
        ),
    ] = False


@dataclass
class Execute_Args(
    Base_Args, _interactive_flag, _execute_options, Filename_Required_Args
): ...


async def execute(args: Execute_Args) -> None:
    """Execute the file in the vm (default)"""
    modname = args.filename.stem
    vm = await init_vm(args)

    if args.interactive:
        # `spy -i FILE [ARGS...]`: same as `ispy -x FILE [ARGS...]`
        session = ReplSession(
            vm,
            args.filename,
            use_spyc=not args.no_spyc,
            run_main=args.argv or [],
        )
        run_repl(session)
        return

    importer = ImportAnalyzer(vm, modname, use_spyc=not args.no_spyc)
    importer.astcompile_all()
    importer.import_all()
    w_mod = vm.modules_w[modname]

    argv: list[str] = [str(args.filename)] + (args.argv or [])
    execute_spy_main(vm, w_mod, argv, redshift=False, _timeit=args.timeit)
