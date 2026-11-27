from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

from typer import Argument, Option

from spy.cli._runners import init_vm
from spy.cli.commands.shared_args import Base_Args, filename_optional_callback
from spy.repl import ReplSession, run_repl


@dataclass
class Repl_Args(Base_Args):
    filename: Annotated[
        Optional[Path],
        Argument(
            help="Module-scope code to start the session with",
            show_default=False,
            callback=filename_optional_callback,
        ),
    ] = None
    module_scope: Annotated[
        bool,
        Option(
            "-m",
            "--module-scope",
            help="Start in module scope instead of main scope",
        ),
    ] = False


async def repl(args: Repl_Args) -> None:
    """Start an interactive session (a REPL)"""
    vm = await init_vm(args)
    session = ReplSession(
        vm,
        args.filename,
        # without a FILE, the session starts in module scope, with the default
        # prelude only
        module_scope=args.module_scope or args.filename is None,
        use_spyc=not args.no_spyc,
    )
    run_repl(session)
