from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import click
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
    execute: Annotated[
        bool,
        Option(
            "-x",
            "--execute",
            help="Run main() of FILE first, then start the session in its frame",
        ),
    ] = False
    argv: Annotated[
        Optional[list[str]],
        Argument(help="Arguments passed to main() when -x is used"),
    ] = None


async def repl(args: Repl_Args) -> None:
    """Start an interactive session (a REPL)"""
    if args.execute and args.filename is None:
        raise click.UsageError("-x/--execute requires a FILE")
    if args.argv is not None and not args.execute:
        raise click.UsageError("ARGS are only passed to main() with -x/--execute")
    vm = await init_vm(args)
    session = ReplSession(
        vm,
        args.filename,
        # without a FILE, the session starts in module scope, with the default
        # prelude only; -x implies main scope
        module_scope=not args.execute and (args.module_scope or args.filename is None),
        use_spyc=not args.no_spyc,
        run_main=(args.argv or []) if args.execute else None,
    )
    run_repl(session)
