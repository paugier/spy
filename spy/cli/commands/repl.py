from dataclasses import dataclass

from spy.cli._runners import init_vm
from spy.cli.commands.shared_args import Base_Args, Filename_Required_Args
from spy.repl import ReplSession, run_repl


@dataclass
class Repl_Args(Base_Args, Filename_Required_Args): ...


async def repl(args: Repl_Args) -> None:
    """Start an interactive session on top of the module-scope code in FILENAME"""
    vm = await init_vm(args)
    session = ReplSession(vm, args.filename, use_spyc=not args.no_spyc)
    run_repl(session)
