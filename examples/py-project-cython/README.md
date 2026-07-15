# Example of Python package using SPy

The example demonstrates how one can create Python extensions in a
Python project from SPy files with the (to be created) Cython backend.

Note: pyproject.toml, Meson and Cython are used. No setup.py!

We cannot yet use simple install commands using isolated build. However,
we mimic an isolated build with commands in the Makefile.
