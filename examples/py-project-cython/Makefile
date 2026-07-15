
help:
	@echo This should work:
	@echo make clean create-build-env build install test

clean:
	rm -rf .venv-build
	rm -rf dist

create-build-env:
	uv venv .venv-build -p 3.12 --clear
	VIRTUAL_ENV=.venv-build uv pip install -e ../..
	VIRTUAL_ENV=.venv-build uv pip install -r requirements-build.txt

.PHONY: build
build:
	VIRTUAL_ENV=.venv-build uv run --no-project python -m build --no-isolation

install:
	uv venv .venv -p 3.12 --clear
	VIRTUAL_ENV=.venv uv pip install dist/my_package-0.1.0*.whl
	VIRTUAL_ENV=.venv uv pip install pytest

.PHONY: test
test:
	.venv/bin/pytest test_my_package.py
