.PHONY: install lint test run format

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

lint:
	ruff check .
	black --check .
	mypy src

format:
	black .
	ruff check --fix .

test:
	pytest -q

run:
	python -m fpl_opt.cli
