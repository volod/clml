.PHONY: venv setenv data explore run run-all runs mlflow list test lint

venv:
	./scripts/venv.sh

setenv:
	./scripts/setenv.sh

data:
	./scripts/data.sh

explore:
	uv run clml dataexplore --dataset credit_risk

run:
	uv run clml run --method random_forest_classifier

run-all:
	uv run clml run-all --trials 0

runs:
	uv run clml runs list

mlflow:
	uv run mlflow ui --backend-store-uri $$(uv run python -c "from clml.config.settings import get_settings; print(get_settings().mlflow_tracking_uri)") --host 127.0.0.1 --port 5000

list:
	uv run clml list-methods

test:
	uv run pytest

lint:
	uv run ruff check .
