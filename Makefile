.PHONY: install data train experiment api app test lint docker

install:
	pip install -r requirements-dev.txt

data:
	python -m neo_sentinel download

train: data
	python -m neo_sentinel train

experiment: data
	python -m neo_sentinel experiment

api:
	uvicorn neo_sentinel.api.app:app --reload

app:
	streamlit run streamlit_app.py

test:
	pytest

lint:
	ruff check . && ruff format --check .

docker:
	docker compose up --build
