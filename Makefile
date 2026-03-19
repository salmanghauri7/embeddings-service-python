.PHONY: run build install

# Variables
APP_NAME=embeddings-service-python
PORT=8000

run:
	uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

build:
	docker build -t $(APP_NAME) .

install:
	pip install -r requirements.txt

background-worker:
	arq app.worker.WorkerSettings
