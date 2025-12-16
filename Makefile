init:
	pip install -r requirements.txt

lint:
	ruff check --fix

format:
	ruff format

test:
	pytest