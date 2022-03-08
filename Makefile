install-deps:
	@python -m pip install -r requirements.txt

install-dev-deps:
	@python -m pip install -r requirements-dev.txt

clean:
	@rm -rf build dist .eggs *.egg-info
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +

black: clean
	@isort --profile black scripts/
	@black scripts/

lint:
	@mypy scripts/ --show-error-codes
