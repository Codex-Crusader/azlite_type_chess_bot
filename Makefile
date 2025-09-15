.PHONY: venv deps selfplay train play lint test clean

venv:
	python3 -m venv venv
	. venv/bin/activate && python -m pip install --upgrade pip

deps: venv
	. venv/bin/activate && pip install -r requirements.txt

selfplay: deps
	. venv/bin/activate && python azlite_portfolio_clean.py selfplay --episodes 5 --sims 80 --pid demo

train: deps
	. venv/bin/activate && python azlite_portfolio_clean.py train

play: deps
	. venv/bin/activate && python azlite_portfolio_clean.py play --sims 200

lint: deps
	. venv/bin/activate && flake8 azlite_portfolio_clean.py tests

test: deps
	. venv/bin/activate && pytest -q

clean:
	rm -rf venv selfplay_data az_checkpoints
