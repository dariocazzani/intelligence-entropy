.PHONY: test experiments paper all clean

test:
	uv run pytest tests/ -v

experiments:
	uv run python experiments/run_all.py

paper:
	cd paper && ./build.sh

all: test experiments paper

clean:
	rm -rf experiments/results/ paper/figures/*.pdf paper/tables/*.tex .pytest_cache __pycache__
