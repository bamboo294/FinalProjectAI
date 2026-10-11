.PHONY: help setup test run demo eval server clean

help:
	@echo Available commands:
	@echo   make setup      - Install dependencies from requirements.txt
	@echo   make demo       - Run the assistant agent demo
	@echo   make server     - Start the MCP server
	@echo   make eval       - Run evaluation on testset.jsonl
	@echo   make test       - Run test suite with pytest
	@echo   make clean      - Clean temporary cache files

setup:
	pip install -r requirements.txt

demo:
	python src/agent.py

run: demo

server:
	python src/mcp_server.py

eval:
	python -m pytest tests/ -v

test:
	python -m pytest tests/ -v

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"

