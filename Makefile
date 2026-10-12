.PHONY: help setup demo run server eval clean

help:
	@echo Available commands:
	@echo   make setup      - Install dependencies from requirements.txt
	@echo   make demo       - Run the assistant agent demo
	@echo   make server     - Start the MCP server
	@echo   make eval       - Run evaluation on testset.jsonl
	@echo   make clean      - Clean temporary cache files

setup:
	pip install -r requirements.txt

demo:
	python -m src.agent

run: demo

server:
	python -m src.mcp_server

eval:
	python -m src.agent

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
