.PHONY: help setup demo chat run server eval interop clean

help:
	@echo Available commands:
	@echo   make setup      - Install dependencies from requirements.txt
	@echo   make demo       - Run the assistant agent demo
	@echo   make chat       - Start interactive terminal chat with the agent
	@echo   make eval       - Run evaluation on testset.jsonl and generate logs
	@echo   make interop    - Test MCP server from secondary host client
	@echo   make server     - Start the MCP server over stdio
	@echo   make clean      - Clean temporary cache files

setup:
	pip install -r requirements.txt

demo:
	python -m src.agent

chat:
	python -m src.agent --chat

run: demo

server:
	python -m src.mcp_server

eval:
	python -m src.eval

interop:
	python -m src.mcp_client_check

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
