#!/bin/bash
SCRIPT_PATH='/home/daisy/cicciara/RAG-with-LLM/LlmWithRag.py'
PYTHON_VENV_PATH='/home/daisy/cicciara/RAG-with-LLM/venv3'

/home/daisy/cicciara/ollama/bin/ollama serve &
source $PYTHON_VENV_PATH/bin/activate

python3 $SCRIPT_PATH > text_output.txt 2>&1

