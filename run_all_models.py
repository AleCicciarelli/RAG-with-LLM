import os

models = [
    "llama3.3:70b",
    "mistral-small3.1:24b",
    "deepseek-r1:8b",
    "deepseek-r1:32b"
]

for model in models:
    output_file = f"outputs_FC_{model.replace(':', '_')}"
    os.system(f"OLLAMA_MODEL='{model}' OUTPUT_FILE='{output_file}' python LWRFullRelevantContext.py ")
