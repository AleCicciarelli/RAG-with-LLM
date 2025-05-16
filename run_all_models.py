import os

models = [
    "llama3:70b",
    "mixtral:8x7b",
    "deepseek-r1:70b",  
       "mistral:7b"
]

for model in models:
    output_file = f"outputs_FC_{model.replace(':', '_')}"
    os.system(f"OLLAMA_MODEL='{model}' OUTPUT_FILE='{output_file}' python LWRFullRelevantContext.py ")
