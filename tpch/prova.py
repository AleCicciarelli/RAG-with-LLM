from transformers import AutoTokenizer
from langchain_community.chat_models import ChatOllama

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
llm = ChatOllama(model="llama3:8b", temperature=0)

print(f'max tokens, {llm.num_ctx}')
MAX_TOKENS = tokenizer.model_max_length
print(f"Max token length from tokenizer: {MAX_TOKENS}")