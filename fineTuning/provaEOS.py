from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit")

# Test tokenizzazione
tokens = tokenizer.tokenize("<|end|>")
ids = tokenizer.convert_tokens_to_ids(tokens)

print(f"Tokens: {tokens}")
print(f"Token IDs: {ids}")
print("eos_token:", tokenizer.eos_token)
print("eos_token_id:", tokenizer.eos_token_id)