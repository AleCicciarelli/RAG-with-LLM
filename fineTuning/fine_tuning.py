import torch
import json
import os
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from unsloth.datasets import get_unsloth_dataset

# --- CONFIG ---
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # oppure 70B
dataset_path = "fineTuning/converted_dataset.jsonl"  # <-- il tuo dataset JSONL
adapter_output = "output_lora"
max_seq_length = 4096
use_bf16 = is_bfloat16_supported()
load_in_4bit = True
batch_size = 2
epochs = 3

# --- 1. Carica modello e tokenizer ---
print("✅ Caricamento modello base...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=load_in_4bit,
    device_map="auto"
)
print("✅ Modello caricato.")

# --- 2. Configura LoRA ---
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM"
)

# --- 3. Carica dataset JSONL ---
print("✅ Caricamento dataset JSONL...")
dataset = load_dataset("json", data_files=dataset_path, split="train")

# --- 4. Formatta prompt e target per l'istruction tuning ---
def formatting(example):
    return {
        "text": f"{example['prompt']}<|end|>\n{example['output']}"
    }

print("✅ Preprocessing dataset...")
dataset = dataset.map(formatting)

# Converti in formato UnsLoTH
dataset = get_unsloth_dataset(dataset, tokenizer, formatting_func=None)

# --- 5. Fine-tuning con SFTTrainer ---
print("🚀 Avvio fine-tuning LoRA...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=max_seq_length,
    dataset_text_field="text",
    packing=False,
    lora_config=lora_config,
    fp16=not use_bf16,
    bf16=use_bf16,
    logging_steps=10,
    save_steps=1000,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    output_dir=adapter_output,
    save_total_limit=1,
)

trainer.train()
model.save_pretrained(adapter_output)
print(f"✅ Adapter LoRA salvato in: {adapter_output}")

# --- 6. INFERENZA ---
print("🔁 Caricamento modello per inferenza...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_output)
FastLanguageModel.for_inference(model)

# --- 7. PROMPT DI TEST ---
inference_prompt = """Metadata: OR present = NO, joins = 1

Return ONLY the JSON output, with no explanation, no introductory sentence, and no trailing comments.
If the answer is not present in the context, return an empty array.

```json
{
    "answer": ["<answer_1>", "<answer_2>", ...],
    "why": ["{{<table_name>_<row>},{<table_name>_<row>}}", "{{<table_name>_<row>}}", ...]
}
```<|end|>
"""

inputs = tokenizer(inference_prompt, return_tensors="pt", truncation=True, max_length=max_seq_length).to("cuda")

# --- 8. GENERAZIONE ---
print("📤 Generazione output JSON...")
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )

output_ids = outputs[0][inputs["input_ids"].shape[1]:]
generated = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

# --- 9. VALIDAZIONE JSON ---
if generated.startswith("```json"):
    generated = generated[len("```json"):].strip()
if generated.endswith("```"):
    generated = generated[:-len("```")].strip()

print("\n🔍 Output Generato:")
print(generated)

try:
    parsed = json.loads(generated)
    print("\n✅ JSON valido:")
    print(json.dumps(parsed, indent=2))
except json.JSONDecodeError as e:
    print(f"\n❌ Errore JSON: {e}")
