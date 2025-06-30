import unsloth
import torch
import json
import os
from transformers import AutoTokenizer, TrainingArguments
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, TaskType
from datasets import load_dataset
# --- CONFIG ---
#base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # oppure 70B
base_model_name = "unsloth/Phi-3-mini-4k-instruct"
dataset_path = "fineTuning/converted_dataset.jsonl"  # <-- il tuo dataset JSONL
adapter_output = "output_lora_phi3mini"  # <-- dove vuoi salvare l'adapter LoRA
max_seq_length = 4096
use_bf16 = is_bfloat16_supported()
load_in_4bit = True
batch_size = 2
epochs = 3

# --- 1. Caricamento modello base + tokenizer ---
print("✅ Caricamento modello base...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)
tokens = tokenizer.tokenize("<|eot_id|>")
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Tokens: {tokens}")
print(f"Token IDs: {ids}")
print("Matches eos_token_id?", ids[0] == tokenizer.eos_token_id)
tokens = tokenizer.tokenize("<|end|>")
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"<|end|> Tokens: {tokens}")
print(f"Token IDs: {ids}")
# --- 2. Configurazione e applicazione LoRA ---
print("✅ Configurazione LoRA...")

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none"
)

# --- 3. Caricamento e preprocessing dataset ---
print("✅ Caricamento dataset JSONL...")
dataset = load_dataset("json", data_files=dataset_path, split="train")

# --- 4. Format dataset in instruction-style ---
def formatting(example):
    return {"text": f"{example['prompt']}<|end|>\n{example['output']}<|end|>"}

print("✅ Formatting dataset...")
dataset = dataset.map(formatting)

# --- 5. Training Arguments ---
training_args = TrainingArguments(
    output_dir=adapter_output,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",
)

# --- 6. Fine-tuning ---
print("🚀 Avvio fine-tuning LoRA...")
print(f"Dataset format: {dataset[0]['text']}")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
print("✅ Fine-tuning completato!")

# --- 7. Salvataggio adapter ---
model.save_pretrained(adapter_output)
print("✅ Adapter salvato in ./output_lora")


# --- 6. INFERENZA ---
print("🔁 Caricamento modello per inferenza...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True
)
model = PeftModel.from_pretrained(model, adapter_output)
FastLanguageModel.for_inference(model)

# --- 7. PROMPT DI TEST ---
inference_prompt = """
Your task is to provide the correct answer(s) to this question: QUESTION_HERE, based ONLY on the given context: CONTEXT_HERE.
For each answer, explain WHY it appears using **Witness Sets**: minimal sets of input tuples that justify the result.
Format of Witness Sets (as strings):  
- If there is ONE relevant tuple set: "{{<table_name>_<row>}}"  
- If there are MULTIPLE: "{{<table_name>_<row>},{<table_name>_<row>},...}}"  
IMPORTANT:
Return ONLY the JSON output, with no explanation, no introductory sentence, and no trailing comments.
If your output is not a valid JSON block in the format described, it will be discarded.
If the answer is not present in the context, return an empty array.


INVALID OUTPUT EXAMPLE (will be discarded):
The answer is: {"answer": [...], "why": [...]}
CONTEXT:
students.csv:
- student_id:1, name:Giulia, surname:Rossi, age:20
- student_id:2, name:Marco, surname:Bianchi, age:22
courses.csv:
- course_id:101, course_name:Machine Learning, credits:6
- course_id:104, course_name:Advanced Algorithms, credits:6
enrollments.csv:
- enrollment_id:1, student_id:1, course_id:101, semester:2023
- enrollment_id:4, student_id:1, course_id:104, semester:2023
- enrollment_id:10, student_id:2, course_id:101, semester:2023

QUESTION:
Which are the students (specify name and surname) enrolled in Machine Learning or in Advanced Algorithm courses?

Return ONLY the JSON output, without any explanation or markdown formatting.

Format:
{
  "answer": ["<answer1>", "<answer2>", ...],
  "why": ["{<table_row>,<table_row>...}, {....}}", ...]
}
If the answer is not present in the context, return:
{
  "answer": [],
  "why": []
}
<|end|>
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

if generated.startswith("```"):
    generated = generated[len("```"):].strip()
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
