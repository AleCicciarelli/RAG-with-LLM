
from transformers import AutoModelForCausalLM
from peft import PeftModel

# ⚠️ Usa lo stesso modello base usato nel fine-tuning
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
lora_path = "output_lora"
merged_output_path = "merged_model"

print("🔄 Caricamento modello base...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype="auto",
    device_map="auto"
)

print("🔗 Caricamento LoRA...")
model = PeftModel.from_pretrained(base_model, lora_path)

print("🧬 Merge LoRA...")
model = model.merge_and_unload()

print("💾 Salvataggio modello mergiato...")
model.save_pretrained(merged_output_path)
'''
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("unsloth/Phi-3-mini-4k-instruct", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "output_lora")

merged = model.merge_and_unload()
merged.save_pretrained("phi3-merged")
'''