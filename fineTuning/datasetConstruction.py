'''
import json
import re

input_file = "ground_truth2.json"
with open(input_file, "r", encoding="utf-8") as f:
    input_data = json.load(f)

print(input_data[:2])


# Prompt fisso
PROMPT = (
    "Return ONLY the JSON output, with no explanation, no introductory sentence, and no trailing comments.\n"
    "If the answer is not present in the context, return an empty array.\n\n"
)

def main():
    output = []
    for item in input_data:
        answer = item["answer"]
        why = item["why"]

        
        # Costruisco la stringa instruction con i metadati base
        instruction = (
            f"Metadata: OR present = , joins = "
        )

        example = {
            "prompt": PROMPT,
            "instruction": instruction,
            "answer": {
                "answer": answer,
                "why": why
            }
        }
        output.append(example)
    
    # Salvo il file JSON
    with open("instruction_tuned_dataset.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Dataset with {len(output)} examples saved to 'instruction_tuned_dataset.json'.")

if __name__ == "__main__":
    main()
'''
import json

input_file = "fineTuning/instruction_tuned_dataset.json"
output_file = "fineTuning/converted_dataset.jsonl"

with open(input_file, "r") as f:
    data = json.load(f)

with open(output_file, "w") as out:
    for example in data:
        base_prompt = example["prompt"]
        instruction = example["instruction"]
        full_prompt = f"{instruction}\n\n{base_prompt}" if instruction else base_prompt
        output_str = json.dumps(example["answer"], ensure_ascii=False, indent=2)
        out.write(json.dumps({"prompt": full_prompt, "output": output_str}, ensure_ascii=False) + "\n")

print(f"Dataset convertito con successo in {output_file}")
