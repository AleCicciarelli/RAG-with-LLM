'''
import json
import re

input_file = "ground_truth2.json"
with open(input_file, "r", encoding="utf-8") as f:
    input_data = json.load(f)

print(input_data[:2])


# Prompt fisso
PROMPT = ( """
     Your task is to provide the correct answer(s) to this question: QUESTION_HERE, based ONLY on the given context: CONTEXT_HERE.
        For each answer, explain WHY it appears using **Witness Sets**: minimal sets of input tuples that justify the result.
        Format of Witness Sets (as strings):  
        - If there is ONE relevant tuple set: "{{<table_name>_<row>}}"  
        - If there are MULTIPLE: "{{<table_name>_<row>},{<table_name>_<row>},...}}"  
        IMPORTANT:
        Return ONLY the JSON output, with no explanation, no introductory sentence, and no trailing comments.
        If your output is not a valid JSON block in the format described, it will be discarded.
        If the answer is not present in the context, return an empty array.
        """
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
    with open("instruction_tuned_dataset2.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Dataset with {len(output)} examples saved to 'instruction_tuned_dataset2.json'.")

if __name__ == "__main__":
    main()
'''
import json

input_file = "fineTuning/instruction_tuned_dataset2.json"
output_file = "fineTuning/converted_dataset2.jsonl"

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
