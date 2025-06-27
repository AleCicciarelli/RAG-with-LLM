import random
import json
from datasets import load_dataset

dataset_path = "fineTuning/converted_dataset.jsonl" 
print("✅ Caricamento dataset JSONL...")
dataset = load_dataset("json", data_files=dataset_path, split="train")

# --- 4. Format dataset in instruction-style ---
def formatting(example):
    return {"text": f"{example['prompt']}<|eot_id|>\n{example['output']}"}

print("✅ Formatting dataset...")
dataset = dataset.map(formatting)

def augment_data(examples):
    augmented_examples = []
    for example in examples:
        prompt = example['prompt']
        output = example['output']

        # Table name replacement
        table_names = ['students', 'departments', 'courses']
        replaced_table_name = random.choice(table_names)
        prompt = prompt.replace('students', replaced_table_name)

        # Row number modification
        row_number = random.randint(0, 10)
        output = output.replace('{{students_0}}', f'{{students_{row_number}}}')

        # Prompt rewording
        reworded_prompt = prompt.replace('Return ONLY the JSON output', 'Generate the JSON output without explanations')
        prompt = reworded_prompt

        # Answer modification
        answer = json.loads(output)['answer']
        random.shuffle(answer)
        output = json.dumps({'answer': answer, 'why': json.loads(output)['why']})

        augmented_examples.append({'prompt': prompt, 'output': output})

    return augmented_examples

# Apply data augmentation to the original dataset
augmented_dataset = augment_data(dataset)


# Salva il dataset aumentato in un file JSON
with open('augmented_dataset.json', 'w') as f:
    json.dump(augmented_dataset, f, indent=4)