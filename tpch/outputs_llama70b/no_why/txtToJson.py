import json
import re

def parse_results_txt(txt):
    results = []
    # Dividi in blocchi per ogni risultato
    blocks = re.split(r"-{4}Results \d+-{4}", txt)
    for block in blocks:
        if "Question:" in block and "Answer:" in block:
            question_match = re.search(r"Question:(.*?)Answer:", block, re.DOTALL)
            answer_match = re.search(r"['\"]?answer['\"]?\s*:\s*(\[[^\]]*\]|\{[^\}]*\}|\".*?\"|\d+)", block, re.DOTALL)
            why_match = re.search(r"['\"]?why['\"]?\s*:\s*(\[[^\]]*\]|\{[^\}]*\}|\".*?\"|\d+)", block, re.DOTALL)

            if question_match and answer_match:
                question = question_match.group(1).strip()

                # Pulizia e parsing dell'answer
                answer_raw = answer_match.group(1).strip().rstrip(",")  # rimuove virgole finali
                try:
                    answer = json.loads(answer_raw)
                    if not isinstance(answer, list):
                        answer = [answer]
                except:
                    answer = [answer_raw]

                # Pulizia e parsing del why
                if why_match:
                    why_raw = why_match.group(1).strip()
                    try:
                        why = json.loads(why_raw)
                        if not isinstance(why, list):
                            why = [why]
                    except:
                        why = [why_raw]
                else:
                    why = []

                results.append({
                    "question": question,
                    "answer": answer,
                    "why": why
                })

    return results

# Leggi da file
with open("outputs_ollama_llama70b/why/outputs_llama70b_why_new.txt", "r", encoding="utf-8") as f:
    content = f.read()

parsed = parse_results_txt(content)

# Salva in JSON
with open("outputs_ollama_llama70b/why/outputs_ollama_llama70b_why_new.json", "w", encoding="utf-8") as f:
    json.dump(parsed, f, indent=2, ensure_ascii=False)

print("✅ Conversione completata. Output salvato in 'outputs_ollama_llama70b_why.json'")

'''
import json
import re

def parse_results_txt(txt):
    results = []
    # Dividi in blocchi per ogni risultato
    blocks = re.split(r"-{4}Results \d+-{4}", txt)
    for block in blocks:
        if "Question:" in block and "Answer:" in block:
            question_match = re.search(r"Question:(.*?)Answer:", block, re.DOTALL)
            answer_match = re.search(r"Answer:\s*\{['\"]answer['\"]\s*:\s*['\"](.*?)['\"]\s*\}", block, re.DOTALL)

            if question_match and answer_match:
                question = question_match.group(1).strip()
                answer_raw = answer_match.group(1).strip()

                # Converti gli \n in una vera lista
                answer_list = [line.strip() for line in answer_raw.split("\\n") if line.strip()]
                
                results.append({
                    "question": question,
                    "answer": answer_list
                })

    return results

# Leggi da file (modifica 'results.txt' con il tuo nome file)
with open("outputs_ollama_llama70b/no_why/outputs_llama70b_nowhy2.txt", "r", encoding="utf-8") as f:
    content = f.read()

parsed = parse_results_txt(content)

# Salva in JSON
with open("outputs_ollama_llama70b/no_why/outputs_ollama_llama70b_nowhy2.json", "w", encoding="utf-8") as f:
    json.dump(parsed, f, indent=2, ensure_ascii=False)

print("✅ Conversione completata. Output salvato in 'results.json'")
'''