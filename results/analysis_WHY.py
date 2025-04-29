import json
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import re

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_lists(true_list, pred_list):
    true_set = set(true_list)
    pred_set = set(pred_list)
    print("true")
    print(true_set)
    print("pred")
    print(pred_set)
    tp = list(true_set & pred_set)
    fp = list(pred_set - true_set)
    fn = list(true_set - pred_set)

    return tp, fp, fn

def compute_metrics(total_tp, total_fp, total_fn, total_exact, n):
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    accuracy = total_exact / n if n else 0
    return precision, recall, accuracy

def main():
    # Carica i dati dei file JSON
    gt_data = load_json("results/true_answers_WHY.json")
    pred_data = load_json("results/outputs_mistral24b_WHY_K50.json")  # Ora il file è in formato .json
    question_types = load_json("questions.json")  # Dizionario dei tipi di domanda

    # Debug: Verifica il numero di domande in ciascun file
    print(f"Numero di domande nel file true_answers.json: {len(gt_data)}")
    print(f"Numero di domande nel file outputs.json: {len(pred_data)}")

    # Verifica che i numeri corrispondano
    if len(gt_data) != len(pred_data):
        print("ATTENZIONE: Il numero di domande nei file non corrisponde!")
        return  # Ferma l'esecuzione se c'è un mismatch

    # Variabili per raccogliere i risultati per tipo di domanda
    results_by_type = defaultdict(lambda: {
        "answer_tp": 0, "answer_fp": 0, "answer_fn": 0, "answer_exact": 0,
        "count": 0
    })
    # Variabili per raccogliere i risultati globali
    answer_tp = answer_fp = answer_fn = answer_exact = 0
    rows = []

    for gt, pred in zip(gt_data, pred_data):
        question = gt["question"]
        q_type = question_types.get(question, "unknown")  # Tipo della domanda
        
        group = results_by_type[q_type]
        group["count"] += 1

        # Calcola le metriche per le risposte
        tp_ans, fp_ans, fn_ans = evaluate_lists(gt["answer"], pred["result"])
        if not fp_ans and not fn_ans:
            answer_exact += 1
            group["answer_exact"] += 1
        answer_tp += len(tp_ans)
        answer_fp += len(fp_ans)
        answer_fn += len(fn_ans)
        group["answer_tp"] += len(tp_ans)
        group["answer_fp"] += len(fp_ans)
        group["answer_fn"] += len(fn_ans)

        
        # Aggiungi i risultati per ogni domanda
        rows.append({
            "question": question,
            "type": q_type,
            "true_answer": gt["answer"],
            "pred_answer": pred["result"],
            "correct_answer": tp_ans,
            "false_positive_answer": fp_ans,
            "false_negative_answer": fn_ans,
        })

    # Scrive i risultati per domanda in un file CSV
    with open("evaluation_results_WHY.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Calcola le metriche globali
    total = len(gt_data)
    answer_prec, answer_rec, answer_acc = compute_metrics(answer_tp, answer_fp, answer_fn, answer_exact, total)

    # Stampa le metriche globali
    print("\n--- Global Metrics ---")
    print(f"Answer - Precision: {answer_prec:.2f}, Recall: {answer_rec:.2f}, Accuracy: {answer_acc:.2f}")

    # Stampa le metriche per tipo di domanda
    print("\n--- Metrics by Question Type ---")
    for q_type, data in results_by_type.items():
        count = data["count"]
        answer_metrics = compute_metrics(data["answer_tp"], data["answer_fp"], data["answer_fn"], data["answer_exact"], count)

        print(f"\nType: {q_type.replace('_', ' ').title()} ({count} questions)")
        print(f"Answer      - Precision: {answer_metrics[0]:.2f}, Recall: {answer_metrics[1]:.2f}, Accuracy: {answer_metrics[2]:.2f}")

    # Salva il grafico globale
    labels = ['Precision', 'Recall', 'Accuracy']
    answer_scores = [answer_prec, answer_rec, answer_acc]

    x = range(len(labels))
    width = 0.25

    plt.bar([i - width for i in x], answer_scores, width=width, label='Answer')
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Global LLM Evaluation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation_metrics_WHY_mistral24b_k50.png")
    plt.show()

if __name__ == "__main__":
    main()
