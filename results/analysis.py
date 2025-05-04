import json
import csv
from collections import defaultdict
import matplotlib.pyplot as plt

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_lists(true_list, pred_list):
    true_set = set(true_list)
    pred_set = set(pred_list)

    tp = list(true_set & pred_set)
    fp = list(pred_set - true_set)
    fn = list(true_set - pred_set)

    return tp, fp, fn

def explanation_to_tuple_list(expl):
    return [(e["file"], str(e["row"])) for e in expl]

def compute_metrics(total_tp, total_fp, total_fn, total_exact, n):
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    accuracy = total_exact / n if n else 0
    return precision, recall, accuracy

def main():
    gt_data = load_json("results/true_answers.json")
    pred_data = load_json("outputs_k50_llama70b.json")
    question_types = load_json("questions.json")  # Dizionario dei tipi di domanda

    assert len(gt_data) == len(pred_data), "Mismatch in number of questions"

    # Normalizza i tipi di domanda
    #question_types = {normalize(k): v for k, v in question_types_raw.items()}

    # Variabili per raccogliere i risultati per tipo di domanda
    results_by_type = defaultdict(lambda: {
        "answer_tp": 0, "answer_fp": 0, "answer_fn": 0, "answer_exact": 0,
        "expl_tp": 0, "expl_fp": 0, "expl_fn": 0, "expl_exact": 0,
        "count": 0
    })

    # Variabili per raccogliere i risultati globali
    answer_tp = answer_fp = answer_fn = answer_exact = 0
    expl_tp = expl_fp = expl_fn = expl_exact = 0
    rows = []

    for gt, pred in zip(gt_data, pred_data):
        question = gt["question"]
        q_type = question_types.get(question, "unknown")  # Tipo della domanda

        group = results_by_type[q_type]
        group["count"] += 1
        if q_type == "unknown":
            print(question)
        # Calcola le metriche per le risposte
        tp_ans, fp_ans, fn_ans = evaluate_lists(gt["answer"], pred["answer"])
        if not fp_ans and not fn_ans:
            answer_exact += 1
            group["answer_exact"] += 1
        answer_tp += len(tp_ans)
        answer_fp += len(fp_ans)
        answer_fn += len(fn_ans)
        group["answer_tp"] += len(tp_ans)
        group["answer_fp"] += len(fp_ans)
        group["answer_fn"] += len(fn_ans)

        # Calcola le metriche per le spiegazioni
        gt_expl = explanation_to_tuple_list(gt["explanation"])
        pred_expl = explanation_to_tuple_list(pred["explanation"])
        tp_expl, fp_expl, fn_expl = evaluate_lists(gt_expl, pred_expl)
        if not fp_expl and not fn_expl:
            expl_exact += 1
            group["expl_exact"] += 1
        expl_tp += len(tp_expl)
        expl_fp += len(fp_expl)
        expl_fn += len(fn_expl)
        group["expl_tp"] += len(tp_expl)
        group["expl_fp"] += len(fp_expl)
        group["expl_fn"] += len(fn_expl)

        # Aggiungi i risultati per ogni domanda
        rows.append({
            "question": question,
            "type": q_type,
            "true_answer": gt["answer"],
            "pred_answer": pred["answer"],
            "correct_answer": tp_ans,
            "false_positive_answer": fp_ans,
            "false_negative_answer": fn_ans,
            "true_explanation": gt_expl,
            "pred_explanation": pred_expl,
            "correct_explanation": tp_expl,
            "false_positive_explanation": fp_expl,
            "false_negative_explanation": fn_expl
        })

    # Scrive i risultati per domanda in un file CSV
    with open("evaluation_results_llama70b.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Calcola le metriche globali
    total = len(gt_data)
    answer_prec, answer_rec, answer_acc = compute_metrics(answer_tp, answer_fp, answer_fn, answer_exact, total)
    expl_prec, expl_rec, expl_acc = compute_metrics(expl_tp, expl_fp, expl_fn, expl_exact, total)

    # Stampa le metriche globali
    print("\n--- Global Metrics ---")
    print(f"Answer - Precision: {answer_prec:.2f}, Recall: {answer_rec:.2f}, Accuracy: {answer_acc:.2f}")
    print(f"Explanation - Precision: {expl_prec:.2f}, Recall: {expl_rec:.2f}, Accuracy: {expl_acc:.2f}")

    # Stampa le metriche per tipo di domanda
    print("\n--- Metrics by Question Type ---")
    for q_type, data in results_by_type.items():
        count = data["count"]
        answer_metrics = compute_metrics(data["answer_tp"], data["answer_fp"], data["answer_fn"], data["answer_exact"], count)
        expl_metrics = compute_metrics(data["expl_tp"], data["expl_fp"], data["expl_fn"], data["expl_exact"], count)

        print(f"\nType: {q_type.replace('_', ' ').title()} ({count} questions)")
        print(f"Answer      - Precision: {answer_metrics[0]:.2f}, Recall: {answer_metrics[1]:.2f}, Accuracy: {answer_metrics[2]:.2f}")
        print(f"Explanation - Precision: {expl_metrics[0]:.2f}, Recall: {expl_metrics[1]:.2f}, Accuracy: {expl_metrics[2]:.2f}")



    # Salva il grafico globale
    labels = ['Precision', 'Recall', 'Accuracy']
    answer_scores = [answer_prec, answer_rec, answer_acc]
    expl_scores = [expl_prec, expl_rec, expl_acc]

    x = range(len(labels))
    width = 0.35

    plt.bar([i - width/2 for i in x], answer_scores, width=width, label='Answer')
    plt.bar([i + width/2 for i in x], expl_scores, width=width, label='Explanation')
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Global LLM Evaluation Metrics")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("evaluation_metrics_mistral.png")
    plt.show()
    # --- Salva le metriche globali ---
    with open("global_metrics_llama70b_K50.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Precision", "Recall", "Accuracy"])
        writer.writerow(["Answer", f"{answer_prec:.4f}", f"{answer_rec:.4f}", f"{answer_acc:.4f}"])
        writer.writerow(["Explanation", f"{expl_prec:.4f}", f"{expl_rec:.4f}", f"{expl_acc:.4f}"])

    # --- Salva le metriche per tipo di domanda ---
    with open("metrics_by_type_llama70b_K50.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question Type", "Answer Precision", "Answer Recall", "Answer Accuracy",
                        "Explanation Precision", "Explanation Recall", "Explanation Accuracy", "#Questions"])
        
        for q_type, data in results_by_type.items():
            count = data["count"]
            answer_metrics = compute_metrics(data["answer_tp"], data["answer_fp"], data["answer_fn"], data["answer_exact"], count)
            expl_metrics = compute_metrics(data["expl_tp"], data["expl_fp"], data["expl_fn"], data["expl_exact"], count)

            writer.writerow([
                q_type,
                f"{answer_metrics[0]:.4f}", f"{answer_metrics[1]:.4f}", f"{answer_metrics[2]:.4f}",
                f"{expl_metrics[0]:.4f}", f"{expl_metrics[1]:.4f}", f"{expl_metrics[2]:.4f}",
                count
            ])

if __name__ == "__main__":
    main()
