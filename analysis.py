import json
import csv
from collections import Counter
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
    gt_data = load_json("true_answers.json")
    pred_data = load_json("all_outputs_cosine.json")

    assert len(gt_data) == len(pred_data), "Mismatch in number of questions"

    rows = []
    answer_tp = answer_fp = answer_fn = answer_exact = 0
    expl_tp = expl_fp = expl_fn = expl_exact = 0

    for gt, pred in zip(gt_data, pred_data):
        question = gt["question"]

        # Evaluate answer
        tp_ans, fp_ans, fn_ans = evaluate_lists(gt["answer"], pred["answer"])
        if not fp_ans and not fn_ans:
            answer_exact += 1
        answer_tp += len(tp_ans)
        answer_fp += len(fp_ans)
        answer_fn += len(fn_ans)

        # Evaluate explanation
        gt_expl = explanation_to_tuple_list(gt["explanation"])
        pred_expl = explanation_to_tuple_list(pred["explanation"])
        tp_expl, fp_expl, fn_expl = evaluate_lists(gt_expl, pred_expl)
        if not fp_expl and not fn_expl:
            expl_exact += 1
        expl_tp += len(tp_expl)
        expl_fp += len(fp_expl)
        expl_fn += len(fn_expl)

        rows.append({
            "question": question,
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

    # Write per-question results to CSV
    with open("evaluation_results_cosine.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Compute global metrics
    total = len(gt_data)
    answer_prec, answer_rec, answer_acc = compute_metrics(answer_tp, answer_fp, answer_fn, answer_exact, total)
    expl_prec, expl_rec, expl_acc = compute_metrics(expl_tp, expl_fp, expl_fn, expl_exact, total)

    print("\n--- Global Metrics ---")
    print(f"Answer - Precision: {answer_prec:.2f}, Recall: {answer_rec:.2f}, Accuracy: {answer_acc:.2f}")
    print(f"Explanation - Precision: {expl_prec:.2f}, Recall: {expl_rec:.2f}, Accuracy: {expl_acc:.2f}")

    # Bar chart
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
    plt.title("LLM Evaluation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation_metrics_cosine.png")
    plt.show()

if __name__ == "__main__":
    main()
