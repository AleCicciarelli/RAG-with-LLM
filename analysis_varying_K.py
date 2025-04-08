import json
import os
import csv
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

def process_file(gt_data, pred_data):
    answer_tp = answer_fp = answer_fn = answer_exact = 0
    expl_tp = expl_fp = expl_fn = expl_exact = 0

    for gt, pred in zip(gt_data, pred_data):
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

    total = len(gt_data)
    answer_metrics = compute_metrics(answer_tp, answer_fp, answer_fn, answer_exact, total)
    explanation_metrics = compute_metrics(expl_tp, expl_fp, expl_fn, expl_exact, total)
    return answer_metrics, explanation_metrics

def main():
    gt_data = load_json("true_answers.json")

    ks = list(range(1, 21))
    answer_results = {"precision": [], "recall": [], "accuracy": []}
    expl_results = {"precision": [], "recall": [], "accuracy": []}

    with open("metrics_by_k.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "answer_precision", "answer_recall", "answer_accuracy",
                         "expl_precision", "expl_recall", "expl_accuracy"])

        for k in ks:
            file_path = f"outputs_k_{k}.json"
            if not os.path.exists(file_path):
                print(f"[k={k}] File not found, skipping.")
                continue

            pred_data = load_json(file_path)
            answer_metrics, expl_metrics = process_file(gt_data, pred_data)

            answer_results["precision"].append(answer_metrics[0])
            answer_results["recall"].append(answer_metrics[1])
            answer_results["accuracy"].append(answer_metrics[2])

            expl_results["precision"].append(expl_metrics[0])
            expl_results["recall"].append(expl_metrics[1])
            expl_results["accuracy"].append(expl_metrics[2])

            writer.writerow([
                k,
                *answer_metrics,
                *expl_metrics
            ])

    # Plot answer metrics
    plt.figure(figsize=(10, 6))
    plt.plot(ks, answer_results["precision"], label="Precision", marker='o')
    plt.plot(ks, answer_results["recall"], label="Recall", marker='s')
    plt.plot(ks, answer_results["accuracy"], label="Accuracy", marker='^')
    plt.title("Answer Metrics vs k")
    plt.xlabel("k (top-k documents retrieved)")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("answer_metrics_vs_k.png")
    plt.show()

    # Plot explanation metrics
    plt.figure(figsize=(10, 6))
    plt.plot(ks, expl_results["precision"], label="Precision", marker='o')
    plt.plot(ks, expl_results["recall"], label="Recall", marker='s')
    plt.plot(ks, expl_results["accuracy"], label="Accuracy", marker='^')
    plt.title("Explanation Metrics vs k")
    plt.xlabel("k (top-k documents retrieved)")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("explanation_metrics_vs_k.png")
    plt.show()

if __name__ == "__main__":
    main()
