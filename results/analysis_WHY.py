import json
import csv
from collections import defaultdict
from typing import List, Tuple, Set, Union

def load_json(path: str) -> Union[dict, list]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_lists(true: Union[List[str], Set[str]], pred: Union[List[str], Set[str]]) -> Tuple[int, int, int]:
    true_set = set(true)
    pred_set = set(pred)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    return tp, fp, fn

def compute_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    return precision, recall, accuracy

def normalize_why_entry(entry: str) -> str:
    """Normalizza una spiegazione per confronto insensibile all'ordine interno."""
    parts = entry.strip("{}").split(",")
    return "{" + ",".join(sorted(p.strip() for p in parts)) + "}"

def main():
    pred_path = "outputs_FC_llama70b_3.json"
    gt_path = "ground_truth.json"
    question_type_path = "questions.json"
    output_csv_type = "metrics_by_type_FC_llama70b_3.csv"
    output_csv_global = "metrics_global_FC_llama70b_3.csv"

    pred_data = load_json(pred_path)
    gt_data = load_json(gt_path)
    question_types = load_json(question_type_path)

    assert len(pred_data) == len(gt_data), "Mismatch in prediction and ground truth lengths"

    metrics_by_type = defaultdict(lambda: {
        "tp_ans": 0, "fp_ans": 0, "fn_ans": 0,
        "tp_expl": 0, "fp_expl": 0, "fn_expl": 0, "count": 0
    })
    
    # Variabili per raccogliere i risultati globali
    answer_tp = answer_fp = answer_fn = 0
    expl_tp = expl_fp = expl_fn = 0

    for gt, pred in zip(gt_data, pred_data):
        question = pred["question"]
        q_type = question_types.get(question, "unknown")

        # Estrai risposte
        true_answer = gt["f1"]
        if not isinstance(true_answer, list):
            true_answer = [str(true_answer)]
        else:
            true_answer = [str(x) for x in true_answer]

        pred_answer = [str(x) for x in pred["answer"][0]["answer"]]

        tp_ans, fp_ans, fn_ans = evaluate_lists(true_answer, pred_answer)

        # Estrai spiegazioni
        true_expl_raw = gt["f2"]
        if isinstance(true_expl_raw, str):
            true_expl = set(s.strip("{}") for s in true_expl_raw.split("}}") if s)
        else:
            true_expl = set()
            for s in true_expl_raw:
                true_expl.update(ss.strip("{}") for ss in s.split("}}") if ss)

        pred_expl = set(x.strip("{}") for x in pred["answer"][0]["why"])

        # Debug: stampiamo le risposte e spiegazioni per vedere cosa succede
        #print(f"Question: {question}")
        #print(f"True Answer: {true_answer}")
        #print(f"Predicted Answer: {pred_answer}")
        #print(f"True Explanation: {true_expl}")
        #print(f"Predicted Explanation: {pred_expl}")

        # Verifica se le risposte sono corrette prima di valutare le spiegazioni
        tp_expl = fp_expl = fn_expl = 0
        for answer in pred_answer:
            if answer in true_answer:
                tp_expl_tmp, fp_expl_tmp, fn_expl_tmp = evaluate_lists(true_expl, pred_expl)
                tp_expl += tp_expl_tmp
                fp_expl += fp_expl_tmp
                fn_expl += fn_expl_tmp

        # Aggiorna metriche globali per risposte e spiegazioni
        answer_tp += tp_ans
        answer_fp += fp_ans
        answer_fn += fn_ans
        expl_tp += tp_expl
        expl_fp += fp_expl
        expl_fn += fn_expl

        # Aggiorna metriche per tipo di domanda
        m = metrics_by_type[q_type]
        m["count"] += 1
        m["tp_ans"] += tp_ans
        m["fp_ans"] += fp_ans
        m["fn_ans"] += fn_ans
        m["tp_expl"] += tp_expl
        m["fp_expl"] += fp_expl
        m["fn_expl"] += fn_expl

    # Calcola le metriche globali
    total = len(gt_data)
    answer_prec, answer_rec, answer_acc = compute_metrics(answer_tp, answer_fp, answer_fn)
    expl_prec, expl_rec, expl_acc = compute_metrics(expl_tp, expl_fp, expl_fn)

    # Stampa le metriche globali
    print("\n--- Global Metrics ---")
    print(f"Answer - Precision: {answer_prec:.2f}, Recall: {answer_rec:.2f}, Accuracy: {answer_acc:.2f}")
    print(f"Explanation - Precision: {expl_prec:.2f}, Recall: {expl_rec:.2f}, Accuracy: {expl_acc:.2f}")

    # --- Salva le metriche globali ---
    with open(output_csv_global, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Precision", "Recall", "Accuracy"])
        writer.writerow(["Answer", f"{answer_prec:.4f}", f"{answer_rec:.4f}", f"{answer_acc:.4f}"])
        writer.writerow(["Explanation", f"{expl_prec:.4f}", f"{expl_rec:.4f}", f"{expl_acc:.4f}"])

    # Scrittura dei risultati per tipo di domanda
    with open(output_csv_type, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_type",
            "precision_ans", "recall_ans", "accuracy_ans",
            "precision_expl", "recall_expl", "accuracy_expl", "#questions"
        ])

        for q_type, m in metrics_by_type.items():
            p_ans, r_ans, a_ans = compute_metrics(m["tp_ans"], m["fp_ans"], m["fn_ans"])
            p_expl, r_expl, a_expl = compute_metrics(m["tp_expl"], m["fp_expl"], m["fn_expl"])

            writer.writerow([
                q_type,
                round(p_ans, 4), round(r_ans, 4), round(a_ans, 4),
                round(p_expl, 4), round(r_expl, 4), round(a_expl, 4), m["count"]
            ])

if __name__ == "__main__":
    main()
