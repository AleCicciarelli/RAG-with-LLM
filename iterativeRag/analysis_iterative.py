import json
import csv
import os
from collections import defaultdict

# Grouping predictions by question: each question will have a list of predictions (multiple iterations)
def group_predictions_by_question(pred_list):
    grouped = defaultdict(list)
    for item in pred_list:
        grouped[item["question"]].append(item)
    return grouped

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def compute_metrics(tp, fp, fn, exact=0, total=0):
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    accuracy = exact / total if total else 0
    return precision, recall, f1, accuracy

def evaluate_lists(true_list, pred_list):
    true_set = set(true_list)
    pred_set = set(pred_list)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    return tp, fp, fn

def main():
    gt_data = load_json("ground_truth2.json")
    question_types = load_json("questions.json")

    # Cartella contenente i file di output predetti
    pred_folder = "iterativeRag/outputs_llama70b/"
    global_metrics_file = os.path.join(pred_folder, "global_metrics_iterative.csv")
    type_metrics_file = os.path.join(pred_folder, "metrics_by_type_iterative.csv")
    iteration_metrics_file = os.path.join(pred_folder, "metrics_per_iteration.csv")

    # Scrivi header CSV solo se i file non esistono
    write_header_global = not os.path.exists(global_metrics_file)
    write_header_type = not os.path.exists(type_metrics_file)
    write_header_iter = not os.path.exists(iteration_metrics_file)

    with open(global_metrics_file, "a", newline="", encoding="utf-8") as f_global, \
         open(type_metrics_file, "a", newline="", encoding="utf-8") as f_type, \
         open(iteration_metrics_file, "a", newline="", encoding="utf-8") as f_iter:

        global_writer = csv.writer(f_global)
        type_writer = csv.writer(f_type)
        iter_writer = csv.writer(f_iter)
        if write_header_global:
            global_writer.writerow(["Category", "Precision", "Recall", "F1", "Accuracy"])
        if write_header_type:
            type_writer.writerow([
                "Question Type",
                "Answer Precision", "Answer Recall", "Answer F1", "Answer Accuracy",
                "Explanation Precision", "Explanation Recall", "Explanation F1", "Explanation Accuracy",
                "Count"
            ])
        if write_header_iter:
            iter_writer.writerow([
                "Question", "Iteration", "Question Type",
                "Answer Precision", "Answer Recall", "Answer F1", "Exact Answer",
                "Explanation Precision", "Explanation Recall", "Explanation F1", "Exact Explanation"
            ])

       
       
        pred_file = os.path.join(pred_folder, f"outputs_llama70b_ollama_iterative.json")


        ungrouped_pred_data = load_json(pred_file)
        pred_data = group_predictions_by_question(ungrouped_pred_data)

        assert len(gt_data) == len(pred_data), "Mismatch in number of questions"
        
        print(f"Domande nel file predetto: {len(pred_data)}")
        print(f"Esempio predizione: {next(iter(pred_data.values()))}")
        missing = [gt["query"] for gt in gt_data if gt["query"] not in pred_data]
        print(f"Domande non trovate nelle predizioni: {len(missing)}")
        for q in missing[:5]:  # solo le prime 5
            print(f" - {q}")


        # Inizializza metriche per tipo domanda
        metrics_by_type = defaultdict(lambda: {
            "tp_ans": 0, "fp_ans": 0, "fn_ans": 0,
            "tp_expl": 0, "fp_expl": 0, "fn_expl": 0,
            "exact_ans": 0, "exact_expl": 0,
            "count": 0
        })

        # Metriche globali
        answer_tp = answer_fp = answer_fn = 0
        answer_exact = 0
        expl_tp = expl_fp = expl_fn = 0
        expl_exact = 0
        questions = list(question_types.keys())

        for idx, gt in enumerate(gt_data):
            question = questions[idx]
            q_type = question_types.get(question, "unknown")
            true_answer = [str(x) for x in gt["answer"]] if isinstance(gt["answer"], list) else [str(gt["answer"])]
            true_expl = set()
            if isinstance(gt["why"], str):
                true_expl.update(s.strip("{} ") for s in gt["why"].split("}}") if s.strip())
            elif isinstance(gt["why"], list):
                for s in gt["why"]:
                    true_expl.update(ss.strip("{} ") for ss in s.split("}}") if ss.strip())

            preds = pred_data[question]
            for pred in preds:
                iteration = pred["iteration"]
                try:
                    pred_answer_raw = pred.get("answer", [])
                    if isinstance(pred_answer_raw, list) and len(pred_answer_raw) > 0 and isinstance(pred_answer_raw[0], dict):
                        pred_answer = [str(x) for x in pred_answer_raw[0].get("answer", [])]
                    else:
                        pred_answer = [str(x) for x in pred_answer_raw]
                except Exception as e:
                    print(f"Errore parsing answer iter {iteration} per '{question}': {e}")
                    pred_answer = []

                try:
                    pred_expl_raw = pred.get("answer", [])
                    if isinstance(pred_expl_raw, list) and len(pred_expl_raw) > 0 and isinstance(pred_expl_raw[0], dict):
                        pred_expl = set(x.strip("{} ") for x in pred_expl_raw[0].get("why", []))
                    else:
                        pred_expl = set()
                except Exception as e:
                    print(f"Errore parsing explanation iter {iteration} per '{question}': {e}")
                    pred_expl = set()

                tp_ans, fp_ans, fn_ans = evaluate_lists(true_answer, pred_answer)
                exact_answer = 1 if set(true_answer) == set(pred_answer) else 0
                ans_prec, ans_rec, ans_f1, _ = compute_metrics(tp_ans, fp_ans, fn_ans)

                tp_expl, fp_expl, fn_expl = evaluate_lists(true_expl, pred_expl)
                exact_expl = 1 if (true_expl == pred_expl and exact_answer) else 0
                expl_prec, expl_rec, expl_f1, _ = compute_metrics(tp_expl, fp_expl, fn_expl)

                iter_writer.writerow([
                    question, iteration, q_type,
                    f"{ans_prec:.4f}", f"{ans_rec:.4f}", f"{ans_f1:.4f}", exact_answer,
                    f"{expl_prec:.4f}", f"{expl_rec:.4f}", f"{expl_f1:.4f}", exact_expl
                ])

                m = metrics_by_type[q_type]
                m["count"] += 1
                m["tp_ans"] += tp_ans
                m["fp_ans"] += fp_ans
                m["fn_ans"] += fn_ans
                m["exact_ans"] += exact_answer
                m["tp_expl"] += tp_expl
                m["fp_expl"] += fp_expl
                m["fn_expl"] += fn_expl
                m["exact_expl"] += exact_expl

                answer_tp += tp_ans
                answer_fp += fp_ans
                answer_fn += fn_ans
                answer_exact += exact_answer
                expl_tp += tp_expl
                expl_fp += fp_expl
                expl_fn += fn_expl
                expl_exact += exact_expl

        ans_prec, ans_rec, ans_f1, ans_acc = compute_metrics(answer_tp, answer_fp, answer_fn, answer_exact, len(gt_data))
        expl_prec, expl_rec, expl_f1, expl_acc = compute_metrics(expl_tp, expl_fp, expl_fn, expl_exact, len(gt_data))

        global_writer.writerow(["Answer", f"{ans_prec:.4f}", f"{ans_rec:.4f}", f"{ans_f1:.4f}", f"{ans_acc:.4f}"])
        global_writer.writerow(["Explanation", f"{expl_prec:.4f}", f"{expl_rec:.4f}", f"{expl_f1:.4f}", f"{expl_acc:.4f}"])

        for q_type, stats in metrics_by_type.items():
            ans_m = compute_metrics(stats["tp_ans"], stats["fp_ans"], stats["fn_ans"], stats["exact_ans"], stats["count"])
            expl_m = compute_metrics(stats["tp_expl"], stats["fp_expl"], stats["fn_expl"], stats["exact_expl"], stats["count"])

            type_writer.writerow([
                q_type,
                f"{ans_m[0]:.4f}", f"{ans_m[1]:.4f}", f"{ans_m[2]:.4f}", f"{ans_m[3]:.4f}",
                f"{expl_m[0]:.4f}", f"{expl_m[1]:.4f}", f"{expl_m[2]:.4f}", f"{expl_m[3]:.4f}",
                stats["count"]
            ])

        print(f"Metriche calcolate e scritte per {len(gt_data)} domande.")
        print(f"Risultati salvati in {global_metrics_file}, {type_metrics_file} e {iteration_metrics_file}")

if __name__ == "__main__":
    main()
