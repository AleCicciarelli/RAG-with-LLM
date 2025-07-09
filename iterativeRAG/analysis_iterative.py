import json
import csv
import os
from collections import defaultdict
import re
def group_predictions_by_question(pred_list):
    grouped = defaultdict(list)
    for item in pred_list:
        grouped[item["question"]].append(item)
    return grouped
def parse_why_item(items):
    if not items:
        return []

    result = []
    
    for item in items:
        item = item.strip()
        
        if item == "{{}}":
            continue

        if not (item.startswith("{{") and item.endswith("}}")):
            continue

        # Rimuovi le doppie graffe esterne
        inner = item[2:-2].strip()

        # Caso con più gruppi dentro una singola stringa: contiene '},{'
        if '},{' in inner:
            # Divide tra i gruppi interni
            group_strings = inner.split('},{')
            for g in group_strings:
                # Rimuovi eventuali graffe residue
                g = g.strip("{} ")
                subitems = [x.strip() for x in g.split(',') if x.strip()]
                if subitems:
                    result.append(subitems)
        else:
            subitems = [x.strip() for x in inner.split(',') if x.strip()]
            if subitems:
                result.append(subitems)

    return result

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def compute_metrics(tp, fp, fn, exact=0, total=0):
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    accuracy = exact / (total) if total else 0
    return precision, recall, f1, accuracy

def evaluate_lists(true_list, pred_list):
    true_set = set(true_list)
    pred_set = set(pred_list)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    return tp, fp, fn
def evaluate_lists_why(true_list, pred_list):
    true_parsed = parse_why_item(true_list)
    pred_parsed = parse_why_item(pred_list)

    # Converti in set di tuple
    true_set = set(tuple(x) if isinstance(x, list) else (x,) for x in true_parsed)
    pred_set = set(tuple(x) if isinstance(x, list) else (x,) for x in pred_parsed)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    #print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    #print(f"True set: {true_set}, Pred set: {pred_set}")

    return tp, fp, fn

def main():
    gt_data = load_json("tpch/ground_truthTpch.json")
    question_types = load_json("tpch/questions.json")

    pred_folder = "tpch/outputs_llama70b/iterative/"
    global_metrics_file = os.path.join(pred_folder, "global_metrics_iterative_kdin_5rounds.csv")
    type_metrics_file = os.path.join(pred_folder, "metrics_by_type_iterative_kdin_5rounds.csv")
    iteration_metrics_file = os.path.join(pred_folder, "metrics_per_iteration_kdin_5rounds.csv")

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
            global_writer.writerow(["Category", "Iteration", "Precision", "Recall", "F1", "Accuracy"])
        if write_header_type:
            type_writer.writerow([
                "Question Type", "Iteration",
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

        pred_file = os.path.join(pred_folder, f"outputs_llama70b_iterative_kdin_5rounds_WHY.json")
        ungrouped_pred_data = load_json(pred_file)
        pred_data = group_predictions_by_question(ungrouped_pred_data)
        assert len(gt_data) == len(pred_data), "Mismatch in number of questions"

        metrics_by_type = defaultdict(lambda: {
            "tp_ans": 0, "fp_ans": 0, "fn_ans": 0,
            "tp_expl": 0, "fp_expl": 0, "fn_expl": 0,
            "exact_ans": 0, "exact_expl": 0,
            "count": 0
        })

        metrics_by_iter = {
            i: {
                "answer_tp": 0, "answer_fp": 0, "answer_fn": 0, "answer_exact": 0,
                "expl_tp": 0, "expl_fp": 0, "expl_fn": 0, "expl_exact": 0
            } for i in range(1,6)
        }

        metrics_by_type_and_iter = defaultdict(lambda: {
            i: {
                "tp_ans": 0, "fp_ans": 0, "fn_ans": 0, "exact_ans": 0,
                "tp_expl": 0, "fp_expl": 0, "fn_expl": 0, "exact_expl": 0,
                "count": 0
            } for i in range(1,6)
        })

        questions = list(question_types.keys())

        for idx, gt in enumerate(gt_data):
            question = questions[idx]
            q_type = question_types.get(question, "unknown")
            true_answer = gt["answer"]
            if not isinstance(true_answer, list):
                true_answer = [str(true_answer)]
            else:
                true_answer = [str(x) for x in true_answer]
            true_expl = gt["why"]
            '''
            true_expl = set()
            if isinstance(gt["why"], str):
                true_expl.update(s.strip("{} ") for s in gt["why"].split("}}") if s.strip())
            elif isinstance(gt["why"], list):
                for s in gt["why"]:
                    true_expl.update(ss.strip("{} ") for ss in s.split("}}") if ss.strip())
            '''
            #print(f"True: '{true_answer}'")  
            preds = pred_data[question]
            

            for pred in preds:
                iteration = pred["iteration"]
                raw_answer = pred.get("answer", "")
                
                pred_answer = []
                pred_expl = []

                if isinstance(raw_answer, dict):
                    pred_answer = [str(x).strip() for x in raw_answer.get("answer", [])]
                    pred_expl = [str(x).strip() for x in raw_answer.get("why", [])]

                elif isinstance(raw_answer, str):
                    try:
                        # Pulisci eventuale risposta testuale prima del JSON
                        matches = re.findall(r'\{[\s\S]*?\}', raw_answer)
                        for m in reversed(matches):  # prova a partire dall'ultimo
                            print(m)
                            try:
                                parsed = json.loads(m)
                                if "answer" in parsed:
                                    pred_answer = [str(x).strip() for x in parsed.get("answer", [])]
                                    pred_expl = [str(x).strip() for x in parsed.get("why", [])]
                                    break
                            except json.JSONDecodeError:
                                continue
                    except Exception:
                        pred_answer = []
               # print(f"Pred: '{pred_answer, pred_expl}'")
            

                tp_ans, fp_ans, fn_ans = evaluate_lists(true_answer, pred_answer)
                exact_answer = 1 if set(true_answer) == set(pred_answer) else 0
                ans_prec, ans_rec, ans_f1, _ = compute_metrics(tp_ans, fp_ans, fn_ans)

                tp_expl, fp_expl, fn_expl = evaluate_lists_why(true_expl, pred_expl)
                exact_expl = 1 if (true_expl == pred_expl) else 0
                expl_prec, expl_rec, expl_f1, _ = compute_metrics(tp_expl, fp_expl, fn_expl)

                iter_writer.writerow([
                    question, iteration, q_type,
                    f"{ans_prec:.4f}", f"{ans_rec:.4f}", f"{ans_f1:.4f}", exact_answer,
                    f"{expl_prec:.4f}", f"{expl_rec:.4f}", f"{expl_f1:.4f}", exact_expl
                ])

                # Aggregazione per tipo
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

                # Aggregazione globale per iterazione
                m_iter = metrics_by_iter[iteration]
                m_iter["answer_tp"] += tp_ans
                m_iter["answer_fp"] += fp_ans
                m_iter["answer_fn"] += fn_ans
                m_iter["answer_exact"] += exact_answer
                m_iter["expl_tp"] += tp_expl
                m_iter["expl_fp"] += fp_expl
                m_iter["expl_fn"] += fn_expl
                m_iter["expl_exact"] += exact_expl

                # Aggregazione per tipo per iterazione
                m_type_iter = metrics_by_type_and_iter[q_type][iteration]
                m_type_iter["count"] += 1
                m_type_iter["tp_ans"] += tp_ans
                m_type_iter["fp_ans"] += fp_ans
                m_type_iter["fn_ans"] += fn_ans
                m_type_iter["exact_ans"] += exact_answer
                m_type_iter["tp_expl"] += tp_expl
                m_type_iter["fp_expl"] += fp_expl
                m_type_iter["fn_expl"] += fn_expl
                m_type_iter["exact_expl"] += exact_expl

        # Scrivi metriche globali per iterazione
        for i in range(1,6):
            m = metrics_by_iter[i]
            a_prec, a_rec, a_f1, a_acc = compute_metrics(m["answer_tp"], m["answer_fp"], m["answer_fn"], m["answer_exact"], len(gt_data))
            e_prec, e_rec, e_f1, e_acc = compute_metrics(m["expl_tp"], m["expl_fp"], m["expl_fn"], m["expl_exact"], len(gt_data))

            global_writer.writerow(["Answer", i, f"{a_prec:.4f}", f"{a_rec:.4f}", f"{a_f1:.4f}", f"{a_acc:.4f}"])
            global_writer.writerow(["Explanation", i, f"{e_prec:.4f}", f"{e_rec:.4f}", f"{e_f1:.4f}", f"{e_acc:.4f}"])

        # Scrivi metriche per tipo per iterazione
        for q_type, iters in metrics_by_type_and_iter.items():
            for i in range(1,6):
                stats = iters[i]
                ans_m = compute_metrics(stats["tp_ans"], stats["fp_ans"], stats["fn_ans"], stats["exact_ans"], stats["count"])
                expl_m = compute_metrics(stats["tp_expl"], stats["fp_expl"], stats["fn_expl"], stats["exact_expl"], stats["count"])

                type_writer.writerow([
                    q_type, i,
                    f"{ans_m[0]:.4f}", f"{ans_m[1]:.4f}", f"{ans_m[2]:.4f}", f"{ans_m[3]:.4f}",
                    f"{expl_m[0]:.4f}", f"{expl_m[1]:.4f}", f"{expl_m[2]:.4f}", f"{expl_m[3]:.4f}",
                    stats["count"]
                ])

        print(f"Metriche calcolate e scritte per {len(gt_data)} domande.")
        print(f"Risultati salvati in {global_metrics_file}, {type_metrics_file} e {iteration_metrics_file}")
if __name__ == "__main__":
    main()
