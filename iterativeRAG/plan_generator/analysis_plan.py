import json
import csv
import os
import regex as re
from collections import defaultdict
import string
from typing import Optional

def find_last_balanced_json(text: str) -> Optional[str]:
    stack = []
    start_idx = None
    for i, c in enumerate(text):
        if c == '{':
            if not stack:
                start_idx = i
            stack.append(c)
        elif c == '}':
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = text[start_idx:i+1]
                    # Verifica se contiene "answer"
                    if '"answer"' in candidate or "'answer'" in candidate:
                        return candidate
    return None
def extract_answer(raw_pred):
    if not raw_pred:
        return []

    # Caso 1: già un dizionario con "answer"
    if isinstance(raw_pred, dict):
        if "answer" in raw_pred and isinstance(raw_pred["answer"], list):
            return [str(x).strip() for x in raw_pred["answer"] if str(x).strip()]
        else:
            return []

    # Caso 2: già una lista pulita
    if isinstance(raw_pred, list) and all(isinstance(x, (str, int)) for x in raw_pred):
        return [str(x).strip() for x in raw_pred if str(x).strip()]

    # Caso 3: è una stringa o lista contenente stringa JSON
    if isinstance(raw_pred, list) and len(raw_pred) == 1 and isinstance(raw_pred[0], str):
        raw_pred_str = raw_pred[0]
    elif isinstance(raw_pred, str):
        raw_pred_str = raw_pred
    else:
        return []

    json_str = find_last_balanced_json(raw_pred_str)

    if json_str:
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict) and "answer" in obj and isinstance(obj["answer"], list):
                return [str(x).strip() for x in obj["answer"] if str(x).strip()]
        except json.JSONDecodeError:
            try:
                json_fixed = json_str.replace("'", '"')
                obj = json.loads(json_fixed)
                if isinstance(obj, dict) and "answer" in obj and isinstance(obj["answer"], list):
                    return [str(x).strip() for x in obj["answer"] if str(x).strip()]
            except:
                pass

    # fallback: cerca lista tra []
    fallback_match = re.search(r'\[(.*?)\]', raw_pred_str)
    if fallback_match:
        items = [x.strip().strip('"').strip("'") for x in fallback_match.group(1).split(',') if x.strip()]
        if items:
            return items

    if raw_pred_str.strip():
        return [raw_pred_str.strip()]

    return []

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
    pred_set = set(pred_list or [])
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    return tp, fp, fn

def clean_answer(ans):
    def strip_punct(s):
        return s.rstrip(string.punctuation).strip()

    if isinstance(ans, list):
        return [strip_punct(str(a)) for a in ans if isinstance(a, (str, int))]
    elif isinstance(ans, str):
        return [strip_punct(ans)]
    return []

def main():
    gt_data = load_json("ground_truth2.json")
    question_types = load_json("questions.json")
    pred_folder = "iterativeRAG/plan_generator/outputs_deepseek70b"
    pred_file = os.path.join(pred_folder, "k10.json")

    global_metrics_file = os.path.join(pred_folder, "global_metrics_deepseek70b_plan_k10.csv")
    type_metrics_file = os.path.join(pred_folder, "metrics_by_type_deepseek70b_plan_k10.csv")


    write_header_global = not os.path.exists(global_metrics_file)
    write_header_type = not os.path.exists(type_metrics_file)

    with open(global_metrics_file, "a", newline="", encoding="utf-8") as f_global, \
         open(type_metrics_file, "a", newline="", encoding="utf-8") as f_type:

        global_writer = csv.writer(f_global)
        type_writer = csv.writer(f_type)

        if write_header_global:
            global_writer.writerow(["Category", "Precision", "Recall", "F1", "Accuracy"])
        if write_header_type:
            type_writer.writerow([
                "Question Type",
                "Answer Precision", "Answer Recall", "Answer F1", "Answer Accuracy",
                #"Explanation Precision", "Explanation Recall", "Explanation F1", "Explanation Accuracy",
                "Count"
            ])

        pred_steps = load_json(pred_file)

        # Raggruppa gli step per domanda
        grouped_preds = defaultdict(list)
        for p in pred_steps:
            grouped_preds[p["question"]].append(p)
        print(f"📄 Trovate {len(grouped_preds)} domande nel file di predizioni.")
        # Indicizza il ground truth
        gt_dict = {ex["query"]: ex for ex in gt_data}

        metrics_by_type = defaultdict(lambda: {
            "tp_ans": 0, "fp_ans": 0, "fn_ans": 0,
            "tp_expl": 0, "fp_expl": 0, "fn_expl": 0,
            "exact_ans": 0, "exact_expl": 0, "count": 0
        })

        answer_tp = answer_fp = answer_fn = answer_exact = 0
        expl_tp = expl_fp = expl_fn = expl_exact = 0

        for idx, (question, steps) in enumerate(grouped_preds.items()):
            steps_sorted = sorted(steps, key=lambda x: x.get("step", 0))
            final = steps_sorted[-1]

            gt_key = f"query{(idx+1):02}.csv"
            gt = gt_dict.get(gt_key)
            if not gt:
                print(f"❗ Domanda non trovata nel ground truth: {gt_key}")
                continue

            q_type = question_types.get(question, "unknown")

            true_answer = [str(x) for x in gt.get("answer", [])]
            pred_answer_raw = final.get("answer", [])#.get("answer", [])
            
            
            pred_answer = extract_answer(pred_answer_raw)
            pred_answer = [str(x).strip() for x in pred_answer] if isinstance(pred_answer, list) else [str(pred_answer)]
            pred_answer = clean_answer(pred_answer)


            print(f"\n📄 {gt_key}")
            print(f"❓ Question: {question}")
                        
            print("RAW prediction:", pred_answer_raw)
            print(f"✅ True answer: {true_answer}")
            print(f"🧪 Pred answer: {pred_answer}")
            tp_ans, fp_ans, fn_ans = evaluate_lists(true_answer, pred_answer)
            print(f"TP: {tp_ans}, FP: {fp_ans}, FN: {fn_ans}")
            print(f"Exact match: {set(true_answer) == set(pred_answer)}")
            exact_answer = 1 if set(true_answer) == set(pred_answer) else 0

            true_expl = gt.get("why", [])
            #pred_expl = final.get("answer", {}).get("why", [])
            #pred_expl = [str(x).strip() for x in pred_expl]

            #tp_expl, fp_expl, fn_expl = evaluate_lists_why(true_expl, pred_expl)
           # exact_expl = 1 if (true_expl == pred_expl and exact_answer) else 0

            answer_tp += tp_ans
            answer_fp += fp_ans
            answer_fn += fn_ans
            answer_exact += exact_answer

           # expl_tp += tp_expl
           # expl_fp += fp_expl
           # expl_fn += fn_expl
           # expl_exact += exact_expl

            m = metrics_by_type[q_type]
            m["count"] += 1
            m["tp_ans"] += tp_ans
            m["fp_ans"] += fp_ans
            m["fn_ans"] += fn_ans
            m["exact_ans"] += exact_answer
            #m["tp_expl"] += tp_expl
            #m["fp_expl"] += fp_expl
            #m["fn_expl"] += fn_expl
            #m["exact_expl"] += exact_expl


        ans_prec, ans_rec, ans_f1, ans_acc = compute_metrics(
            answer_tp, answer_fp, answer_fn, answer_exact, len(grouped_preds)
        )
        expl_prec, expl_rec, expl_f1, expl_acc = compute_metrics(
            expl_tp, expl_fp, expl_fn, expl_exact, len(grouped_preds)
        )

        global_writer.writerow(["Answer", f"{ans_prec:.4f}", f"{ans_rec:.4f}", f"{ans_f1:.4f}", f"{ans_acc:.4f}"])
        #global_writer.writerow(["Explanation", f"{expl_prec:.4f}", f"{expl_rec:.4f}", f"{expl_f1:.4f}", f"{expl_acc:.4f}"])

        for q_type, stats in metrics_by_type.items():
            ans_m = compute_metrics(stats["tp_ans"], stats["fp_ans"], stats["fn_ans"], stats["exact_ans"], stats["count"])
            #expl_m = compute_metrics(stats["tp_expl"], stats["fp_expl"], stats["fn_expl"], stats["exact_expl"], stats["count"])

            type_writer.writerow([
                q_type,
                f"{ans_m[0]:.4f}", f"{ans_m[1]:.4f}", f"{ans_m[2]:.4f}", f"{ans_m[3]:.4f}",
                #f"{expl_m[0]:.4f}", f"{expl_m[1]:.4f}", f"{expl_m[2]:.4f}", f"{expl_m[3]:.4f}",
                stats["count"]
            ])

        print(f"✔️ Valutazione completata su {len(grouped_preds)} domande.")
        print(f"📁 Salvate metriche in: {global_metrics_file} e {type_metrics_file}")

if __name__ == "__main__":
    main()
