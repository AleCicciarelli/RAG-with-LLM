import json
import csv
import os
from collections import defaultdict

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
    gt_data = load_json("tpch/ground_truthTpch.json")
    question_types = load_json("tpch/questions.json")

    # Cartella contenente i file di output predetti
    pred_folder = "tpch/outputs_mixtral8x7b/full_context/"
    global_metrics_file = os.path.join(pred_folder, "global_metrics_FC_ollama_cleaned.csv")
    type_metrics_file = os.path.join(pred_folder, "metrics_by_type_FC_ollama_cleaned.csv")

    # Scrivi header CSV solo se i file non esistono
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
                "Explanation Precision", "Explanation Recall", "Explanation F1", "Explanation Accuracy",
                "Count"
            ])

       
       
        pred_file = os.path.join(pred_folder, f"outputs_mixtral8x7bCleaned.json")


        pred_data = load_json(pred_file)
        
        assert len(gt_data) == len(pred_data), "Mismatch in number of questions"
           

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

        for gt, pred in zip(gt_data, pred_data):
            question = pred["question"]
            q_type = question_types.get(question, "unknown")

            # Normalizza risposte (sempre lista di stringhe)
            true_answer = gt["answer"]
            if not isinstance(true_answer, list):
                true_answer = [str(true_answer)]
            else:
                true_answer = [str(x) for x in true_answer]

            try:
                pred_answer_raw = pred.get("answer", [])
                if isinstance(pred_answer_raw, list) and len(pred_answer_raw) > 0 and isinstance(pred_answer_raw[0], dict):
                    pred_answer = [str(x) for x in pred_answer_raw[0].get("answer", [])]
                else:
                    raise ValueError("Invalid prediction format")
            except Exception as e:
                # In caso di errore, considera la risposta completamente sbagliata
                print(f"Errore nel parsing della risposta per la domanda '{question}': {e}")
                pred_answer = []

            # Calcola TP, FP, FN per risposta
            tp_ans, fp_ans, fn_ans = evaluate_lists(true_answer, pred_answer)

            # Calcola se risposta esatta (tutti corrispondono)
            exact_answer = 1 if (set(true_answer) == set(pred_answer)) else 0

            # Estrai spiegazioni
            true_expl_raw = gt["why"]
            true_expl = set()
            if isinstance(true_expl_raw, str):
                # Può essere una stringa singola o multipla separata da "}}"
                parts = [s.strip("{} ") for s in true_expl_raw.split("}}") if s.strip()]
                true_expl.update(parts)
            elif isinstance(true_expl_raw, list):
                for s in true_expl_raw:
                    parts = [ss.strip("{} ") for ss in s.split("}}") if ss.strip()]
                    true_expl.update(parts)

            try:
                pred_expl_raw = pred.get("answer", [])
                if isinstance(pred_expl_raw, list) and len(pred_expl_raw) > 0 and isinstance(pred_expl_raw[0], dict):
                    pred_expl = set(x.strip("{} ") for x in pred_expl_raw[0].get("why", []))
                else:
                    raise ValueError("Invalid prediction format")
            except Exception as e:
                # In caso di errore, considera la spiegazione completamente sbagliata
                print(f"Errore nel parsing della spiegazione per la domanda '{question}': {e}")
                pred_expl = set()

            # Calcola TP, FP, FN per spiegazione
            # Valuta solo se risposta corretta (intersection non vuota)
            #if tp_ans > 0:
            tp_expl, fp_expl, fn_expl = evaluate_lists(true_expl, pred_expl)
            #else:
            #    tp_expl = fp_expl = fn_expl = 0

            # Calcola se spiegazione esatta
            exact_expl = 1 if (true_expl == pred_expl and exact_answer) else 0

            # Aggiorna metriche globali
            answer_tp += tp_ans
            answer_fp += fp_ans
            answer_fn += fn_ans
            answer_exact += exact_answer

            expl_tp += tp_expl
            expl_fp += fp_expl
            expl_fn += fn_expl
            expl_exact += exact_expl

            # Aggiorna metriche per tipo domanda
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

        # Calcola metriche globali
        ans_prec, ans_rec, ans_f1, ans_acc = compute_metrics(
            answer_tp, answer_fp, answer_fn, answer_exact, len(gt_data)
        )
        expl_prec, expl_rec, expl_f1, expl_acc = compute_metrics(
            expl_tp, expl_fp, expl_fn, expl_exact, len(gt_data)
        )

        # Scrivi metriche globali nel CSV
        global_writer.writerow(["Answer", f"{ans_prec:.4f}", f"{ans_rec:.4f}", f"{ans_f1:.4f}", f"{ans_acc:.4f}"])
        global_writer.writerow(["Explanation", f"{expl_prec:.4f}", f"{expl_rec:.4f}", f"{expl_f1:.4f}", f"{expl_acc:.4f}"])

        # Scrivi metriche per tipo domanda
        for q_type, stats in metrics_by_type.items():
            ans_m = compute_metrics(stats["tp_ans"], stats["fp_ans"], stats["fn_ans"], stats["exact_ans"], stats["count"])
            expl_m = compute_metrics(stats["tp_expl"], stats["fp_expl"], stats["fn_expl"], stats["exact_expl"], stats["count"])

            type_writer.writerow([
                q_type,
                f"{ans_m[0]:.4f}", f"{ans_m[1]:.4f}", f"{ans_m[2]:.4f}", f"{ans_m[3]:.4f}",
                f"{expl_m[0]:.4f}", f"{expl_m[1]:.4f}", f"{expl_m[2]:.4f}", f"{expl_m[3]:.4f}",
                stats["count"]
            ])

#stampa nome del file di output
            print(f"Metriche calcolate e scritte per {len(gt_data)} domande.")
            print(f"Risultati salvati in {global_metrics_file} e {type_metrics_file}")

if __name__ == "__main__":
    main()
