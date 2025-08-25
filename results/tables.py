import os
import pandas as pd

results_root = "./results"
llms = [d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]

header = (
    "\\begin{table}[ht]\n"
    "\\centering\n"
    "\\caption{Confronto delle medie per tipo tra diversi LLMs e metodi (FC vs FCGEN)}\n"
    "\\begin{tabular}{lcccccc|cccccc}\n"
    "\\toprule\n"
    " & \\multicolumn{6}{c|}{\\textbf{FC}} & \\multicolumn{6}{c}{\\textbf{FCGEN}} \\\\\n"
    "\\textbf{LLM} & Prec (A) & Rec (A) & Acc (A) & Prec (E) & Rec (E) & Acc (E)"
    " & Prec (A) & Rec (A) & Acc (A) & Prec (E) & Rec (E) & Acc (E) \\\\\n"
    "\\midrule"
)

print(header)

for llm in llms:
    llm_dir = os.path.join(results_root, llm)
    fc_path = os.path.join(llm_dir, "average_per_type.csv")
    fcgen_path = os.path.join(llm_dir, "average_per_type_FCGEN.csv")
    # Default values
    fc_vals = ["-"] * 6
    fcgen_vals = ["-"] * 6

    if os.path.exists(fc_path):
        df_fc = pd.read_csv(fc_path)
        # Calcola la media su tutte le tipologie di domanda
        fc_vals = [
            f"{df_fc['precision_ans'].mean():.4f}",
            f"{df_fc['recall_ans'].mean():.4f}",
            f"{df_fc['accuracy_ans'].mean():.4f}",
            f"{df_fc['precision_expl'].mean():.4f}",
            f"{df_fc['recall_expl'].mean():.4f}",
            f"{df_fc['accuracy_expl'].mean():.4f}",
        ]
    if os.path.exists(fcgen_path):
        df_fcgen = pd.read_csv(fcgen_path)
        fcgen_vals = [
            f"{df_fcgen['precision_ans'].mean():.4f}",
            f"{df_fcgen['recall_ans'].mean():.4f}",
            f"{df_fcgen['accuracy_ans'].mean():.4f}",
            f"{df_fcgen['precision_expl'].mean():.4f}",
            f"{df_fcgen['recall_expl'].mean():.4f}",
            f"{df_fcgen['accuracy_expl'].mean():.4f}",
        ]
    print(
        f"{llm} & "
        f"{' & '.join(fc_vals)} & "
        f"{' & '.join(fcgen_vals)} \\\\"
    )

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")