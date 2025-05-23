import os
import pandas as pd

def average_per_type(llm_dir):
    dfs = []
    for fname in os.listdir(llm_dir):
        if fname.startswith("metrics_by_type") and fname.endswith(".csv") and "FCGEN" in fname:
            df = pd.read_csv(os.path.join(llm_dir, fname))
            dfs.append(df)
    if not dfs:
        print(f"Nessun file 'metrics_by_type' trovato in {llm_dir}")
        return
    df_all = pd.concat(dfs)
    metric_cols = [col for col in df_all.columns if col not in ['question_type', '#questions']]
    avg_per_type = df_all.groupby('question_type')[metric_cols].mean()
    avg_per_type.to_csv(os.path.join(llm_dir, 'average_per_type_FCGEN.csv'))
    print(f"Salvato average_per_type.csv in {llm_dir}")

def average_global(llm_dir):
    dfs = []
    for fname in os.listdir(llm_dir):
        if fname.startswith("metrics_global") and fname.endswith(".csv") and "FCGEN" in fname:
            df = pd.read_csv(os.path.join(llm_dir, fname), index_col=0)
            dfs.append(df)
    if not dfs:
        print(f"Nessun file 'metrics_global' trovato in {llm_dir}")
        return
    df_all = pd.concat(dfs)
    avg_global = df_all.groupby(df_all.index).mean()
    avg_global.to_csv(os.path.join(llm_dir, 'average_global_FCGEN.csv'))
    print(f"Salvato average_global.csv in {llm_dir}")

if __name__ == "__main__":
    results_root = "./results"
    for llm_name in os.listdir(results_root):
        llm_dir = os.path.join(results_root, llm_name)
        if os.path.isdir(llm_dir):
            average_per_type(llm_dir)
            average_global(llm_dir)