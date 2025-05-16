import os
import pandas as pd
from glob import glob
from collections import defaultdict

def load_global_metrics(path):
    df = pd.read_csv(path, index_col="Category")
    return df.loc["Answer"].to_dict(), df.loc["Explanation"].to_dict()

def load_type_metrics(path):
    df = pd.read_csv(path)
    df.set_index("question_type", inplace=True)
    return df

def average_type_metrics(dfs):
    df_concat = pd.concat(dfs)
    return df_concat.groupby(df_concat.index).mean()

def average_global_metrics(dicts):
    df = pd.DataFrame(dicts)
    return df.mean().to_dict()

def process_llm(llm_name, folder_paths):
    global_answers, global_explanations = [], []
    type_metric_dfs = []

    for path in folder_paths:
        global_file = glob(os.path.join(path, "metrics_global_*.csv"))[0]
        type_file = glob(os.path.join(path, "metrics_by_type_*.csv"))[0]

        a, e = load_global_metrics(global_file)
        global_answers.append(a)
        global_explanations.append(e)

        type_metric_dfs.append(load_type_metrics(type_file))

    # Compute averages
    avg_global_ans = average_global_metrics(global_answers)
    avg_global_expl = average_global_metrics(global_explanations)
    avg_type_metrics = average_type_metrics(type_metric_dfs)

    # Save outputs
    os.makedirs("output", exist_ok=True)
    pd.DataFrame([avg_global_ans, avg_global_expl], index=["Answer", "Explanation"]).to_csv(f"output/global_avg_{llm_name}.csv")
    avg_type_metrics.to_csv(f"output/by_type_avg_{llm_name}.csv")
    print(f"Saved results for {llm_name}")

def main():
    base_folders = [d for d in os.listdir() if os.path.isdir(d) and ("llama70b" in d or "mistral" in d)]
    llm_groups = defaultdict(list)

    for folder in base_folders:
        if "llama70b" in folder:
            llm_groups["llama70b"].append(folder)
        elif "mistral" in folder:
            llm_groups["mistral"].append(folder)

    for llm, folders in llm_groups.items():
        process_llm(llm, folders)

if __name__ == "__main__":
    main()
