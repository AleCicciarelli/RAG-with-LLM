import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIG ===

metrics = ['Answer F1', 'Explanation F1']
base_path = 'iterativeRag/outputs_'
output_dir = base_path + 'plots_ollama_per_type/'
os.makedirs(output_dir, exist_ok=True)

models = ['llama8b', 'llama70b', 'mixtral8x7b']
palette = {'llama8b': 'blue', 'llama70b': 'green', 'mixtral8x7b': 'orange'}

# === CARICAMENTO DATI ===
all_data = []
for model in models:
    file_path = os.path.join(base_path + model + '/metrics_by_type_iterative.csv')
    df = pd.read_csv(file_path)
    df['Model'] = model
    all_data.append(df)

df_all = pd.concat(all_data)
sns.set_theme(style="whitegrid", font_scale=1.1)

# === PLOT ===
for metric in metrics:
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=df_all, x='Question Type', y=metric, hue='Model', palette=palette)
    
    # Etichette sulle barre
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge')

    plt.title(f'{metric} per question type')
    plt.xticks(rotation=30, ha='right')
    plt.xlabel('Question Type')
    plt.ylabel(metric)
    plt.legend(title='Model')
    plt.tight_layout()

    # Salvataggio
    metric_slug = metric.replace(' ', '_').lower()
    plt.savefig(os.path.join(output_dir, f"{metric_slug}_by_question_type_ollama_iterative.png"))
    plt.close()

# === COMPLETAMENTO ===
print("Bar plot salvati in:", output_dir)
