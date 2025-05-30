import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIG ===

metrics = ['Answer F1', 'Explanation F1']

base_path = 'outputs_ollama_'
output_dir = 'plots_ollama_per_k/plots_ollama_per_k_questionType'
os.makedirs(output_dir, exist_ok=True)
# Lista dei modelli = nomi delle cartelle
models = ['llama8b', 'llama70b', 'mixtral8x7b']

# Colori per modello
palette = {'llama8b': 'blue', 'llama70b': 'green', 'mixtral8x7b': 'orange'}
# === CREA CARTELLA DI OUTPUT ===

# === CARICAMENTO DATI ===
all_data = []
for model in models:
    file_path = os.path.join(base_path + model, 'metrics_by_type_all_k.csv')
    df = pd.read_csv(file_path)
    df['Model'] = model
    all_data.append(df)

df_all = pd.concat(all_data)
question_types = df_all['Question Type'].unique()
sns.set_theme(style="whitegrid", font_scale=1.1)

# === PLOT PER OGNI QUESTION TYPE E METRICA ===
for qtype in question_types:
    df_q = df_all[df_all['Question Type'] == qtype]
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for model in models:
            df_model = df_q[df_q['Model'] == model]
            sns.lineplot(data=df_model, x='K', y=metric, label=model, color=palette[model])

            # Punto con valore massimo
            best_row = df_model.loc[df_model[metric].idxmax()]
            plt.plot(best_row['K'], best_row[metric], 'o', color=palette[model], markersize=10,
                     label=f'{model} Best K={int(best_row["K"])} {metric}={best_row[metric]:.2f}')
        
        plt.title(f'{metric} vs K\nQuestion type: {qtype}')
        plt.xlabel('Top-K')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Saving the plot
        qtype_slug = qtype.replace(' ', '_').lower()
        metric_slug = metric.replace(' ', '_').lower()
        filename = f"{qtype_slug}_{metric_slug}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
# === STAMPA COMPLETAMENTO ===
print("Plots saved in:", output_dir)