import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Imposta percorso principale dove ci sono le cartelle dei modelli
base_path = 'tpch/outputs_'
output_path = 'tpch/plots_ollama_per_k'
os.makedirs(output_path, exist_ok=True)

# Lista dei modelli = nomi delle cartelle
models = ['llama8b', 'llama70b', 'mixtral8x7b']
metrics_to_plot = ['F1', 'Accuracy']

# Colori per modello
palette = {'llama8b': 'blue', 'llama70b': 'green', 'mixtral8x7b': 'orange'}

# Caricamento dati
all_data = []
for model in models:
    file_path = os.path.join(base_path + model, 'global_metrics_all_k.csv')
    df = pd.read_csv(file_path)
    df['Model'] = model
    all_data.append(df)

df_all = pd.concat(all_data)

# Plot
sns.set_theme(style="whitegrid", font_scale=1.1)

for category in ['Answer', 'Explanation']:
    df_cat = df_all[df_all['Category'] == category]
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(data=df_cat, x='K', y=metric, hue='Model', palette=palette)

        # Evidenzia il k ottimale per ogni modello
        for model in models:
            df_model = df_cat[df_cat['Model'] == model]
            best_row = df_model.loc[df_model[metric].idxmax()]
            plt.plot(best_row['K'], best_row[metric], 'o', 
                     color=palette[model], markersize=8, 
                     label=f'Best {model} (k={int(best_row["K"])} {metric}={best_row[metric]:.3f})')

        plt.title(f'{metric} per K - Category: {category}')
        plt.xlabel('Top-K')
        plt.ylabel(metric)
        plt.legend(title='Model', loc='best')
        plt.grid(True)
        plt.tight_layout()

        # Salvataggio grafico
        filename = f"{metric}_{category}_G.png".replace(" ", "_")
        plt.savefig(os.path.join(output_path, filename))
        plt.close()
# Stampa completamento
print("Plot saved in:", output_path)
