import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Imposta percorso principale e di output
base_path = 'tpch/outputs_'
output_path = 'tpch/plots_FC'
os.makedirs(output_path, exist_ok=True)

# Lista dei modelli = nomi delle cartelle
models = ['llama8b', 'llama70b', 'mixtral8x7b']
metrics_to_plot = ['Precision', 'Recall', 'F1', 'Accuracy']

# Colori per modello
palette = {'llama8b': 'blue', 'llama70b': 'green', 'mixtral8x7b': 'orange'}

# Caricamento dati
all_data = []
for model in models:
    file_path = os.path.join(base_path + model +  '/full_context/global_metrics_FC.csv')
    df = pd.read_csv(file_path)
    df['Model'] = model
    all_data.append(df)

df_all = pd.concat(all_data)

# Plot per ogni metrica
sns.set_theme(style="whitegrid", font_scale=1.1)

for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_all, x='Category', y=metric, hue='Model', palette=palette)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge')

    plt.title(f'{metric} for model and category')
    plt.xlabel('Category')
    plt.ylabel(metric)
    plt.legend(title='Model')
    plt.tight_layout()

    # Salvataggio grafico
    filename = f"{metric}_barplot.png".replace(" ", "_")
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

print("Plot salvati in:", output_path)
