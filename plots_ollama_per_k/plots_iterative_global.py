import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Percorsi
base_path = 'iterativeRag/outputs_'
output_path = base_path
os.makedirs(output_path, exist_ok=True)

# Modelli
models = ['llama8b', 'llama70b', 'mixtral8x7b']
metrics_to_plot = ['Precision', 'Recall', 'F1', 'Accuracy']

# Colori
palette = {'llama8b': 'blue', 'llama70b': 'green', 'mixtral8x7b': 'orange'}

# Caricamento e unione dati
all_data = []
for model in models:
    file_path = os.path.join(base_path + model + '/global_metrics_iterativefaiss.csv')
    df = pd.read_csv(file_path)
    df['Model'] = model
    all_data.append(df)

df_all = pd.concat(all_data)

# Converte 'Iteration' in int se necessario
df_all['Iteration'] = df_all['Iteration'].astype(int)

# Plot: una figura per ogni metrica e per ogni categoria (Answer / Explanation)
sns.set_theme(style="whitegrid", font_scale=1.1)

for metric in metrics_to_plot:
    for category in df_all['Category'].unique():
        plt.figure(figsize=(10, 6))
        subset = df_all[df_all['Category'] == category]
        ax = sns.lineplot(data=subset, x='Iteration', y=metric, hue='Model', palette=palette, marker='o')

        for line in ax.lines:
            for x_val, y_val in zip(line.get_xdata(), line.get_ydata()):
                ax.text(x_val, y_val + 0.005, f'{y_val:.2f}', color=line.get_color(), ha='center')

        plt.title(f'{metric} over Iterations ({category})')
        plt.xlabel('Iteration')
        plt.ylabel(metric)
        plt.xticks(sorted(df_all['Iteration'].unique()))
        plt.legend(title='Model')
        plt.tight_layout()

        filename = f"{metric}_{category}_lineplot_iterative3round.png".replace(" ", "_")
        plt.savefig(os.path.join(output_path, filename))
        plt.close()

print("Plot salvati in:", output_path)
