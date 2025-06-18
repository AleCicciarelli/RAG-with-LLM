import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIG ===

base_path = 'iterativeRag/outputs_'
output_dir = os.path.join(base_path, 'plots_ollama_per_type_iterations/')
os.makedirs(output_dir, exist_ok=True)

models = ['llama8b', 'llama70b', 'mixtral8x7b']
palette = {'llama8b': 'blue', 'llama70b': 'green', 'mixtral8x7b': 'orange'}

# === CARICAMENTO DATI ===

all_data = []
for model in models:
    file_path = os.path.join(base_path + model + '/metrics_by_type_iterativefaiss.csv')
    df = pd.read_csv(file_path)
    df['Model'] = model
    df['Iteration'] = df['Iteration'].astype(int)
    all_data.append(df)

df_all = pd.concat(all_data)
sns.set_theme(style="whitegrid", font_scale=1.1)

# === RISTRUTTURA DATI ===

df_melted = df_all.melt(
    id_vars=['Model', 'Question Type', 'Iteration'],
    value_vars=['Answer F1', 'Explanation F1'],
    var_name='Metric',
    value_name='F1'
)
df_melted['Metric'] = df_melted['Metric'].str.replace(' F1', '', regex=False)

# === PLOT ===

question_types = df_melted['Question Type'].unique()
metrics = ['Answer', 'Explanation']

for qtype in question_types:
    for metric in metrics:
        subset = df_melted[(df_melted['Question Type'] == qtype) & (df_melted['Metric'] == metric)]

        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(
            data=subset,
            x='Iteration',
            y='F1',
            hue='Model',
            palette=palette,
            marker='o'
        )

        # Etichette sui punti
        for line in ax.lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            for x, y in zip(x_data, y_data):
                ax.text(x, y + 0.01, f'{y:.2f}', ha='center', color=line.get_color())

        plt.title(f'{metric} F1 over Iterations - {qtype}')
        plt.xlabel('Iteration')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1.05)
        plt.xticks(sorted(df_all['Iteration'].unique()))
        plt.legend(title='Model')
        plt.tight_layout()

        # Salvataggio
        fname = f"{qtype}_{metric}_F1_by_model.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

print("Plot per tipo di domanda e metrica salvati in:", output_dir)
