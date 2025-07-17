import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Percorsi ===
base_path = 'iterativeRAG/plan_generator/'
output_path = os.path.join(base_path, 'outputs_plots/no_why/global/')
os.makedirs(output_path, exist_ok=True)

# === Config ===
models = ['llama8b', 'mixtral8x7b', 'llama70b', 'deepseek70b']
metrics_to_plot = ['F1']

variant_colors = {
    'FC': 'blue',
    'k10': 'green',
    'k20': 'orange'
}

# === Caricamento dati ===
all_data = []
for model in models:
    variants = {
        'k10': f'{base_path}outputs_{model}/global_metrics_{model}_plan_k10.csv',
        'k20': f'{base_path}outputs_{model}/global_metrics_{model}_plan_k20.csv',
        'FC': f'full_context/global_metrics_{model}_no_why_FC.csv'
    }

    for variant, file_path in variants.items():
        print(f"Loading data for {model} - {variant} from {file_path}")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Model'] = model
            df['Variant'] = variant
            all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)
df_all.rename(columns={'Question Type': 'QuestionType'}, inplace=True)

# === Plot ===
sns.set_theme(style="whitegrid", font_scale=1.0)

for metric in metrics_to_plot:
    for category in df_all['Category'].unique():
        df_cat = df_all[df_all['Category'] == category]

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=df_cat,
            x='Model',
            y=metric,
            hue='Variant',
            palette=variant_colors
        )

        # Etichette sopra le barre
        for p in ax.patches:
            height = p.get_height()
            ax.text(
                p.get_x() + p.get_width() / 2.,
                height + 0.01,
                f'{height:.4f}',
                ha="center"
            )

        plt.title(f'{metric} - {category}')
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.xlabel('Model')
        plt.legend(title='Variant')
        plt.tight_layout()

        filename = f"{metric}_{category}_barplot_global_no_iter.png"
        plt.savefig(os.path.join(output_path, filename))
        plt.close()
