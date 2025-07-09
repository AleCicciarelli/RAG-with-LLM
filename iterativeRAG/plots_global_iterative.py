import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Percorsi ===
base_path = 'tpch/'
output_path = os.path.join(base_path, 'outputs_plots_iterative/no_why/global/')
os.makedirs(output_path, exist_ok=True)

# === Config ===
models = ['llama8b', 'mixtral8x7b', 'llama70b']
metrics_to_plot = ['F1']

variant_colors = {
    'FC': 'blue',
    'ksemidin': 'orange',
    'k10': 'green'
}

variant_linestyles = {
    'FC': '-',                  # linea continua
    'ksemidin': (0, (5, 5)),    # tratteggiata lunga
    'k10': (0, (1, 1))          # tratteggiata corta
}

# === Caricamento dati ===
all_data = []
for model in models:
    variants = {
        'k10': f'{base_path}outputs_{model}/iterative/global_metrics_iterative_k10_5rounds_NOWHY.csv',
        'ksemidin': f'{base_path}outputs_{model}/iterative/global_metrics_iterative_ksemidin_5rounds_NOWHY.csv',
        'FC': f'{base_path}full_context/global_metrics_{model}_no_why_FC.csv'

    }

    for variant, file_path in variants.items():
        print(f"Loading data for {model} - {variant} from {file_path}")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Model'] = model
            df['Variant'] = variant
            # Assicuriamoci che Iteration sia numerico, NaN per FC
            all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)
if 'Iteration' in df_all.columns:
    df_all['Iteration'] = pd.to_numeric(df_all['Iteration'], errors='coerce')
df_all.rename(columns={'Question Type': 'QuestionType'}, inplace=True)
# === Plot ===
sns.set_theme(style="whitegrid", font_scale=1.0)

for metric in metrics_to_plot:
    # Consideriamo solo dati iterativi per FacetGrid (escludiamo FC dalle linee tracciate)
    df_iterative = df_all[df_all['Variant'] != 'FC'].copy()

    # FacetGrid con righe = Model, colonne = Category (Answer, Explanation)
    g = sns.FacetGrid(
        df_iterative,
        row='Model',
        #col='Category',
        height=4,
        aspect=1.3,
        sharey=True
    )

    def plot_iterative(data, **kwargs):
        ax = plt.gca()
        for variant in ['ksemidin', 'k10']:
            df_variant = data[data['Variant'] == variant]
            if df_variant.empty:
                continue
            ax.plot(
                df_variant['Iteration'],
                df_variant[metric],
                label=variant,
                linestyle=variant_linestyles[variant],
                marker='o',
                color=variant_colors[variant]
            )
            for _, row in df_variant.iterrows():
                ax.text(
                    row['Iteration'],
                    row[metric],
                    f"{row[metric]:.4f}",
                    fontsize=8,
                    ha='center',
                    va='bottom',
                    color=variant_colors[variant]
                )

    g.map_dataframe(plot_iterative)

    # Aggiungiamo linea orizzontale FC per ogni combinazione Model x Category
    #for ax, (model, category) in zip(g.axes.flat, [(r, c) for r in g.row_names for c in g.col_names]):
    #    df_fc = df_all[(df_all['Model'] == model) & (df_all['Variant'] == 'FC')] & (df_all['Category'] == category)]
    for ax, model in zip(g.axes.flat, g.row_names):
        df_fc = df_all[(df_all['Model'] == model) & (df_all['Variant'] == 'FC')]
        if not df_fc.empty:
            y_val = df_fc[metric].values[0]
            iterations = sorted(df_iterative['Iteration'].dropna().unique())
            if not iterations:
                iterations = [0]
            ax.plot(
                iterations,
                [y_val] * len(iterations),
                label='FC',
                linestyle=variant_linestyles['FC'],
                marker='o',
                color=variant_colors['FC']
            )
            for x in iterations:
                ax.text(
                    x,
                    y_val,
                    f"{y_val:.4f}",
                    fontsize=8,
                    ha='center',
                    va='bottom',
                    color=variant_colors['FC']
                )

    # Legenda (solo nel primo subplot)
    handles = [
        plt.Line2D([], [], color=variant_colors[v], linestyle=variant_linestyles[v], marker='o', label=v)
        for v in ['ksemidin', 'k10', 'FC']
    ]
    g.axes[0, 0].legend(handles=handles, title='Variant', loc='best')

    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    g.set_axis_labels("Iteration", metric)
    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle(f"{metric} over Iterations - Models × Category × Variant", fontsize=16)

    filename = f"{metric}_iterative_global_no_why.png"
    g.savefig(os.path.join(output_path, filename))
    plt.close()
