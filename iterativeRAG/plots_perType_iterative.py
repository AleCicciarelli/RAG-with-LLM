import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Percorsi
base_path = 'tpch/'
output_path = base_path + 'outputs_plots_iterative/no_why/per_type/'
os.makedirs(output_path, exist_ok=True)

# Config
models = ['llama8b', 'mixtral8x7b', 'llama70b']
metrics_to_plot = ['F1']  # metrica unica F1, divisa per Category (Answer/Explanation)

# Caricamento dati
all_data = []
for model in models:
    variants = {
        'k10': f'{base_path}outputs_{model}/iterative/metrics_by_type_iterative_k10_5rounds_NOWHY.csv',
        'ksemidin': f'{base_path}outputs_{model}/iterative/metrics_by_type_iterative_ksemidin_5rounds_NOWHY.csv',
        'FC': f'{base_path}full_context/metrics_by_type_{model}_no_why_FC.csv'
    }
    for variant, file_path in variants.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Model'] = model
            df['Variant'] = variant
            # Assumiamo che la colonna 'Question Type' esista e rinominiamola subito
            if 'Question Type' in df.columns:
                df.rename(columns={'Question Type': 'QuestionType'}, inplace=True)
            if 'Answer F1' in df.columns:
                df.rename(columns={'Answer F1': 'F1'}, inplace = True)
            # Assumiamo che le colonne Answer F1 e Explanation F1 esistano:
            # Ora trasformiamo i dati da wide a long per avere Category e F1 in colonne separate
            if 'Answer F1' in df.columns and 'Explanation F1' in df.columns:
                df_long = pd.melt(df,
                                  id_vars=[col for col in df.columns if col not in ['Answer F1', 'Explanation F1']],
                                  value_vars=['Answer F1', 'Explanation F1'],
                                  var_name='Category',
                                  value_name='F1')
                # Cambiamo i nomi Category da 'Answer F1' a 'Answer', 'Explanation F1' a 'Explanation'
                df_long['Category'] = df_long['Category'].str.replace(' F1', '')
                df = df_long
            all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# Converti Iteration in int, ignorando eventuali NaN (per FC)
if 'Iteration' in df_all.columns:
    df_all['Iteration'] = pd.to_numeric(df_all['Iteration'], errors='coerce')

sns.set_theme(style="whitegrid", font_scale=1.0)

variant_colors = {
    'FC': 'blue',
    'ksemidin': 'orange',
    'k10': 'green'
}

variant_linestyles = {
    'FC': '-',                  # linea continua (per FC sarà orizzontale)
    'ksemidin': (0, (5, 5)),        # tratteggiata lunga
    'k10': (0, (1, 1))          # tratteggiata corta
}

question_types = df_all['QuestionType'].unique()
for q_type in question_types:
    df_q = df_all[df_all['QuestionType'] == q_type]
    df_plot = df_q[df_q['Variant'] != 'FC']

    if df_plot.empty:
        print(f"No data for QuestionType={q_type} skipping plot.")
        continue

    for metric in metrics_to_plot:
        if metric not in df_plot.columns:
            print(f"Metric '{metric}' not in data for QuestionType={q_type}, skipping.")
            continue

        print(f"Plotting {metric} for QuestionType={q_type}")

        g = sns.FacetGrid(
            df_plot,
            row='Model',
            #col='Category',  # colonna per Answer / Explanation
            height=4,
            aspect=1.5,
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

        # Aggiungi la linea FC come orizzontale su tutte le sottotrame (Model × Category)
        #for ax, (model, category) in zip(g.axes.flat, 
        #                                [(m, c) for m in g.row_names for c in g.col_names]):
        
        #    df_fc = df_q[(df_q['Model'] == model) & (df_q['Category'] == category) & (df_q['Variant'] == 'FC')]
        for ax, model in zip(g.axes.flat, g.row_names):
            df_fc = df_q[(df_q['Model'] == model) & (df_q['Variant'] == 'FC')]
            if not df_fc.empty:
                y_val = df_fc[metric].values[0]
                iterations = sorted(df_q[(df_q['Model'] == model) & (df_q['Variant'] != 'FC')]['Iteration'].dropna().unique())
                if len(iterations) == 0:
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

        handles = [
            plt.Line2D([], [], color=variant_colors[v], linestyle=variant_linestyles[v], marker='o', label=v)
            for v in ['ksemidin', 'k10', 'FC']
        ]

        ax_for_legend = g.axes[0, 0] if g.axes.ndim == 2 else g.axes[0]
        ax_for_legend.legend(handles=handles, title='Variant', loc='best')

        g.set_titles(row_template='{row_name}', col_template='{col_name}')
        g.set_axis_labels("Iteration", metric)
        g.figure.subplots_adjust(top=0.9)
        g.figure.suptitle(f"{metric} over Iterations - {q_type}", fontsize=16)

        safe_qtype = q_type.replace(" ", "_").replace("/", "_")
        filename = f"{metric}_iterative_{safe_qtype}_no_why.png"
        print(f"Saving figure: {filename}")
        g.savefig(os.path.join(output_path, filename))
        plt.close()
