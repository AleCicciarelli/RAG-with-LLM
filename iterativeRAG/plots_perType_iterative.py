import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Percorsi
base_path = 'iterativeRAG/'
output_path = base_path + 'outputs_plots_iterative/why/per_type/'
os.makedirs(output_path, exist_ok=True)

# Config
models = ['llama8b', 'llama8b-ft2','mixtral8x7b', 'llama70b', 'deepseek70b']
metrics_to_plot = ['F1']  # metrica unica F1, divisa per Category (Answer/Explanation)

# Caricamento dati
all_data = []
for model in models:
    variants = {
        'k10': f'{base_path}outputs_{model}/why/metrics_by_type_iterative_k10_5rounds.csv',
        'kdin': f'{base_path}outputs_{model}/why/metrics_by_type_iterative_kdin_5rounds.csv',
        'FC': f'full_context/metrics_by_type_{model}_why_FC.csv',
        #'plan_k10': f'{base_path}plan_generator/outputs_{model}/metrics_by_type_{model}_plan_k10.csv'
    }
    for variant, file_path in variants.items():
        print(f"Loading data for {model} - {variant} from {file_path}")
        #required_columns = ['F1']#, 'Category']
        optional_columns = ['Iteration']

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)

                # Rinominare colonna 'Question Type' se esiste
                if 'Question Type' in df.columns:
                    df.rename(columns={'Question Type': 'QuestionType'}, inplace=True)
                # Trasforma in formato long se ci sono Answer F1 / Explanation F1
                if 'Answer F1' in df.columns and 'Explanation F1' in df.columns:
                    df_long = pd.melt(
                        df,
                        id_vars=[col for col in df.columns if col not in ['Answer F1', 'Explanation F1']],
                        value_vars=['Answer F1', 'Explanation F1'],
                        var_name='Category',
                        value_name='F1'
                    )
                    df_long['Category'] = df_long['Category'].str.replace(' F1', '')
                    df = df_long

                # Rinominare metrica aggregata se presente
                if 'Answer F1' in df.columns:
                    df.rename(columns={'Answer F1': 'F1'}, inplace=True)

                
                # Verifica colonne obbligatorie
                #missing_required = [col for col in required_columns if col not in df.columns]
                #if missing_required:
                #    print(f"⚠️  [SKIP] {file_path} - Missing required columns: {missing_required}")
                #    continue

                # Aggiunta info modello e variante
                df['Model'] = model
                df['Variant'] = variant

                # Segnala se manca Iteration ma serve (solo per varianti iterative)
                if variant in ['k10', 'kdin'] and 'Iteration' not in df.columns:
                    print(f"⚠️  [WARNING] {variant} for {model} has no 'Iteration' column.")

                all_data.append(df)

            except Exception as e:
                print(f"❌  [ERROR] Failed to load {file_path}: {e}")
        else:
            print(f"❌  [MISSING] File not found: {file_path}")

df_all = pd.concat(all_data, ignore_index=True)

# Converti Iteration in int, ignorando eventuali NaN (per FC)
if 'Iteration' in df_all.columns:
    df_all['Iteration'] = pd.to_numeric(df_all['Iteration'], errors='coerce')

sns.set_theme(style="whitegrid", font_scale=1.0)

variant_colors = {
    'FC': 'blue',
    'kdin': 'orange',
    'k10': 'green',
    #'plan_k10': 'purple'
}

variant_linestyles = {
    'FC': '-',                  # linea continua (per FC sarà orizzontale)
    'kdin': (0, (5, 5)),        # tratteggiata lunga
    'k10': (0, (1, 1)),         # tratteggiata corta
    #'plan_k10': (0, (3, 1, 1, 1))  # tratteggiata media
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
            col='Category',  # colonna per Answer / Explanation
            height=4,
            aspect=1.5,
            sharey=True
        )

        def plot_iterative(data, **kwargs):
            ax = plt.gca()
            for variant in ['kdin', 'k10']:
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
        for ax, (model, category) in zip(g.axes.flat, 
                                       [(m, c) for m in g.row_names for c in g.col_names]):
        
        #   df_fc = df_q[(df_q['Model'] == model) & (df_q['Category'] == category) & (df_q['Variant'] == 'FC')]
        #for ax, model in zip(g.axes.flat, g.row_names):
        #    iterations = sorted(
        #        df_q[(df_q['Model'] == model) & (df_q['Variant'] != 'FC')] #& (df_q['Variant'] != 'plan_k10')]['Iteration']
        #        .dropna()
                #.unique()
        #    )
        # Calcola le iteration presenti solo nelle varianti iterative (escludi FC e plan_k20)
            valid_iter_df = df_q[(df_q['Model'] == model) & (df_q['Variant'].isin(['k10', 'ksemidin']))]
            iterations = sorted(valid_iter_df['Iteration'].dropna().unique())
            if len(iterations) == 0:
                iterations = [0]

            # === FC ===
            df_fc = df_q[(df_q['Model'] == model) & (df_q['Category'] == category) & (df_q['Variant'] == 'FC')]
            #df_fc = df_q[(df_q['Model'] == model) & (df_q['Variant'] == 'FC')]
            if not df_fc.empty:
                y_val = df_fc[metric].values[0]
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

            # === plan_k10 ===
#            df_plan = df_q[(df_q['Model'] == model) & (df_q['Variant'] == 'plan_k10')]
#            if not df_plan.empty:
#                y_val = df_plan[metric].values[0]
#                ax.plot(
#                    iterations,
#                    [y_val] * len(iterations),
#                    label='plan_k10',
#                    linestyle=variant_linestyles['plan_k10'],
#                    marker='o',
#                    color=variant_colors['plan_k10']
#                )

#                for x in iterations:
#                    ax.text(
#                        x,
#                        y_val,
#                        f"{y_val:.4f}",
#                        fontsize=8,
#                        ha='center',
#                        va='bottom',
#                        color=variant_colors['plan_k10']
#                    )

        handles = [
            plt.Line2D([], [], color=variant_colors[v], linestyle=variant_linestyles[v], marker='o', label=v)
            for v in ['kdin', 'k10', 'FC']#,'plan_k10']
        ]

        ax_for_legend = g.axes[0, 0] if g.axes.ndim == 2 else g.axes[0]
        ax_for_legend.legend(handles=handles, title='Variant', loc='best')

        g.set_titles(row_template='{row_name}', col_template='{col_name}')
        g.set_axis_labels("Iteration", metric)
        g.figure.subplots_adjust(top=0.9)
        g.figure.suptitle(f"{metric} over Iterations - {q_type}", fontsize=16)

        safe_qtype = q_type.replace(" ", "_").replace("/", "_")
        filename = f"{metric}_iterative&plan_k10_{safe_qtype}_why.png"
        print(f"Saving figure: {filename}")
        g.savefig(os.path.join(output_path, filename))
        plt.close()
