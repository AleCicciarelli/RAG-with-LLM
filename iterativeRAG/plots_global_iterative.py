import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Percorsi ===
base_path = 'iterativeRAG/'
output_path = os.path.join(base_path, 'outputs_plots_iterative/no_why/global/')
os.makedirs(output_path, exist_ok=True)

# === Config ===
models = ['llama8b','llama8b-ft2', 'mixtral8x7b', 'llama70b', 'deepseek70b']
metrics_to_plot = ['F1']

variant_colors = {
    'FC': 'blue',
    'kdin': 'orange',
    'k10': 'green',
    #'plan_k10': 'purple'
}

variant_linestyles = {
    'FC': '-',                  # linea continua
    'kdin': (0, (5, 5)),    # tratteggiata lunga
    'k10': (0, (1, 1)),          # tratteggiata corta
    #'plan_k10': (0, (3, 1, 1, 1))  # tratteggiata media
}

# === Caricamento dati ===
all_data = []
for model in models:
    variants = {
        'k10': f'{base_path}outputs_{model}/why/global_metrics_iterative_k10_5rounds.csv',
        'kdin': f'{base_path}outputs_{model}/why/global_metrics_iterative_kdin_5rounds.csv',
        'FC': f'full_context/global_metrics_{model}_why_FC.csv',
        #'plan_k10': f'{base_path}plan_generator/outputs_{model}/global_metrics_{model}_plan_k10.csv'

    }

    for variant, file_path in variants.items():
        print(f"Loading data for {model} - {variant} from {file_path}")
        #required_columns = ['F1', 'Category']
        optional_columns = ['Iteration']

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)

                # Verifica colonne obbligatorie
                #missing_required = [col for col in required_columns if col not in df.columns]
                #if missing_required:
                #    print(f"⚠️  [SKIP] {file_path} - Missing required columns: {missing_required}")
                #    continue

                # Aggiunta info modello e variante
                df['Model'] = model
                df['Variant'] = variant

                # Se manca Iteration e la variante è iterativa, lo segnaliamo
                if variant in ['k10', 'kdin'] and 'Iteration' not in df.columns:
                    print(f"⚠️  [WARNING] {variant} for {model} has no 'Iteration' column.")

                all_data.append(df)

            except Exception as e:
                print(f"❌  [ERROR] Failed to load {file_path}: {e}")
        else:
            print(f"❌  [MISSING] File not found: {file_path}")

df_all = pd.concat(all_data, ignore_index=True)
if 'Iteration' in df_all.columns:
    df_all['Iteration'] = pd.to_numeric(df_all['Iteration'], errors='coerce')
df_all.rename(columns={'Question Type': 'QuestionType'}, inplace=True)
# === Plot ===
sns.set_theme(style="whitegrid", font_scale=1.0)

for metric in metrics_to_plot:
    # Consideriamo solo dati iterativi per FacetGrid (escludiamo FC e plan_k10 dalle linee tracciate)
    df_iterative = df_all[df_all['Variant'] != 'FC'].copy()

    # FacetGrid con righe = Model, colonne = Category (Answer, Explanation)
    g = sns.FacetGrid(
        df_iterative,
        row='Model',
        col='Category',
        height=4,
        aspect=1.3,
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

    # Aggiungiamo linea orizzontale FC per ogni combinazione Model x Category
    for ax, (model, category) in zip(g.axes.flat, [(r, c) for r in g.row_names for c in g.col_names]):
        df_fc = df_all[(df_all['Model'] == model) & (df_all['Variant'] == 'FC') & (df_all['Category'] == category)]
    #for ax, model in zip(g.axes.flat, g.row_names):
    #    iterations = (
    #    sorted(df_iterative['Iteration'].dropna().unique())
    #    if 'Iteration' in df_iterative.columns
     #   else [0]
    #)
        # Linea orizzontale per FC
        valid_iter_df = df_all[(df_all['Model'] == model) & (df_all['Variant'].isin(['k10', 'ksemidin']))]
        iterations = sorted(valid_iter_df['Iteration'].dropna().unique())
        if len(iterations) == 0:
            iterations = [0]

        #df_fc = df_all[(df_all['Model'] == model) & (df_all['Variant'] == 'FC')]
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

        # Linea orizzontale per plan_k10
#        df_plan = df_all[(df_all['Model'] == model) & (df_all['Variant'] == 'plan_k10')]
#        if not df_plan.empty:
#            y_val = df_plan[metric].values[0]
#            ax.plot(
#                iterations,
#                [y_val] * len(iterations),
 #               label='plan_k10',
 #               linestyle=variant_linestyles['plan_k10'],
 #               marker='o',
   #             color=variant_colors['plan_k10']
  #          )
   #         for x in iterations:
    #            ax.text(
     #               x,
      #              y_val,
       #             f"{y_val:.4f}",
        #            fontsize=8,
         #           ha='center',
          #          va='bottom',
           #         color=variant_colors['plan_k10']
            #    )


    # Legenda (solo nel primo subplot)
    handles = [
    plt.Line2D([], [], color=variant_colors[v], linestyle=variant_linestyles[v], marker='o', label=v)
    for v in ['kdin', 'k10', 'FC']#, 'plan_k10']
]

    g.axes[0, 0].legend(handles=handles, title='Variant', loc='best')

    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    g.set_axis_labels("Iteration", metric)
    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle(f"{metric} over Iterations - Models × Category × Variant", fontsize=16)

    filename = f"{metric}_iterative_global_why.png"
    g.savefig(os.path.join(output_path, filename))
    plt.close()
    print(f"Saved plot for {metric} to {os.path.join(output_path, filename)}")
