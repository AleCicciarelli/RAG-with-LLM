'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Percorsi
base_path = 'tpch/outputs_'
output_path = base_path + 'plots/why/global/iterative/'
os.makedirs(output_path, exist_ok=True)

# Modelli
models = ['llama8b', 'mixtral8x7b', 'llama70b']
metrics_to_plot = ['F1']

# Colori
palette = {'llama8b': 'blue', 'mixtral8x7b': 'orange', 'llama70b': 'green'}

# Caricamento e unione dati
all_data = []
for model in models:
    file_path = os.path.join(base_path + model + '/iterative/global_metrics_iterative.csv')
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

        filename = f"{metric}_{category}_lineplot_iterative3round_previous.png".replace(" ", "_")
        plt.savefig(os.path.join(output_path, filename))
        plt.close()

print("Plot salvati in:", output_path)


#FC as upper bound


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Percorsi
base_path = 'tpch/outputs_'
output_path = base_path + 'plots/why/global/iterative/'
os.makedirs(output_path, exist_ok=True)

# Modelli
models = ['llama8b', 'mixtral8x7b', 'llama70b']
metrics_to_plot = ['F1']

# Colori base
base_palette = {'llama8b': 'blue', 'mixtral8x7b': 'orange', 'llama70b': 'green'}
variant_styles = {
    #'ksemidin': {'color': 'black', 'linestyle': ':'},
    'FC': {'color': 'green', 'linestyle': '-'},
    'kdin': {'color': 'blue', 'linestyle': '--'}
}
# Stili e colori estesi (Model+Variant)
line_styles = {
    'ksemidin': ':',
    'FC': '-',
    'k10': '--'
}
palette = {}

# Caricamento e unione dati
all_data = []

for model in models:
    variants = {
        #'ksemidin': f'{base_path}{model}/iterative/global_metrics_iterative_ksemidin_5rounds_NOWHY.csv',
        'FC': f'{base_path}{model}/iterative/global_metrics_iterative_k10_FC_5rounds_WHY.csv',
        'kdin': f'{base_path}{model}/iterative/global_metrics_iterative_k10_5rounds_WHY.csv'
    }

    for variant, file_path in variants.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Model'] = model
            df['Variant'] = variant
            all_data.append(df)

            # Palette estesa per varianti
            palette[f'{model}_{variant}'] = base_palette[model]
df_all = pd.concat(all_data)
df_all['Iteration'] = df_all['Iteration'].astype(int)

# Plot per modello
sns.set_theme(style="whitegrid", font_scale=1.1)

for model in models:
    model_data = df_all[df_all['Model'] == model]

    for metric in metrics_to_plot:
        for category in model_data['Category'].unique():
            plt.figure(figsize=(10, 6))
            subset = model_data[model_data['Category'] == category]

            for variant, group_data in subset.groupby('Variant'):
                style = variant_styles.get(variant, {'color': 'gray', 'linestyle': '-'})
                label = f'{variant}'

                ax = sns.lineplot(
                    data=group_data,
                    x='Iteration',
                    y=metric,
                    label=label,
                    color=style['color'],
                    linestyle=style['linestyle'],
                    marker='o'
                )

                for x_val, y_val in zip(group_data['Iteration'], group_data[metric]):
                    plt.text(x_val, y_val, f'{y_val:.3f}', color=style['color'], ha='center')

            plt.title(f'{model} - {metric} over Iterations ({category})')
            plt.xlabel('Iteration')
            plt.ylabel(metric)
            plt.xticks(sorted(df_all['Iteration'].unique()))
            plt.legend(title='Variant')
            plt.tight_layout()

            filename = f"{model}_{metric}_{category}_lineplot_variants.png".replace(" ", "_")
            plt.savefig(os.path.join(output_path, filename))
            plt.close()

print("Plot salvati in:", output_path)


# Plot per Answer ed Explanation con colori e stili distinti


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Percorsi
base_path = 'tpch/outputs_'
output_path = base_path + 'plots/why/global/iterative/'
os.makedirs(output_path, exist_ok=True)

# Modelli
models = ['llama8b', 'mixtral8x7b', 'llama70b']
metrics_to_plot = ['F1']

# Colori per Answer / Explanation
category_colors = {
    'Answer': 'darkorange',
    'Explanation': 'steelblue'
}

# Linestyle per variante
variant_linestyles = {
    'FC': '-',
    'kdin': '--'
}

# Caricamento dati
all_data = []
for model in models:
    variants = {
        'FC': f'{base_path}{model}/iterative/global_metrics_iterative_k10_FC_5rounds_WHY.csv',
        'kdin': f'{base_path}{model}/iterative/global_metrics_iterative_k10_5rounds_WHY.csv'
    }

    for variant, file_path in variants.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Model'] = model
            df['Variant'] = variant
            all_data.append(df)

df_all = pd.concat(all_data)
df_all['Iteration'] = df_all['Iteration'].astype(int)

sns.set_theme(style="whitegrid", font_scale=1.1)

# Per ogni modello e metrica, plottiamo Answer ed Explanation con colori diversi
for model in models:
    model_data = df_all[df_all['Model'] == model]

    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        ax = plt.gca()

        for (variant, category), group_data in model_data.groupby(['Variant', 'Category']):
            color = category_colors.get(category, 'gray')
            linestyle = variant_linestyles.get(variant, '-')
            label = f'{category} ({variant})'

            sns.lineplot(
                data=group_data,
                x='Iteration',
                y=metric,
                label=label,
                color=color,
                linestyle=linestyle,
                marker='o',
                ax=ax
            )

            for x_val, y_val in zip(group_data['Iteration'], group_data[metric]):
                plt.text(x_val, y_val + 0.003, f'{y_val:.2f}', color=color, ha='center')

        plt.title(f'{model} - {metric} over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel(metric)
        plt.xticks(sorted(df_all['Iteration'].unique()))
        plt.legend(title='Category (Variant)')
        plt.tight_layout()

        filename = f"{model}_{metric}_lineplot_by_category.png".replace(" ", "_")
        plt.savefig(os.path.join(output_path, filename))
        plt.close()

print("Plot aggiornati salvati in:", output_path)
'''
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Percorsi
base_path = 'tpch/outputs_'
output_path = base_path + 'plots/why/global/iterative/'
os.makedirs(output_path, exist_ok=True)

# Configurazioni
models = ['llama8b', 'mixtral8x7b', 'llama70b']
metrics_to_plot = ['F1']

# Colori per modello
model_colors = {
    'llama8b': 'blue',
    'mixtral8x7b': 'orange',
    'llama70b': 'green'
}

# Linestyle per categoria
category_linestyles = {
    'Answer': '-',
    'Explanation': '--'
}

# Caricamento e unione dati
all_data = []
for model in models:
    variants = {
        'FC': f'{base_path}{model}/iterative/global_metrics_iterative_k10_FC_5rounds_WHY.csv',
        'kdin': f'{base_path}{model}/iterative/global_metrics_iterative_k10_5rounds_WHY.csv'
    }

    for variant, file_path in variants.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Model'] = model
            df['Variant'] = variant
            all_data.append(df)

df_all = pd.concat(all_data)
df_all['Iteration'] = df_all['Iteration'].astype(int)

sns.set_theme(style="whitegrid", font_scale=1.1)

# Plot combinato
for metric in metrics_to_plot:
    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    for (model, variant, category), group_data in df_all.groupby(['Model', 'Variant', 'Category']):
        color = model_colors.get(model, 'gray')
        linestyle = category_linestyles.get(category, '-')
        label = f'{model} - {category} ({variant})'

        sns.lineplot(
            data=group_data,
            x='Iteration',
            y=metric,
            label=label,
            color=color,
            linestyle=linestyle,
            marker='o',
            ax=ax
        )

        for x_val, y_val in zip(group_data['Iteration'], group_data[metric]):
            plt.text(x_val, y_val + 0.003, f'{y_val:.2f}', color=color, ha='center', fontsize=8)

    plt.title(f'{metric} over Iterations - All Models')
    plt.xlabel('Iteration')
    plt.ylabel(metric)
    plt.xticks(sorted(df_all['Iteration'].unique()))
    plt.legend(title='Model - Category (Variant)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    filename = f"all_models_{metric}_lineplot_by_model_and_category.png".replace(" ", "_")
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

print("Plot con tutti i modelli salvati in:", output_path)
'''
''' facet grid version'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Percorsi
base_path = 'tpch/outputs_'
output_path = base_path + 'plots/no_why/global/iterative/'
os.makedirs(output_path, exist_ok=True)

# Config
models = ['llama8b', 'mixtral8x7b', 'llama70b']
metrics_to_plot = ['F1']

# Caricamento dati
all_data = []
for model in models:
    variants = {
        'ksemidin': f'{base_path}{model}/iterative/global_metrics_iterative_ksemidin_5rounds_NOWHY.csv',
        'FC': f'{base_path}{model}/iterative/global_metrics_iterative_FC_5rounds_NOWHY.csv',
        'k10': f'{base_path}{model}/iterative/global_metrics_iterative_k10_5rounds_NOWHY.csv'
    }
    for variant, file_path in variants.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Model'] = model
            df['Variant'] = variant
            all_data.append(df)

df_all = pd.concat(all_data)
df_all['Iteration'] = df_all['Iteration'].astype(int)

sns.set_theme(style="whitegrid", font_scale=1.0)
variant_colors = {
    'FC': 'blue',
    'ksemidin': 'orange',
    'k10': 'green'
}

variant_linestyles = {
    'FC': '-',
    'ksemidin': (0, (5, 5)),
    'k10': (0, (1, 1))
}

for metric in metrics_to_plot:
    g = sns.FacetGrid(
        df_all,
        row='Model',
        height=4,
        aspect=1.5,
        sharey=True
    )
    
    def plot_variants(data, color, **kwargs):
        ax = plt.gca()
        for variant, linestyle in variant_linestyles.items():
            df_variant = data[data['Variant'] == variant]
            if df_variant.empty:
                continue
            line_color = variant_colors.get(variant, 'black')
            ax.plot(
                df_variant['Iteration'],
                df_variant[metric],
                label=variant,
                linestyle=linestyle,
                marker='o',
                color=line_color
            )
            for _, row in df_variant.iterrows():
                ax.text(
                    row['Iteration'],
                    row[metric],
                    f"{row[metric]:.4f}",
                    fontsize=8,
                    ha='center',
                    va='bottom',
                    color=line_color
                )

    g.map_dataframe(plot_variants)

    ax1 = g.axes[0,0]
    handles = []
    for variant, linestyle in variant_linestyles.items():
        handles.append(
            plt.Line2D(
                [], [], 
                color=variant_colors.get(variant, 'black'), 
                linestyle=linestyle, 
                marker='o', 
                label=variant
            )
        )
    ax1.legend(handles=handles, title='Variant', loc='best')

    g.set_titles(row_template='{row_name}')
    g.set_axis_labels("Iteration", metric)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"{metric} over Iterations - Models × Variant", fontsize=16)

    filename = f"{metric}_facetgrid_models_variants_manual.png"
    g.savefig(os.path.join(output_path, filename))
    plt.close()