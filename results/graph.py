import matplotlib.pyplot as plt
import numpy as np

# Abbreviated question types and full labels
question_types = ['STSR', 'TTSJ', 'TTTJ', 'STMR']
question_labels = [
    'Single Table Single Row',
    'Two Tables Single Join',
    'Three Tables Two Joins',
    'Single Table Multiple Rows'
]

# Nuove metriche: [Precision, Recall, Accuracy]
answer_metrics = [
    [0.80, 0.89, 0.80],  # STSR
    [0.19, 0.36, 0.25],  # TTSJ
    [0.20, 0.20, 0.20],  # TTTJ
    [0.59, 0.91, 0.33],  # STMR
]

explanation_metrics = [
    [0.60, 0.55, 0.20],  # STSR
    [0.30, 0.22, 0.08],  # TTSJ
    [0.50, 0.13, 0.00],  # TTTJ
    [0.56, 0.75, 0.00],  # STMR
]

answer_metrics = np.array(answer_metrics)
explanation_metrics = np.array(explanation_metrics)

n_groups = len(question_types)
bar_width = 0.12
index = np.arange(n_groups)

fig, ax = plt.subplots(figsize=(12, 6))

# Plot each metric
ax.bar(index - 1.5*bar_width, answer_metrics[:, 0], bar_width, label='Answer Precision', color='skyblue')
ax.bar(index - 0.5*bar_width, answer_metrics[:, 1], bar_width, label='Answer Recall', color='dodgerblue')
ax.bar(index + 0.5*bar_width, answer_metrics[:, 2], bar_width, label='Answer Accuracy', color='navy')

ax.bar(index + 1.5*bar_width, explanation_metrics[:, 0], bar_width, label='Explanation Precision', color='lightcoral')
ax.bar(index + 2.5*bar_width, explanation_metrics[:, 1], bar_width, label='Explanation Recall', color='indianred')
ax.bar(index + 3.5*bar_width, explanation_metrics[:, 2], bar_width, label='Explanation Accuracy', color='darkred')

# Labels and formatting
ax.set_xlabel('Question Type')
ax.set_ylabel('Score')
ax.set_title('Evaluation Metrics by Question Type')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(question_types)
ax.set_ylim(0, 1.05)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Legend for abbreviations directly in the plot
for i, (abbr, full_desc) in enumerate(zip(question_types, question_labels)):
    ax.text(i, 1.02, f"{abbr}: {full_desc}", ha='center', va='top', fontsize=9, color='black')

plt.tight_layout()
plt.savefig("metrics_by_question_llama8bk76_.png", bbox_inches='tight', dpi=300)
plt.show()
