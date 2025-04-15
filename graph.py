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

# Metrics: [Precision, Recall, Accuracy]
answer_metrics = [
    [0.60, 0.67, 0.60],  # STSR
    [0.20, 0.27, 0.17],  # TTSJ
    [0.00, 0.00, 0.00],  # TTTJ
    [0.73, 0.73, 0.33],  # STMR
]

explanation_metrics = [
    [0.80, 0.73, 0.70],  # STSR
    [0.20, 0.11, 0.08],  # TTSJ
    [0.20, 0.07, 0.00],  # TTTJ
    [0.55, 0.50, 0.00],  # STMR
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
ax.set_ylim(0, 1)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add a custom legend for the abbreviations inside the plot
for i, (abbr, full_desc) in enumerate(zip(question_types, question_labels)):
    ax.text(i, 0.95, f"{abbr}: {full_desc}", ha='center', va='top', fontsize=9, color='black')

plt.tight_layout()
plt.savefig("metrics_by_question_type_keywords.png", bbox_inches='tight', dpi=300)
plt.show()
