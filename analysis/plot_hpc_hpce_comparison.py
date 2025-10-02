import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
sys.path.append(os.getcwd())
from dotenv import load_dotenv

load_dotenv()

# Styling configuration
sns.set_theme(font_scale=2.1, style='whitegrid')
sns.color_palette("colorblind")
font = {'family': 'serif', 'size': 19}
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', **font)
mpl.rc('xtick', labelsize=19)
plt.rcParams["font.family"] = "Nimbus Roman"
mpl.rc('ytick', labelsize=19)

# Load the results
perf_olmo = pd.read_csv(os.path.join(os.environ["base_dir"], "results", "olmo2-7B_full_v2_mult_perf_len_ablation.csv"))
perf_mistral = pd.read_csv(os.path.join(os.environ["base_dir"], "results", "mistral7B_full_v2_mult_perf_len_ablation.csv"))
perf_qwen = pd.read_csv(os.path.join(os.environ["base_dir"], "results", "qwen7B-instruct_full_v2_mult_perf_len_ablation.csv"))
perf_olmo13B = pd.read_csv(os.path.join(os.environ["base_dir"], "results", "olmo2-13B_full_v2_mult_perf_len_ablation.csv"))
perf_qwen14B = pd.read_csv(os.path.join(os.environ["base_dir"], "results", "qwen2.5-14B-instruct_full_v2_mult_perf_len_ablation.csv"))

# Combine the model DataFrames using different metrics per task
dfs = []
for name, df in [('OLMo2-7B', perf_olmo), 
                 ('Mistral7B', perf_mistral), 
                 ('Qwen-7B', perf_qwen), 
                 ('OLMo2-13B', perf_olmo13B), 
                 ('Qwen-14B', perf_qwen14B)]:
    # Use F1 for PCK and RAG tasks, exact_match for others
    tmp_pck = df[(df['metric'] == 'f1') & (df['task'] == 'PCK')].copy()
    tmp_rag = df[(df['metric'] == 'f1') & (df['task'] == 'RAG')].copy()
    tmp_others = df[(df['metric'] == 'exact_match') & (df['task'] != 'PCK') & (df['task'] != 'RAG')].copy()
    tmp = pd.concat([tmp_pck, tmp_rag, tmp_others], ignore_index=True)
    tmp['model'] = name
    dfs.append(tmp)

data = pd.concat(dfs, ignore_index=True)

# Create long format data for HPC (single context), HPC-double (doubled context), and HPCE (double context)
evidence_cols = ['HPC', 'HPC-double', 'HPCE']

long_mean = data.melt(
    id_vars=['model', 'task', 'metric'],
    value_vars=evidence_cols,
    var_name='evidence',
    value_name='performance'
)

long_sem = data.melt(
    id_vars=['model', 'task', 'metric'],
    value_vars=['HPC_sem', 'HPC-double_sem', 'HPCE_sem'],
    var_name='evidence',
    value_name='sem'
)
long_sem['evidence'] = long_sem['evidence'].str.replace('_sem', '', regex=False)

long = long_mean.merge(long_sem, on=['model', 'task', 'metric', 'evidence'])

# Define consistent color palette
set2 = sns.color_palette('Set2', 4)
PALETTE = {'HPC': set2[1], 'HPC-double': set2[2], 'HPCE': set2[3]}
HUE_ORDER = evidence_cols

# Create figure with subplots for different tasks
tasks = ["CK", "PK", "PCK"]
pretty_tasks = {
    "CK": 'Contextual Knowledge', 
    "PK": 'Parametric Knowledge', 
    "PCK": 'Parametric-Contextual',
    "RAG": 'RAG'
}
n_tasks = len(tasks)

fig, axes = plt.subplots(
    1, n_tasks,
    figsize=(5.0 * n_tasks, 4.5),
    sharey=True
)

for ax, t in zip(axes, tasks):
    sub = long[(long['task'] == t) & (long['evidence'].isin(['HPC', 'HPC-double', 'HPCE']))]
    
    sns.barplot(
        ax=ax,
        data=sub,
        x='model', 
        y='performance',
        hue='evidence', 
        hue_order=HUE_ORDER,
        palette=PALETTE,
        width=0.65, 
        errorbar=None
    )

    # Add error bars
    for bar, se in zip(ax.patches, sub['sem']):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.errorbar(
            x, y, yerr=se,
            color='black',
            capsize=6, elinewidth=1, capthick=1
        )
    
    # Subplot cosmetics
    ax.set_title(pretty_tasks[t], fontsize=20, pad=6)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=0, labelsize=11)

# Single shared legend
handles, labels = axes[0].get_legend_handles_labels()
leg = fig.legend(
    handles, labels, 
    loc='lower right',
    bbox_to_anchor=(0.98, 0.3),
    frameon=True,
    prop={'size': 12}
)
leg.get_title().set_fontsize(12)

# Remove individual legends
for ax in axes:
    ax.get_legend().remove()

# Add shared y-label
fig.text(0.04, 0.5, 'Performance', 
         va='center', rotation='vertical', fontsize=17)

fig.tight_layout(rect=[0.05, 0.12, 1, 1])

# Save figure
output_dir = os.path.join(os.environ['base_dir'], "results", "figures")
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, f"HPC_HPCE_HPC_double_comparison_mixed_metrics.pdf")
fig.savefig(out_path, bbox_inches='tight')
print(f"Plot saved to: {out_path}")
plt.show()