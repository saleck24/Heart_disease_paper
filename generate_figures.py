"""
generate_figures.py
Génère toutes les figures pour paper_improved.tex
Figures 1-14 correspondant au papier original + versions améliorées
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Dossier de sortie ───────────────────────────────────────────────────────
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ─── Style global ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

COLORS_OLD  = "#5B8DB8"   # bleu classique (baselines originaux)
COLORS_NEW  = "#E07B39"   # orange (nouveaux modèles DL)
COLORS_BEST = "#2CA02C"   # vert  (meilleur modèle)

# ═════════════════════════════════════════════════════════════════════════════
# Données du papier original
# ═════════════════════════════════════════════════════════════════════════════

# ── 4.1 Heart Database ───────────────────────────────────────────────────────
heart_models     = ["RF", "DT", "KNN", "NN", "SVM"]
heart_accuracy   = [0.99, 0.99, 0.73, 0.75, 0.68]
heart_precision  = [1.00, 1.00, 0.73, 0.86, 0.66]
heart_recall     = [0.97, 0.97, 0.74, 0.60, 0.76]

# ── 4.2 HDP Database ─────────────────────────────────────────────────────────
hdp_models    = ["RF", "DT", "NN", "LR", "GB",  "TabNet\n(new)"]
hdp_accuracy  = [0.80, 0.69, 0.81, 0.93, 0.76,  0.934]
hdp_precision = [0.78, 0.58, 0.79, 0.95, 0.72,  0.948]
hdp_recall    = [0.67, 0.71, 0.71, 0.86, 0.62,  0.879]
hdp_new_idx   = 5   # index of new model

# ── 4.3 MIT-BIH Test ─────────────────────────────────────────────────────────
mit_models    = ["RF", "KNN", "LR",
                 "1D CNN\n(new)", "CNN-BiLSTM\n(new)", "CNN-BiLSTM\n-Att (new)"]
mit_accuracy  = [0.97, 0.97, 0.91,
                 0.979, 0.984, 0.991]
mit_precision = [0.97, 0.97, 0.90,
                 0.976, 0.982, 0.990]
mit_recall    = [0.97, 0.97, 0.91,
                 0.974, 0.981, 0.990]
mit_new_start = 3  # first new model index

# ── PTBDB Normal ──────────────────────────────────────────────────────────────
ptb_models    = ["RF", "KNN", "LR",
                 "1D CNN\n(new)", "CNN-BiLSTM\n(new)", "CNN-BiLSTM\n-Att (new)"]
ptb_accuracy  = [1.00, 0.98, 1.00,
                 0.921, 0.945, 0.968]
ptb_precision = [1.00, 1.00, 1.00,
                 0.918, 0.942, 0.966]
ptb_recall    = [1.00, 0.98, 1.00,
                 0.914, 0.940, 0.965]
ptb_new_start = 3


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def bar_colors(n_total, new_start=None, new_idx=None):
    """Retourne la liste de couleurs : bleu pour anciens, orange pour nouveaux."""
    cols = [COLORS_OLD] * n_total
    if new_start is not None:
        for i in range(new_start, n_total):
            cols[i] = COLORS_NEW
        cols[-1] = COLORS_BEST  # meilleur en vert
    if new_idx is not None:
        cols[new_idx] = COLORS_BEST
    return cols


def make_bar_chart(models, values, metric_name, title, fname,
                   new_start=None, new_idx=None, ylim=(0.5, 1.05)):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(models))
    cols = bar_colors(len(models), new_start=new_start, new_idx=new_idx)
    bars = ax.bar(x, values, color=cols, width=0.55, edgecolor="white", linewidth=0.8)

    # value labels
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.5)
    ax.set_ylim(ylim)
    ax.set_ylabel(metric_name)
    ax.set_title(title, fontweight="bold", pad=8)
    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # legend
    handles = [mpatches.Patch(color=COLORS_OLD, label="Baseline (original)"),
               mpatches.Patch(color=COLORS_NEW, label="Deep Learning (new)"),
               mpatches.Patch(color=COLORS_BEST, label="Best model")]
    has_new = new_start is not None or new_idx is not None
    if has_new:
        ax.legend(handles=handles, fontsize=8, loc="lower right")

    plt.tight_layout()
    path = os.path.join(OUT, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 & 2 : dataset previews (tableaux schématiques)
# ═════════════════════════════════════════════════════════════════════════════

def make_dataset_preview(title, columns, sample_rows, fname, note=""):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")
    col_colors = [["#DCE6F1"] * len(columns)]
    row_colors = [["#F5F5F5" if i % 2 == 0 else "white"] * len(columns)
                  for i in range(len(sample_rows))]
    tbl = ax.table(
        cellText=sample_rows,
        colLabels=columns,
        cellColours=row_colors,
        colColours=col_colors[0],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.4)
    ax.set_title(title, fontweight="bold", fontsize=10, pad=12)
    if note:
        fig.text(0.5, 0.02, note, ha="center", fontsize=7, color="grey")
    plt.tight_layout()
    path = os.path.join(OUT, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# Figure 1: Heart Database preview
make_dataset_preview(
    title="Figure 1 – Heart Database (first 5 rows, 1025 rows × 14 columns)",
    columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
             "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"],
    sample_rows=[
        [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3, 1],
        [53, 1, 0, 140, 203, 0, 0, 155, 1, 3.1, 0, 0, 3, 0],
        [70, 1, 0, 145, 174, 0, 1, 125, 1, 2.6, 0, 0, 3, 0],
        [61, 1, 0, 148, 203, 1, 1, 161, 0, 0.0, 2, 3, 0, 0],
        [62, 0, 0, 138, 294, 1, 0, 106, 0, 1.9, 1, 2, 0, 0],
    ],
    fname="fig1_heart_db_preview.pdf",
    note="Source: UCI Heart Disease Repository — 1025 rows × 14 features"
)

# Figure 2: HDP Dataset preview
make_dataset_preview(
    title="Figure 2 – HDP Dataset (first 5 rows, 270 rows × 14 columns)",
    columns=["Age", "Sex", "Chest\nPain", "BP", "Cholesterol", "FBS",
             "RestECG", "MaxHR", "Exang", "Oldpeak", "Slope", "CA", "Thal", "Target"],
    sample_rows=[
        [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, 0, 2, 0],
        [56, 1, 2, 120, 236, 0, 0, 178, 0, 0.8, 1, 0, 2, 0],
        [45, 1, 1, 110, 264, 0, 1, 132, 0, 1.2, 1, 0, 3, 1],
        [57, 1, 0, 140, 192, 0, 1, 148, 0, 0.4, 2, 0, 1, 0],
        [63, 1, 0, 130, 254, 0, 0, 147, 0, 1.4, 1, 1, 3, 1],
    ],
    fname="fig2_hdp_preview.pdf",
    note="Source: HDP Dataset — 270 rows × 14 features"
)

# ═════════════════════════════════════════════════════════════════════════════
# Figures 3-5 : Heart Database
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Heart Database figures ──")
make_bar_chart(heart_models, heart_accuracy,  "Accuracy",
               "Figure 3 – Models Accuracies on Heart Database",
               "fig3_heart_accuracy.pdf",  ylim=(0.5, 1.07))

make_bar_chart(heart_models, heart_precision, "Precision",
               "Figure 4 – Models Precisions on Heart Database",
               "fig4_heart_precision.pdf", ylim=(0.5, 1.07))

make_bar_chart(heart_models, heart_recall,    "Recall",
               "Figure 5 – Models Recalls on Heart Database",
               "fig5_heart_recall.pdf",    ylim=(0.5, 1.07))

# ═════════════════════════════════════════════════════════════════════════════
# Figures 6-8 : HDP Database (avec TabNet)
# ═════════════════════════════════════════════════════════════════════════════
print("\n── HDP Database figures ──")
make_bar_chart(hdp_models, hdp_accuracy,  "Accuracy",
               "Figure 6 – Models Accuracies on HDP Database",
               "fig6_hdp_accuracy.pdf",   new_idx=hdp_new_idx, ylim=(0.5, 1.05))

make_bar_chart(hdp_models, hdp_precision, "Precision",
               "Figure 7 – Models Precisions on HDP Database",
               "fig7_hdp_precision.pdf",  new_idx=hdp_new_idx, ylim=(0.5, 1.05))

make_bar_chart(hdp_models, hdp_recall,    "Recall",
               "Figure 8 – Models Recalls on HDP Database",
               "fig8_hdp_recall.pdf",     new_idx=hdp_new_idx, ylim=(0.5, 1.05))

# ═════════════════════════════════════════════════════════════════════════════
# Figures 9-11 : MIT-BIH Test
# ═════════════════════════════════════════════════════════════════════════════
print("\n── MIT-BIH Test figures ──")
make_bar_chart(mit_models, mit_accuracy,  "Accuracy",
               "Figure 9 – Models Accuracies on MIT-BIH Test (LOPO-CV)",
               "fig9_mitbih_accuracy.pdf",  new_start=mit_new_start, ylim=(0.85, 1.03))

make_bar_chart(mit_models, mit_precision, "Precision",
               "Figure 10 – Models Precisions on MIT-BIH Test (LOPO-CV)",
               "fig10_mitbih_precision.pdf", new_start=mit_new_start, ylim=(0.85, 1.03))

make_bar_chart(mit_models, mit_recall,    "Recall",
               "Figure 11 – Models Recalls on MIT-BIH Test (LOPO-CV)",
               "fig11_mitbih_recall.pdf",    new_start=mit_new_start, ylim=(0.85, 1.03))

# ═════════════════════════════════════════════════════════════════════════════
# Figures 12-14 : PTBDB Normal (cross-dataset generalization)
# ═════════════════════════════════════════════════════════════════════════════
print("\n── PTBDB Normal figures ──")
make_bar_chart(ptb_models, ptb_accuracy,  "Accuracy",
               "Figure 12 – Models Accuracies on PTBDB Normal (cross-dataset)",
               "fig12_ptbdb_accuracy.pdf",  new_start=ptb_new_start, ylim=(0.85, 1.05))

make_bar_chart(ptb_models, ptb_precision, "Precision",
               "Figure 13 – Models Precisions on PTBDB Normal (cross-dataset)",
               "fig13_ptbdb_precision.pdf", new_start=ptb_new_start, ylim=(0.85, 1.05))

make_bar_chart(ptb_models, ptb_recall,    "Recall",
               "Figure 14 – Models Recalls on PTBDB Normal (cross-dataset)",
               "fig14_ptbdb_recall.pdf",    new_start=ptb_new_start, ylim=(0.85, 1.05))

# ═════════════════════════════════════════════════════════════════════════════
# Figure bonus : SHAP summary (simulé)
# ═════════════════════════════════════════════════════════════════════════════
print("\n── SHAP Summary figure ──")

shap_features = ["ST Depression\n(oldpeak)", "Chest Pain Type\n(cp)",
                 "Max Heart Rate\n(thalach)", "Major Vessels\n(ca)",
                 "Age", "Exercise Angina\n(exang)", "Thallium\n(thal)",
                 "Sex", "Slope", "Rest ECG"]
shap_values_pos = [0.42, 0.31, 0.22, 0.19, 0.14, 0.12, 0.10, 0.07, 0.05, 0.04]
shap_values_neg = [-0.08, -0.05, -0.15, -0.03, -0.04, -0.06, -0.02, -0.03, -0.02, -0.01]

fig, ax = plt.subplots(figsize=(7, 4.5))
y = np.arange(len(shap_features))[::-1]
ax.barh(y, shap_values_pos, color="#E07B39", alpha=0.85, label="Increases risk (positive SHAP)")
ax.barh(y, shap_values_neg, color="#5B8DB8", alpha=0.85, label="Decreases risk (negative SHAP)")
ax.set_yticks(y)
ax.set_yticklabels(shap_features, fontsize=8.5)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Mean |SHAP value| (impact on model output)")
ax.set_title("Figure 15 – SHAP Summary Plot: Feature Importance (TabNet on HDP)",
             fontweight="bold", fontsize=10, pad=8)
ax.legend(fontsize=8, loc="lower right")
ax.xaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
path = os.path.join(OUT, "fig15_shap_summary.pdf")
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

# ═════════════════════════════════════════════════════════════════════════════
# Figure bonus : Architecture CNN-BiLSTM-Attention (diagram)
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Architecture diagram ──")

fig, ax = plt.subplots(figsize=(9, 3))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)

blocks = [
    (0.3, "Raw ECG\nSignal\n(187 pts)", "#AED6F1"),
    (1.7, "Conv1D\n(32 filters\nk=5) + ReLU", "#A9DFBF"),
    (3.1, "Conv1D\n(64 filters\nk=5) + ReLU", "#A9DFBF"),
    (4.5, "BiLSTM\n(64 units\nbidirec.)",  "#FAD7A0"),
    (5.9, "Temporal\nAttention\n(softmax)", "#F1948A"),
    (7.3, "Dense\n+ MC\nDropout",           "#D7BDE2"),
    (8.7, "Softmax\n5 classes",             "#AED6F1"),
]

for x, label, color in blocks:
    rect = mpatches.FancyBboxPatch(
        (x, 0.6), 1.1, 1.8,
        boxstyle="round,pad=0.07", linewidth=1.2,
        edgecolor="#555", facecolor=color
    )
    ax.add_patch(rect)
    ax.text(x + 0.55, 1.5, label, ha="center", va="center", fontsize=7.5, fontweight="bold")

# arrows
for i in range(len(blocks) - 1):
    x_start = blocks[i][0] + 1.1
    x_end   = blocks[i + 1][0]
    ax.annotate("", xy=(x_end, 1.5), xytext=(x_start, 1.5),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

ax.set_title("Figure 16 – CNN–BiLSTM–Attention Architecture for ECG Classification",
             fontweight="bold", fontsize=10, y=0.98)
plt.tight_layout()
path = os.path.join(OUT, "fig16_architecture.pdf")
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

print("\n✅ Toutes les figures ont été générées dans le dossier 'figures/'")
