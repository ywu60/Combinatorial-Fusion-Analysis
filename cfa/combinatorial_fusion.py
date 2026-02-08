from .fusion_function import compute_cd_ds, average_score_combination, average_rank_combination, \
    weighted_score_combination_by_ds, weighted_rank_combination_by_ds, compute_performance


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string # get letters A, B, etc
import itertools # use combination function
from typing import Literal # function definition
import matplotlib.ticker as ticker # for RSC function graph



def cfa_single_layer(probs_df, y_true, perf_metric: Literal["accuracy", "auroc"]):
    # ========= Step 1 basic check for df and target vector ==========
    # check if probs_df has more than 2 columns
    if probs_df.shape[1] < 3:
        print("The data frame should have more than 2 columns!")
        raise SystemExit

    # check if probs_df and y_true have same number of rows
    if probs_df.shape[0] != y_true.shape[0]:
        print("The data frame and the target vector don't have same number of rows!")
        raise SystemExit 


    #======== Step 2 get performance for base models ========
    # change column names to letters, such as A, B, C, etc
    n_col = probs_df.shape[1] # get number of columns
    probs_df.columns = list(string.ascii_uppercase[:n_col])

    # get performance for base models: accuracy or auroc
    base_perf = compute_performance(probs_df, perf_metric, y_true, score=True)
        

    #======== Step 3 combine models using CFA ========
    # use CFA to combine them (4 types of combination: ASC, ARC, WSCDS, WRCDS)
    df_asc = average_score_combination(probs_df)
    df_arc = average_rank_combination(probs_df)
    df_wscds = weighted_score_combination_by_ds(probs_df)
    df_wrcds = weighted_rank_combination_by_ds(probs_df)


    #======== Step 4 compute perforomance for combined models ========
    perf_asc = compute_performance(df_asc, perf_metric, y_true, score = True)
    perf_arc = compute_performance(df_arc, perf_metric, y_true, score = False)
    perf_wscds = compute_performance(df_wscds, perf_metric, y_true, score = True)
    perf_wrcds = compute_performance(df_wrcds, perf_metric, y_true, score = False)

    # put all performance together by index
    combined_df = pd.concat([perf_asc, perf_wscds, perf_arc, perf_wrcds], axis = 1, keys = ['asc', 'wscds', 'arc', 'wrcds'])


    #======== Step 5 put base and combined performance together ========
    base_block = pd.DataFrame(index=base_perf.index, columns=combined_df.columns)
    for col in combined_df.columns:
        base_block[col] = base_perf

    # Put base rows before df rows
    fusion_df = pd.concat([base_block, combined_df], axis=0)
    
    return fusion_df


def performance_plot(
    fusion_df,
    sort_col="asc",                  # sort within each group by this column
    draw_cols=("asc", "wscds", "arc", "wrcds"),  # columns to draw lines for
    palette=None, markers=None,
    figsize=(10, 4),
    ylabel="Accuracy", xlabel="CFA models",
):
    import seaborn
    # compute number of letters for index, so we can sort within each group with same number of letters
    f = fusion_df.copy()
    f["k"] = f.index.to_series().astype(str).str.len()
    base_perf = f.loc[f['k'] == 1, :].drop(columns = ['k'])

    # sort by group k then by sort_col within group
    f = f.sort_values(["k", sort_col], ascending=[True, True])

    # ---------- palette/markers ----------
    if palette is None:
        palette = ['#B368FF', "#223deca6", '#FF8C00', "#0DA86C"]
    if markers is None:
        markers = ['x', '^', 's', 'o']

    # ---------- set x as numeric positions ----------
    x = np.arange(len(f))
    labels = f.index.astype(str).tolist()

    fig, ax = plt.subplots(figsize=figsize)

    # style maps for metric columns (use first len(draw_cols) colors/markers)
    draw_cols = list(draw_cols)
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(draw_cols)}
    marker_map = {c: markers[i % len(markers)] for i, c in enumerate(draw_cols)}

    # ---------- plot group by group ----------
    # We also only draw ONE column in k=1 group (because all cols are the same there).
    # keep track of which labels have already been used
    labeled = set()
    single_model_color = "#FF3030"   # same as the best single line

    first_group_only_col = draw_cols[0] # data for single model, only draw one column
    for k, g in f.groupby("k", sort=False):
        pos = np.array([f.index.get_loc(ii) for ii in g.index])

        for col in draw_cols:
            if k == 1:
                if col != first_group_only_col:
                    continue

                # draw singe models in red with circle markers
                lab = "single model" if "Single" not in labeled else "_nolegend_"
                labeled.add("Single")

                ax.plot(
                    pos, g[col].values,
                    color=single_model_color,
                    marker="o", mfc="none",
                    mec=single_model_color, markersize=5,
                    linestyle="-",linewidth=1,
                    label=lab
                )
                continue

            # ---- all other groups ----
            lab = col.upper() if col not in labeled else "_nolegend_"
            labeled.add(col)

            ax.plot(
                pos, g[col].values,
                color=color_map[col],
                marker=marker_map[col], markersize=5, mfc = 'none',
                linestyle="-", linewidth=1,
                label=lab
            )

    # ---------- vertical separators between groups ----------
    # boundary after each group block: at last_index_of_group + 0.5
    ends = []; start = 0
    for k in sorted(f["k"].unique()):
        size = (f["k"] == k).sum()
        end = start + size - 1
        ends.append(end)
        start = end + 1

    # draw vlines at boundaries between groups (not after last group)
    for end in ends[:-1]:
        ax.axvline(end + 0.5, color="black", linestyle="--", linewidth=1)

    # ---------- best single horizontal line ----------

    base_best = float(pd.DataFrame(base_perf).max().max())
    ax.axhline(base_best, color=single_model_color, linestyle="-.", label="best single", linewidth=1)

    # ---------- formatting ----------
    ax.set_xlabel(xlabel, fontsize=13); ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=70)
    ax.grid(True)
    ax.legend(title="", fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f"Best combination: {f[draw_cols].max().max():.4f}")
    print(f"Best single: {pd.DataFrame(base_perf).max().max():.4f}")

    f = f.drop(columns=['k'])
    return f  # return sorted df


def RSC_graph(df: pd.DataFrame,
            palette=None, markers=None,
            title=r"Rank-score function graph", max_markers=60, leg = True):

    import seaborn
    # ---- compute rank-score function ----
    f1, _, _ = compute_cd_ds(df)

    m = f1.shape[1]  # number of lines/columns
    n = len(f1)      # number of rows (ranks)

    # ---- default colors/markers (up to 6) ----
    default_palette = [
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#009E73",  # green
        "#D55E00",  # red
        "#CC79A7",  # purple
        "#56B4E9",  # sky blue
    ]
    default_markers = ["o", "s", "^", "D", "x", "v"]
    
    if (m > 6) and (palette is None) and (markers is None):
        raise ValueError(
            f"RSC_graph: df has {m} columns but default palette/markers support up to 6.\n"
            "Please pass `palette=[...]` and `markers=[...]` with length >= number of columns."
        )

    if palette is None:
        palette = default_palette
    if markers is None:
        markers = default_markers


    if len(palette) < m or len(markers) < m:
        raise ValueError(
            f"RSC_graph: Need at least {m} colors and {m} markers "
            f"(got {len(palette)} colors, {len(markers)} markers)."
        )

    # ---- rename columns nicely (f_A, f_B, ...) based on how many columns ----
    f1.columns = [rf"$f_{{{chr(ord('A') + i)}}}$" for i in range(m)]

    # ---- adaptive marker placement ----
    # We want ~max_markers markers total at most, and all markers if small n.
    if n <= max_markers:
        mark_idx = np.arange(n)  # all points
    else:
        # choose ~max_markers evenly spaced marker indices
        mark_idx = np.linspace(0, n - 1, max_markers)
        mark_idx = np.unique(np.rint(mark_idx).astype(int))

    # ---- plot ----
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for i, col in enumerate(f1.columns):
        ax.plot(
            np.arange(n), f1[col].values,
            color=palette[i], marker=markers[i],
            linewidth=1.1, markersize=6,
            markevery=mark_idx,   # adaptive marker density
            label=col
        )

    ax.set_title(title, fontsize = 15)
    ax.set_xlabel("Rank", fontsize = 15)
    ax.set_ylabel("Normalized score", fontsize = 15)

    # ---- adaptive x ticks ----
    x0, x1 = 0, n - 1
    if n <= 25:
        ticks = np.arange(n)
    else:
        ticks = np.linspace(x0, x1, 15)
        ticks = np.unique(np.rint(ticks).astype(int))
        ticks[0] = x0
        ticks[-1] = x1

    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)+1}"))

    # padding before the first x tick and last x tick
    pad = max(2, int(0.02 * n))
    ax.set_xlim(-pad, (n - 1) + pad)

    ax.minorticks_on(); ax.grid(True)
    ax.tick_params(axis='both', which='major', length=6, width=1.0, direction='in')
    ax.tick_params(axis='x', labelrotation=40)

    leg = ax.get_legend(fontsize = 15)
    if leg is False:
        leg.remove()

    fig.tight_layout()
    plt.show()