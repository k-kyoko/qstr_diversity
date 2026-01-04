import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
import seaborn as sns
from typing import Optional, Sequence, Union, Mapping
from adjustText import adjust_text


def plt_dissim_heatmap(
    dissimilarity_matrix: pd.DataFrame,
    mode_dissim: bool = True,
    save: bool = False,
    figpath: str = "/home/jovyan/work/fig/temp/",
    annotation: bool = False
    ):

    """
    dissimilarity matrixをヒートマップとして表示する関数。

    Parameters:
    - dissimilarity_matrix (pd.DataFrame): 表示するdissimilarity matrix
    - title (str): ヒートマップのタイトル
    - annotation: numbers in each cell (or not)
    """

    fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
    if mode_dissim:
        sns.heatmap(dissimilarity_matrix, annot=annotation, fmt=".2f",
                    cmap="GnBu", cbar=True, vmin=0, vmax=7, square=True)
    else:
        sns.heatmap(dissimilarity_matrix, annot=annotation, fmt=".2f",
                    cmap="GnBu_r", cbar=True, square=True)

    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()
    ax.set_xticklabels(xticklabels, fontsize=15)
    ax.set_yticklabels(yticklabels, fontsize=15)

    figpath = Path(figpath)
    safe_stem = re.sub(r"[:/\s]", "_", figpath.stem)
    save_path = figpath.with_name(safe_stem).with_suffix(".png")

    title = figpath.stem  # last part of figpath
    ax.set_title(title, fontsize=24)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)

    if save:
        figpath.parent.mkdir(parents=True, exist_ok=True)
        save_path = figpath.with_suffix(".png")
        plt.savefig(save_path, dpi=300)
    plt.show()


def plt_mds(mds_res: pd.DataFrame,
            dissimilarity_matrix: pd.DataFrame,
            figpath: str = '/home/jovyan/work/fig/temp/'):
    # Plot the MDS results
    fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
    ax.scatter(mds_res[:, 0], mds_res[:, 1])

    # ラベル調整
    texts = [plt.text(mds_res[i, 0], mds_res[i, 1], word, fontsize=15)
             for i, word in enumerate(dissimilarity_matrix.index)]
    adjust_text(texts, only_move={'points': 'xy', 'text': 'xy'},
                arrowprops=dict(arrowstyle="->", color='b', lw=1.2))

    # 最初の10点を赤い線、次の15点を黒い線でつなぐ
    ax.plot(mds_res[:10, 0], mds_res[:10, 1], 'r-', lw=1, label='Colours')
    ax.plot(mds_res[10:25, 0], mds_res[10:25, 1], 'g-', lw=1, label='Emotions')

    # 凡例
    ax.legend(fontsize=22)
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()
    ax.set_xticklabels(xticklabels, fontsize=15)
    ax.set_yticklabels(yticklabels, fontsize=15)
    ax.set_xlabel('Dim 1', fontsize=18)
    ax.set_ylabel('Dim 2', fontsize=18)
    figpath = Path(figpath)
    title = figpath.stem
    ax.set_title(title, fontsize=24)

    figpath.parent.mkdir(parents=True, exist_ok=True)
    save_path = figpath.with_suffix(".png")
    plt.savefig(save_path, dpi=300)
    plt.show()


def plt_metric_individual(
    metric_df: pd.DataFrame,
    metric_name: str,
    output_dir: str,
    nrows: int = 5,
    ncols: int = 4,
    ylim: Optional[tuple[float, float]] = None,
    xlim: Optional[tuple[float, float]] = None,
    yticks_all: bool = False,
    xticks_all: bool = False,
) -> list[str]:

    os.makedirs(output_dir, exist_ok=True)
    t_sorted, df = prepare_metric_df(metric_df)

    subjects = list(df.index)
    subjects_per_page = nrows * ncols
    n_total = len(subjects)
    n_pages = int(np.ceil(n_total / subjects_per_page))

    # xlim / ylim
    x_lim = xlim if xlim is not None else (float(t_sorted.min()), float(t_sorted.max()))
    if ylim is not None:
        y_lim = ylim
    else:
        y_lim = None

    saved = []

    for page in range(n_pages):
        start = page * subjects_per_page
        end = min((page + 1) * subjects_per_page, n_total)
        sub_list = subjects[start:end]

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(ncols * 3.2, nrows * 3.2),
            sharex=True, sharey=True
        )
        axes = np.array(axes).reshape(-1)

        for k, ax in enumerate(axes):
            if k >= len(sub_list):
                ax.axis("off")
                continue

            sub = sub_list[k]
            y = df.loc[sub].to_numpy()

            ax.plot(t_sorted, y)
            _apply_log_x_pretty(ax, x_lim)

            if y_lim is not None:
                ax.set_ylim(*y_lim)

            ax.set_title(f"Subject {sub}", fontsize=14)

        if yticks_all:
            for ax in axes:
                if not ax.axison:
                    continue
                ax.tick_params(axis="y", which="both", labelleft=True, labelsize=11)

        if xticks_all:
            for ax in axes:
                if not ax.axison:
                    continue
                ax.tick_params(axis="x", which="both", labelbottom=True, labelsize=11)

        fig.text(0.5, 0.04, "Ratio (log scale)", ha="center", fontsize=16)
        fig.text(0.04, 0.5, metric_name, va="center", rotation="vertical", fontsize=16)
        fig.subplots_adjust(
            left=0.08, right=0.99,
            bottom=0.10, top=0.94,
            wspace=0.25, hspace=0.55
            )

        out_png = os.path.join(output_dir, f"{metric_name}_page{page+1:02d}.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        saved.append(out_png)
        plt.show()
        plt.close(fig)

    return saved


SubjectID = Union[int, str]
def plt_metric_overlay(
    metric_df: pd.DataFrame,
    subjects: Sequence[SubjectID],
    metric_name: str,
    title: Optional[str] = None,
    show_aggregate: bool = True,
    aggregate: str = "median",   # "mean" or "median"
    show_band: bool = False,
    band_quantiles: tuple[float, float] = (0.25, 0.75),
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    output_dir: Optional[str] = None,  # if None, figure won't be saved
    subject_labels: Optional[Mapping[SubjectID, str]] = None,
    alpha: Optional[float] = None,
    linewidth: float = 1.25,
) -> plt.Figure:

    t, df = prepare_metric_df(metric_df)
    x_lim = xlim if xlim is not None else (float(t.min()), float(t.max()))

    missing = [s for s in subjects if s not in df.index]
    if missing:
        raise KeyError(f"Subjects not found in metric_df.index: {missing}")

    if alpha is None:
        alpha = 0.25 if len(subjects) >= 10 else 0.7

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    Y = []
    for s in subjects:
        y = df.loc[s].to_numpy()
        Y.append(y)

        if len(subjects) <= 6:
            lab = subject_labels[s] if (subject_labels is not None and s in subject_labels) else f"Sub {s}"
        else:
            lab = "_nolegend_"

        ax.plot(t, y, alpha=alpha, linewidth=linewidth, label=lab)

    Y = np.vstack(Y)

    if show_band:
        qlo, qhi = band_quantiles
        lo = np.nanquantile(Y, qlo, axis=0)
        hi = np.nanquantile(Y, qhi, axis=0)
        ax.fill_between(t, lo, hi, alpha=0.15, label="_nolegend_")

    if show_aggregate:
        if aggregate == "mean":
            agg = np.nanmean(Y, axis=0)
            agg_label = "Mean"
        elif aggregate == "median":
            agg = np.nanmedian(Y, axis=0)
            agg_label = "Median"
        else:
            raise ValueError("aggregate must be 'mean' or 'median'")

        ax.plot(t, agg, linewidth=2.6, alpha=0.95, label=agg_label)

    _apply_log_x_pretty(ax, x_lim)

    ax.set_xlabel("Ratio (log scale)", fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    if title is not None:
        ax.set_title(title, fontsize=18)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if len(subjects) <= 6:
        ax.legend(frameon=True, fontsize=14)
    else:
        if show_aggregate:
            ax.legend(frameon=True, fontsize=14)

    ax.tick_params(axis="x", which="both", labelsize=13)
    ax.tick_params(axis="y", which="both", labelsize=13)
    fig.tight_layout()

    if output_dir is not None:
        out_png = os.path.join(output_dir, f"{metric_name}_{title}.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return fig


def prepare_metric_df(metric_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    metric_df: index=subject, columns=t（float or numeric str）
    return: (t_sorted, df_sorted_float)
    """
    t_vals = pd.to_numeric(metric_df.columns, errors="raise").to_numpy()
    if np.any(t_vals <= 0):
        bad = t_vals[t_vals <= 0]
        raise ValueError(f"t must be > 0 for log scale. Found: {bad}")

    df = metric_df.astype(float)

    return t_vals, df


def _apply_log_x_pretty(ax: plt.Axes, xlim: tuple[float, float]) -> None:
    ax.set_xscale("log")
    ax.set_xlim(*xlim)

    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.grid(True, which="major", linestyle="--", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.1)
