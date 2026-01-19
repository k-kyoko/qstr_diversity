from __future__ import annotations
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
import seaborn as sns
from typing import Optional, Sequence, Union, Mapping, Literal
from adjustText import adjust_text

SubjectID = Union[int, str]


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


def plt_mds(
    mds_res: Union[np.ndarray, pd.DataFrame],
    dissimilarity_matrix: pd.DataFrame,
    save_path: Union[str, Path] = "/home/jovyan/work/fig/temp/mds.png",
    point_groups: Sequence[tuple[slice, str]] = (
        (slice(0, 10), "Colors"),
        (slice(10, 23), "Emotions"),
    ),
    plot_line: bool = False,
) -> Path:
    """
    sklearn.manifold.MDS の embedding（mds_res）を 2D/3D で可視化して保存する。

    - 点の色は point_groups（slice ごとのグルーピング）に合わせて変える
    - plot_line=True の場合、各グループ内の点を同色で順に結ぶ
    - 2D: adjust_text でラベル衝突回避を試みる
    - 3D: adjust_text は非対応なので単純配置（ax.text）にする
    """

    X = np.asarray(mds_res)
    if X.ndim != 2:
        raise ValueError(f"mds_res must be 2D array-like (n_samples, n_dims). Got shape={X.shape}")

    n, d = X.shape
    if d not in (2, 3):
        raise ValueError(f"Only 2D or 3D supported. Got d={d}")

    labels = list(map(str, dissimilarity_matrix.index))
    if len(labels) != n:
        raise ValueError(f"Label count mismatch: len(labels)={len(labels)} vs n_samples={n}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- figure settings ---
    figsize = (7, 7)
    point_size = 40
    label_fontsize = 13
    tick_labelsize = 15
    title_fontsize = 24
    legend_fontsize = 18
    axis_labelsize = 13

    # --- figure/axes ---
    fig = plt.figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(111, projection="3d") if d == 3 else fig.add_subplot(111)

    # --- group scatter (colored) ---
    all_idx = np.arange(n)
    assigned = np.zeros(n, dtype=bool)

    for gi, (sl, name) in enumerate(point_groups):
        idx = all_idx[sl]
        if idx.size == 0:
            continue

        # グループの重複を禁止（意図しない色の上書きを避ける）
        if assigned[idx].any():
            raise ValueError("point_groups contains overlapping slices (a point belongs to multiple groups).")
        assigned[idx] = True

        color = f"C{gi}"
        if d == 3:
            ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2], s=point_size, color=color, label=name)
        else:
            ax.scatter(X[idx, 0], X[idx, 1], s=point_size, color=color, label=name)

        if plot_line and idx.size >= 2:
            if d == 3:
                ax.plot(X[idx, 0], X[idx, 1], X[idx, 2], lw=1, color=color, label="_nolegend_")
            else:
                ax.plot(X[idx, 0], X[idx, 1], lw=1, color=color, label="_nolegend_")

    # --- optional: points not covered by any group ---
    rest = all_idx[~assigned]
    if rest.size > 0:
        if d == 3:
            ax.scatter(X[rest, 0], X[rest, 1], X[rest, 2], s=point_size, color="0.6", label="Other")
        else:
            ax.scatter(X[rest, 0], X[rest, 1], s=point_size, color="0.6", label="Other")

    # --- labels ---
    if d == 2:
        texts = [ax.text(X[i, 0], X[i, 1], labels[i], fontsize=label_fontsize) for i in range(n)]
        adjust_text(
            texts,
            ax=ax,
            only_move={"points": "xy", "text": "xy"},
            arrowprops=dict(arrowstyle="->", lw=1.2),
        )
    else:
        for i in range(n):
            ax.text(X[i, 0], X[i, 1], X[i, 2], labels[i], fontsize=label_fontsize)

    # --- cosmetics (Warning回避: set_xticklabels/set_yticklabels を使わない) ---
    ax.tick_params(labelsize=tick_labelsize)
    ax.set_xlabel("Dim 1", fontsize=axis_labelsize)
    ax.set_ylabel("Dim 2", fontsize=axis_labelsize)
    if d == 3:
        ax.set_zlabel("Dim 3", fontsize=axis_labelsize)

    ax.legend(fontsize=legend_fontsize)
    ax.set_title(save_path.stem, fontsize=title_fontsize)

    # --- save/show ---
    fig.savefig(save_path, dpi=300)
    plt.show()

    return save_path


def plt_metric_single(
    metric_df: pd.DataFrame,
    subject: SubjectID,
    metric_name: str,
    output_dir: str,
    ylim: Optional[tuple[float, float]] = None,
    xlim: Optional[tuple[float, float]] = None,
    positive_definite_df: Optional[pd.DataFrame] = None,
    posdef_mode: Literal["two_color", "true_only"] = "two_color",
    figsize: tuple[float, float] = (7.2, 4.6),
    show: bool = True,
) -> str:
    """
    show plot of one subject.

    - x: t (log scale; _apply_log_x_pretty is spplied)
    - y: metric_df value
    - positive_definite_df is not None -> overlay scatter
    """
    os.makedirs(output_dir, exist_ok=True)
    t_sorted, df = prepare_metric_df(metric_df)

    if subject not in df.index:
        raise KeyError(f"Subject not found in metric_df.index: {subject}")

    # xlim / ylim
    x_lim = xlim if xlim is not None else (float(t_sorted.min()), float(t_sorted.max()))
    y_lim = ylim if ylim is not None else None

    # check consistency with positive definite df
    pos_df = None
    if positive_definite_df is not None:
        t_pos, pos_df = prepare_metric_df(positive_definite_df)
        if len(t_pos) != len(t_sorted) or not np.allclose(
            np.asarray(t_pos, float), np.asarray(t_sorted, float),
            rtol=0, atol=0
        ):
            raise ValueError("positive_definite_df columns (t) must match metric_df columns (t).")
        if subject not in pos_df.index:
            raise KeyError(f"Subject not found in positive_definite_df.index: {subject}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)

    y = df.loc[subject].to_numpy()
    ax.plot(t_sorted, y)

    _apply_log_x_pretty(ax, x_lim)

    if pos_df is not None:
        pos_row = pos_df.loc[subject].to_numpy()
        scatter_by_positive_definite(
            ax=ax,
            t_all=t_sorted,
            y=y,
            posdef_row=pos_row,
            mode=posdef_mode,
            s=22,
            alpha=0.9,
            zorder=3,
        )

    if y_lim is not None:
        ax.set_ylim(*y_lim)

    #ax.set_title(f"Subject {subject}", fontsize=14)
    ax.set_xlabel("Scale parameter t", fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)

    fig.tight_layout()

    out_png = os.path.join(output_dir, f"{metric_name}_sub{subject}.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return out_png


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
    positive_definite_df: Optional[pd.DataFrame] = None,
    posdef_mode: Literal["two_color", "true_only"] = "two_color",
) -> list[str]:

    os.makedirs(output_dir, exist_ok=True)
    t_sorted, df = prepare_metric_df(metric_df)

    pos_df = None
    if positive_definite_df is not None:
        t_pos, pos_df = prepare_metric_df(positive_definite_df)
        if len(t_pos) != len(t_sorted) or not np.allclose(
            np.asarray(t_pos, float), np.asarray(t_sorted, float),
            rtol=0, atol=0
        ):
            raise ValueError("positive_definite_df columns (t) must match metric_df columns (t).")

        missing_pos = [s for s in df.index if s not in pos_df.index]
        if missing_pos:
            raise KeyError(f"Subjects not found in positive_definite_df.index: {missing_pos}")

    subjects = list(df.index)
    subjects_per_page = nrows * ncols
    n_total = len(subjects)
    n_pages = int(np.ceil(n_total / subjects_per_page))

    # xlim / ylim
    x_lim = xlim if xlim is not None else (float(t_sorted.min()), float(t_sorted.max()))
    y_lim = ylim if ylim is not None else None

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

            if pos_df is not None:
                pos_row = pos_df.loc[sub].to_numpy()
                scatter_by_positive_definite(
                    ax=ax,
                    t_all=t_sorted,
                    y=y,
                    posdef_row=pos_row,
                    mode=posdef_mode,
                    s=22,
                    alpha=0.9,
                    zorder=3,
                )

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


def scatter_by_positive_definite(
    ax: Axes,
    t_all: np.ndarray,  # all t, row indices of positive_definite(boolean) df
    y: np.ndarray,
    posdef_row: Optional[np.ndarray],  # True/False, exact row of positive_definite(boolean) df
    mode: Literal["two_color", "true_only"] = "two_color",
    s: float = 22,
    alpha: float = 0.9,
    zorder: int = 3,
) -> None:
    """
    overlay scatter depends on the matrix is positive definite or not
    - if posdef_row: None -> Do nothing
    - mode:
        - "two_color": True=Blue, False=Red
        - "true_only": True=Blue (False is not plotted)
    """
    if posdef_row is None:
        return

    t_all = np.asarray(t_all, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(posdef_row)

    if t_all.shape != y.shape or p.shape != y.shape:
        raise ValueError(f"Shape mismatch: t{t_all.shape}, y{y.shape}, posdef{p.shape}")

    mask_valid_y = np.isfinite(y)
    mask_known_p = ~pd.isna(p)
    mask = mask_valid_y & mask_known_p
    if not np.any(mask):
        return

    p_bool = np.asarray(p[mask]).astype(bool)

    if mode == "true_only":
        m = mask & p_bool
        if np.any(m):
            ax.scatter(t[m], y[m], s=s, alpha=alpha, color="tab:blue",
                       linewidths=0, zorder=zorder)
        return

    if mode == "two_color":
        m_true = mask & p_bool
        m_false = mask & (~p_bool)

        if np.any(m_false):
            ax.scatter(t_all[m_false], y[m_false], s=s, alpha=alpha, color="tab:red",
                       linewidths=0, zorder=zorder)
        if np.any(m_true):
            ax.scatter(t_all[m_true], y[m_true], s=s, alpha=alpha, color="tab:blue",
                       linewidths=0, zorder=zorder)
        return

    raise ValueError("mode must be 'two_color' or 'true_only'")


def plt_metric_overlay(
    metric_df: pd.DataFrame,
    subjects: Sequence[SubjectID],
    metric_name: str,
    title: Optional[str] = None,
    show_aggregate: bool = True,
    aggregate: str = "median",   # "mean" or "median"
    show_band: bool = False,
    band_quantiles: tuple[float, float] = (0.25, 0.75),
    show_legend: bool = True,
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

    agg_label = None
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

    overlay_text = f"Overlay:{100*band_quantiles[0]}~{100*band_quantiles[1]}%, N={len(subjects)}"

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        overlay_handle = Line2D([], [], linestyle="none", marker=None, color="none", label=overlay_text)
        handles = [overlay_handle] + handles
        labels = [overlay_text] + labels
        ax.legend(handles=handles, labels=labels, frameon=True, fontsize=14)

    ax.tick_params(axis="x", which="both", labelsize=13)
    ax.tick_params(axis="y", which="both", labelsize=13)
    fig.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        safe_title = "" if title is None else str(title).replace("/", "_")
        out_png = os.path.join(output_dir, f"{metric_name}_{safe_title}_{aggregate}.png")
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
