import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text


#aw: pd.DataFrame, unique_words: np.array, sub_no: int=-1, save: bool=False
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


def plt_mds(mds_res: pd.DataFrame, dissimilarity_matrix: pd.DataFrame,
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
