import numpy as np
import pandas as pd
from inspect import currentframe
from typing import Optional
import warnings


def create_dissim_amy(raw: pd.DataFrame, sub_no:int=-1, unique_words: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Amy式データフレームから単語のdissimilarity matrixを作成する関数
    For symmetry matrix
    Parameters:
    - raw (pd.DataFrame): 
    - unique_words (np.array):
    - sub_no (int): -1→全体のmeanをとる それ以外：その番号を参照

    Return:
    - dissim_mtx (pd.DataFrame)
    """

    word1 = raw.iloc[0]
    word2 = raw.iloc[1]

    if unique_words is None:
        all_words_combination = np.concatenate([word1, word2])
        unique_words = np.sort(np.unique(all_words_combination))

    dissim_mtx = pd.DataFrame(np.nan, index=unique_words, columns=unique_words)
    for i in range(len(word1)):
        w1 = word1.iloc[i]
        w2 = word2.iloc[i]

        if sub_no == -1:
            temp = raw.iloc[2:, i].astype(float)
            dissim_val = temp.mean()
        else:
            row_idx = 2 + sub_no
            if row_idx >= raw.shape[0]:
                raise IndexError(f"no={sub_no} is out of range for {raw.shape[0]} rows.")

            dissim_val = raw.iloc[row_idx, i]

        dissim_mtx.at[w1, w2] = float(dissim_val)
        dissim_mtx.at[w2, w1] = float(dissim_val)

    # NaNのうち対角成分を0に置換、他にNaNがあれば警告
    for i in range(len(dissim_mtx)):
        for j in range(len(dissim_mtx.iloc[0, :])):
            if pd.isna(dissim_mtx.iloc[i, j]):
                if i == j:
                    dissim_mtx.iloc[i, j] = 0
                else:
                    print(f"NaN in this Dissimilarity matrix: ({i}, {j})")

    return dissim_mtx


def cal_dissim2dist(dissim_mtx: pd.DataFrame) -> pd.DataFrame:
    # check input validity
    dissim = internal_check_input_instance(dissim_mtx)
    internal_check_basic_errors(0, dissim=dissim)

    # calculation
    dist_mtx = dissim_mtx.copy()  # for time you want to change the def of dist

    return dist_mtx


def cal_dist2sim(dist_mtx: pd.DataFrame, t: float) -> pd.DataFrame:
    # check input validity
    dist = internal_check_input_instance(dist_mtx)
    internal_check_basic_errors(0, dist=dist)
    internal_check_dist_nonnegative(dist)
    internal_check_t(t)

    # calculation
    sim_mtx = np.exp(-1 * t * dist)

    return pd.DataFrame(sim_mtx,
                        index=dist_mtx.index,
                        columns=dist_mtx.columns)


def cal_sim2genmag(sim_mtx: pd.DataFrame) -> float:
    # check input validity
    A = internal_check_input_instance(sim_mtx)
    internal_check_basic_errors(1, A=A)

    # calculation
    A_pinv = np.linalg.pinv(A)
    genmag = np.sum(A_pinv)

    return genmag


def cal_dist2spread(dist_mtx: pd.DataFrame, t: float) -> float:
    # check input validity
    D = internal_check_input_instance(dist_mtx)
    internal_check_basic_errors(0, D=D)
    internal_check_dist_nonnegative(D)
    internal_check_t(t)

    # calculation
    eps = 1e-15  # calculation safety

    Z = np.exp(-1 * t * D)
    denom = Z.sum(axis=1)
    denom = np.maximum(denom, eps)
    spread = float((1.0 / denom).sum())

    if denom.any() < eps:
        warnings.warn("Calculation was near to zero-division.")

    return spread


########## internal functions ##########

def internal_check_input_instance(pd_input):
    if isinstance(pd_input, pd.DataFrame):
        np_output = pd_input.to_numpy(dtype=float)
    else:
        raise ValueError(f"input must be pd.DataFrame, got {type(pd_input)}.")
    return np_output


def internal_check_basic_errors(diag_val, **inputs):
    for name, np_input in inputs.items():
        try:
            if not np.isfinite(np_input).all():
                raise ValueError(f"{name} contains NaN or inf.")
        except TypeError:
            warnings.warn(
                f"{name} has non-numeric dtype (dtype={np_input.dtype})")
        if np_input.shape[0] != np_input.shape[1]:
            warnings.warn(f"{name} is not square")
            continue  # guarantee square
        if not np.allclose(np.diag(np_input), diag_val):
            warnings.warn(f"diag vals are not {diag_val} in {name}")
        if not np.allclose(np_input, np_input.T):
            warnings.warn(f"{name} is not symmetric")


def internal_check_t(t):
    if t <= 0 or not np.isfinite(t):
        raise ValueError(f"t must be float value > 0, got {t}")


def internal_check_dist_nonnegative(dist_mtx):
    if np.any(dist_mtx < 0):
        raise ValueError("Distance matrix contains negative values.")


# 変数の名前と値をまとめてprint
def print_(*args):
    names = {id(v): k for k, v in currentframe().f_back.f_locals.items()}
    print('\n'.join([names.get(id(arg), '???') + ' = ' + repr(arg) for arg in args]))


# 変数の名前をprint
def print_name(*args):
    names = {id(v): k for k, v in currentframe().f_back.f_locals.items()}
    print('\n'.join([names.get(id(arg), '???') for arg in args]))