import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_dissim_amy(raw: pd.DataFrame, unique_words: np.array, sub_no: int = -1) -> pd.DataFrame: 
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
    dissim_mtx = pd.DataFrame(np.nan, index=unique_words, columns=unique_words)

    word1 = raw.iloc[0, :]
    word2 = raw.iloc[1, :]

    for i in range(len(word1)):
        w1 = word1.iloc[i]
        w2 = word2.iloc[i]

        if sub_no == -1:
            dissim_val = raw.iloc[2:, i].mean()
        else:
            row_idx = 2 + sub_no
            if row_idx >= raw.shape[0]:
                raise IndexError(f"no={sub_no} is out of range for {raw.shape[0]} rows.")

            dissim_val = raw.iloc[row_idx, i]

        dissim_mtx.at[w1, w2] = dissim_val
        dissim_mtx.at[w2, w1] = dissim_val

    # NaNのうち対角成分を0に置換、他にNaNがあれば警告
    for i in range(len(dissim_mtx)):
        for j in range(len(dissim_mtx[0])):
            if dissim_mtx[i, j].isna() == True:
                if i == j:
                    dissim_mtx[i, j] = 0
                else:
                    print(f"NaN in this Dissimilarity matrix: ({i}, {j})")

    return dissim_mtx


def viz_dissim(raw: pd.DataFrame, unique_words: np.array, sub_no: int=-1, save: bool=False):

    return None


def cal_dissim2dist(dissim_mtx: pd.DataFrame) -> pd.DataFrame:
    dist_mtx = dissim_mtx.copy()
    # for the time you want to change the definition of distance
    
    return dist_mtx

def cal_dist2sim(dist_mtx: pd.DataFrame, t: float) -> pd.DataFrame:

    if t <= 0:
        raise ValueError(f"t must be > 0, got {t}")

    dist = dist_mtx.to_numpy(dtype=float)

    if not np.isfinite(dist).all():
        raise ValueError("dist_mtx contains NaN or inf.")
    
    sim_mtx = np.exp(-1 * t * dist)
    
    return pd.DataFrame(sim_mtx, index=dist_mtx.index, columns=dist_mtx.columns)


def cal_sim2genmag(sim_mtx: pd.DataFrame) -> float:
    A = sim_mtx.to_numpy(dtype=float)

    if not np.isfinite(A).all():
        raise ValueError("sim_mtx contains NaN or inf.")
    if A.ndim != 2:
        raise ValueError(f"sim_mtx must be 2D, got ndim={A.ndim}")

    if warn:
        if A.shape[0] != A.shape[1]:
            print(
                "Warning: sim_mtx is not square. "
                "Generalized magnitude is computable via pseudoinverse, "
                "but interpretation as 'diversity' may be non-standard."
            )

        # 対称性チェック（近似）
        if A.shape[0] == A.shape[1]:
            if not np.allclose(A, A.T, atol=1e-10, rtol=1e-8):
                print(
                    "Warning: sim_mtx is not symmetric. "
                    "This can make the interpretation hard."
                )

        # 対角成分が1か
        diag = np.diag(A) if A.shape[0] == A.shape[1] else None
        if diag is not None:
            if not np.allclose(diag, 1.0, atol=1e-8, rtol=1e-8):
                print(
                    "Warning: All sim mtx diagonal entries are not 1."
                )
    
    
    A_pinv = np.linalg.pinv(A)
    genmag = np.sum(A_pinv)

    return genmag


def cal_dist2spread(dist_mtx: pd.DataFrame, t: float) -> float:
    
    D = dist_mtx.to_numpy(dtype=float)
    
    if t <= 0:
        raise ValueError(f"t must be > 0, got {t}")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"dist_mtx must be square, got shape {D.shape}")
    if not np.isfinite(D).all():
        raise ValueError("dist_mtx contains NaN or inf.")

    Z = np.exp(-1 * t * D)
    denom = Z.sum(axis=1)
    
    eps = 1e-15 #calculation safety
    denom = np.maximum(denom, eps)
    spread = float((1.0 / denom).sum())

    if denom<eps:
        print("Warning: calculation was near to zero-division.")
    
    return spread
    