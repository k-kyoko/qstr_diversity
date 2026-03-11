import numpy as np
import pandas as pd
from inspect import currentframe
from typing import Optional
import warnings


def create_dissim_amy(raw: pd.DataFrame, sub_no:int=-1, unique_words: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Amy式データフレームから単語のdissimilarity matrixを作成する関数
    For the symmetry matrix
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


import pandas as pd
from typing import Union, Sequence, Optional


def create_dissim_angus(
    raw: pd.DataFrame,
    sub_no: Union[int, Sequence[int]],
    unique_words: Optional[Sequence] = None,
    *,
    col1_name: str = "col1",
    col2_name: str = "col2",
    pid_name: str = "pID",
    dist_name: str = "dist",
):
    """
    Construct a dist matrix from a raw DataFrame, using unique values of col1 and col2 as row and column labels.

    Parameters
    ----------
    raw : pd.DataFrame A DataFrame containing [col1_name, col2_name, pid_name, dist_name].
    sub_no : int or sequence of int
        - int -> the matrix for that participant.
        - sequence of int -> a list of matrices, one for each participant.
        - -1 -> Constructs matrices for all pIDs and returns their average matrix.
    unique_words : sequence, optional
        If provided, used as the row and column labels of the matrix.
    col1_name, col2_name, pid_name, dist_name : str
        Column names in the input DataFrame.
    ----------
    -> return pd.DataFrame or list[pd.DataFrame]
    """

    required_cols = {col1_name, col2_name, pid_name, dist_name}
    missing = required_cols - set(raw.columns)
    if missing:
        raise ValueError(f"raw is missing required columns: {missing}")

    def _get_axis_labels(df: pd.DataFrame):
        if unique_words is not None:
            return list(unique_words)
        labels = pd.unique(
            pd.concat([df[col1_name], df[col2_name]], ignore_index=True)
        )
        try:
            return sorted(labels)
        except TypeError: # If mixed types prevent sorting, preserve the order of appearance
            return list(labels)

    def _make_one_matrix(df_sub: pd.DataFrame, axis_labels):
        mat = df_sub.pivot(
            index=col1_name,
            columns=col2_name,
            values=dist_name
        )
        mat = mat.reindex(index=axis_labels, columns=axis_labels)
        return mat

    axis_labels = _get_axis_labels(raw)

    # Case: average across all participants
    if sub_no == -1:
        all_pids = raw[pid_name].dropna().unique()
        matrices = []

        for pid in all_pids:
            df_sub = raw[raw[pid_name] == pid]
            mat = _make_one_matrix(df_sub, axis_labels)
            matrices.append(mat)

        if len(matrices) == 0:
            raise ValueError("No valid pID values were found for averaging.")

        avg_mat = sum(matrices) / len(matrices)
        return avg_mat

    # Case: multiple participants
    if isinstance(sub_no, Sequence) and not isinstance(sub_no, (str, bytes)):
        result = []
        for pid in sub_no:
            df_sub = raw[raw[pid_name] == pid]
            mat = _make_one_matrix(df_sub, axis_labels)
            result.append(mat)
        return result

    # Case: single participant
    df_sub = raw[raw[pid_name] == sub_no]
    
    return _make_one_matrix(df_sub, axis_labels)
    

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


def check_positivedefinite(A: np.array) -> bool:
    # check positive definiteness
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False  # semi-positive definite is here
    return False


def find_jump_near_target(
    df: pd.DataFrame,
    subj: int,
    target: float = 15.0,
    band: float = 3.0,
    pad: int = 5,
    topk: int = 10,
    verbose: bool = True,
) -> dict:
    # target付近のyでの不連続点を検出
    # --- 1. データ準備 (抽出・変換・クリーニング) ---
    s = pd.to_numeric(df.loc[subj], errors="coerce")
    s.index = pd.to_numeric(s.index, errors="coerce")
    s = s.dropna().sort_index()
    
    x, y = s.index.to_numpy(), s.to_numpy()
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid_mask], y[valid_mask]

    if len(x) < 2:
        raise ValueError("有効なデータ点が不足しています（2点未満）。")

    # --- 2. 差分計算と探索候補の絞り込み ---
    dy = np.diff(y)
    ady = np.abs(dy)
    lo_y, hi_y = target - band, target + band

    # 区間の最小値が上限以下 かつ 最大値が下限以上 なら交差している
    seg_min, seg_max = np.minimum(y[:-1], y[1:]), np.maximum(y[:-1], y[1:])
    cand_indices = np.flatnonzero((seg_min <= hi_y) & (seg_max >= lo_y))

    if cand_indices.size == 0:
        raise ValueError(f"範囲 [{lo_y:.3g}, {hi_y:.3g}] を横切る区間が見つかりません。")

    # --- 3. Top-K の抽出 ---
    # 候補内での絶対値ジャンプが大きい順にソート
    sorted_cands = cand_indices[np.argsort(ady[cand_indices])[::-1]]
    best_idx = sorted_cands[0]
    top_indices = sorted_cands[:topk]

    # 結果作成用のヘルパー関数
    def make_jump_info(idx, rank=None):
        return {
            "rank": rank,
            "x_left": float(x[idx]),   "x_right": float(x[idx+1]),
            "y_left": float(y[idx]),   "y_right": float(y[idx+1]),
            "delta_y": float(dy[idx]), "abs_delta_y": float(ady[idx]),
        }

    # --- 4. 結果の構築 ---
    # 表示用範囲（候補全体のインデックス範囲 ± pad）
    disp_lo = max(0, int(cand_indices.min()) - pad)
    disp_hi = min(len(y) - 1, int(cand_indices.max()) + 1 + pad)
    
    # 最大ジャンプ周辺データ
    w = 3
    around_sl = slice(max(0, best_idx - w), min(len(x), best_idx + w + 2))

    best_jump = make_jump_info(best_idx)
    result = {
        "subj": subj,
        "target": target, "band": band, "pad": pad,
        "band_range_y": (lo_y, hi_y),
        "search_range_idx_display": (disp_lo, disp_hi),
        "candidate_diff_indices": cand_indices,
        **best_jump,  # best_jumpの中身を展開して統合
        "top_jumps_local": [make_jump_info(i, r+1) for r, i in enumerate(top_indices)],
        "around": pd.DataFrame({"x": x[around_sl], "y": y[around_sl]}),
    }

    if verbose:
        print(f"[subj={subj}] band_y=[{lo_y:.6g}, {hi_y:.6g}] | candidates: {cand_indices.size}")
        print(f">> Max Jump: x={best_jump['x_left']:.4g}->{best_jump['x_right']:.4g}, "
              f"dy={best_jump['delta_y']:.4g}")
        print("\n-- Around Points --\n", result["around"].to_string(index=False))
        print("\n-- Top Jumps --\n", pd.DataFrame(result["top_jumps_local"]).to_string(index=False))

    return result


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