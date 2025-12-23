import numpy as np
import pandas as pd
from itertools import product
import pytest
from funcs.core import *


#def test_create_dissim_amy():
    # 入力、出力はpd
    # NaNを含んでいる場合warn
    # 行列の要素がdissimの正しい場所に移されている
    # 対称行列でない場合error
    # dim=2


########### dissim2dist ###########
@pytest.mark.parametrize("dissim_val", [1.0, 4.5])
def test_cal_dissim2dist_basic(dissim_val):
    # 入力pd, 出力はpdで形状が一致している
    # 行列の要素がsim_matの正しい場所に移されている
    # 入力の対角成分が0でなかった場合にwarn
    # 行列が正方行列でなかった場合にwarn
    # t>0でない, nan, infの場合にerror
    # distにNaN, infを含んだ場合にerror

    dissim = pd.DataFrame(
        [[0.0, dissim_val, 0.5],
         [dissim_val, 0.0, 4],
         [0.5, 4, 0.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"]
        )
    dist = cal_dissim2dist(dissim)

    assert isinstance(dist, pd.DataFrame)
    assert dist.shape == dissim.shape

    assert dist.index.equals(dissim.index)
    assert dist.columns.equals(dissim.columns)

    assert np.isclose(dist.loc["a", "b"], dissim_val)
    assert np.allclose(np.diag(dist.values), 0.0)


@pytest.mark.parametrize("bad_input", ["str", 1.0])
def test_cal_dissim2dist_instance_error(bad_input):
    with pytest.raises(ValueError):
        cal_dissim2dist(bad_input)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_cal_dissim2dist_nan_inf_error(bad_value):
    dissim = pd.DataFrame([[0.0, bad_value], [1.0, 0.0]])
    with pytest.raises(ValueError):
        cal_dissim2dist(dissim)


def test_cal_dissim2dist_diag_warning():
    dissim = pd.DataFrame([[0.5, 1.0], [1.0, 0.0]])
    with pytest.warns(UserWarning):
        dist = cal_dissim2dist(dissim)
    assert isinstance(dist, pd.DataFrame)  # 実行の確認


def test_cal_dissim2dist_shape_warning():
    dissim = pd.DataFrame([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    with pytest.warns(UserWarning):
        dist = cal_dissim2dist(dissim)
    assert dist.shape == dissim.shape


def test_cal_dissim2dist_symmetry_warning():
    dissim = pd.DataFrame([[0.0, 1.0, 2.0],
                           [1.0, 0.0, 3.0],
                           [0.0, 0.0, 0.0]])
    with pytest.warns(UserWarning):
        dist = cal_dissim2dist(dissim)
    assert dist.shape == dissim.shape


########## dist2sim ##########
@pytest.mark.parametrize(
    "dist_val, t",
    list(product([1.0, 4.5], [1.0, 100]))
)
def test_cal_dist2sim_basic(dist_val, t):
    # 入力pd, 出力はpdで形状が一致している
    # 行列の要素がsim_matの正しい場所に移されている
    # 入力の対角成分が0でなかった場合にwarn
    # 行列が正方行列でなかった場合にwarn
    # t>0でない, nan, infの場合にerror
    # distにNaN, infを含んだ場合にerror

    dist = pd.DataFrame(
        [[0.0, dist_val, 0.5],
         [dist_val, 0.0, 4],
         [0.5, 4, 0.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"]
        )
    sim = cal_dist2sim(dist, t)

    assert isinstance(sim, pd.DataFrame)
    assert sim.shape == dist.shape

    assert sim.index.equals(dist.index)
    assert sim.columns.equals(dist.columns)

    exp_1 = np.exp(-1 * dist_val * t)
    assert np.isclose(sim.loc["a", "b"], exp_1)
    assert np.allclose(np.diag(sim.values), 1.0)


@pytest.mark.parametrize("bad_input", ["str", 1.0])
def test_cal_dist2sim_instance_error(bad_input):
    with pytest.raises(ValueError):
        cal_dist2sim(bad_input, t=1.0)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_cal_dist2sim_nan_inf_error(bad_value):
    dist = pd.DataFrame([[0.0, bad_value], [1.0, 0.0]])
    with pytest.raises(ValueError):
        cal_dist2sim(dist, t=1.0)


@pytest.mark.parametrize("t", [0, -10.0, np.inf, np.nan])
def test_cal_dist2sim_t_error(t):
    dist = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        cal_dist2sim(dist, t=t)


def test_cal_dist2sim_diag_warning():
    dist = pd.DataFrame([[0.5, 1.0], [1.0, 0.0]])
    with pytest.warns(UserWarning):
        sim = cal_dist2sim(dist, t=1.0)
    assert isinstance(sim, pd.DataFrame)  # 実行の確認


def test_cal_dist2sim_shape_warning():
    dist = pd.DataFrame([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    with pytest.warns(UserWarning):
        sim = cal_dist2sim(dist, t=1.0)
    assert sim.shape == dist.shape


def test_cal_dist2sim_symmetry_warning():
    dist = pd.DataFrame([[0.0, 1.0, 2.0],
                         [1.0, 0.0, 3.0],
                         [0.0, 0.0, 0.0]])
    with pytest.warns(UserWarning):
        sim = cal_dist2sim(dist, 1.0)
    assert sim.shape == dist.shape


########### sim2genmag ###########
@pytest.mark.parametrize(
    "sim_val, ans",
    list(zip([0.0, 1.0, 0.5], [2.17647058824, 1.0, 2.15384615385])))
def test_cal_sim2genmag_basic(sim_val, ans):
    # 入力はpd, 出力はfloat
    # 入力の対角成分が1でなかった場合にwarn
    # 行列が正方行列でなかった場合にwarn
    # 行列が対称行列でなかった場合にwarn
    # t>0でない, nan, infの場合にerror
    # distにNaN, infを含んだ場合にerror

    sim = pd.DataFrame([[1.0, 0.0, sim_val],
                        [0.0, 1.0, 0.7],
                        [sim_val, 0.7, 1.0]],
                       index=["a", "b", "c"],
                       columns=["a", "b", "c"])
    genmag = cal_sim2genmag(sim)

    assert isinstance(genmag, float)
    assert np.isclose(genmag, ans)


@pytest.mark.parametrize("bad_input", ["str", 1.0])
def test_cal_sim2genmag_instance_error(bad_input):
    with pytest.raises(ValueError):
        cal_sim2genmag(bad_input)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_cal_sim2genmag_nan_inf_error(bad_value):
    sim = pd.DataFrame([[0.0, bad_value], [1.0, 0.0]])
    with pytest.raises(ValueError):
        cal_sim2genmag(sim)


def test_cal_sim2genmag_diag_warning():
    sim = pd.DataFrame([[0.5, 1.0], [1.0, 1.0]])
    with pytest.warns(UserWarning):
        genmag = cal_sim2genmag(sim)
    assert isinstance(genmag, float)  # 実行の確認


def test_cal_sim2genmag_shape_warning():
    sim = pd.DataFrame([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    with pytest.warns(UserWarning):
        genmag = cal_sim2genmag(sim)
    assert isinstance(genmag, float)


def test_cal_sim2genmag_symmetry_warning():
    sim = pd.DataFrame([[1.0, 0.5], [0.0, 1.0], [1.0, 1.0]])
    with pytest.warns(UserWarning):
        genmag = cal_sim2genmag(sim)
    assert isinstance(genmag, float)


########## cal_dist2spread ##########

@pytest.mark.parametrize(
    "dist_val, ans",
    list(zip([0.0, 1.0, 0.5], [#未計算])))
def test_cal_dist2spread_basic(dist_val, ans):
    # 入力はpd, 出力はfloat
    # 入力の対角成分が1でなかった場合にwarn
    # 行列が正方行列でなかった場合にwarn
    # 行列が対称行列でなかった場合にwarn
    # t>0でない, nan, infの場合にerror
    # distにNaN, infを含んだ場合にerror

    dist = pd.DataFrame([[0.0, 1.0, dist_val],
                        [1.0, 0.0, 0.7],
                        [dist_val, 0.7, 0.0]],
                       index=["a", "b", "c"],
                       columns=["a", "b", "c"])
    spread = cal_dist2spread(dist, 1.0)

    assert isinstance(spread, float)
    assert np.isclose(spread, ans)


@pytest.mark.parametrize("bad_input", ["str", 1.0])
def test_cal_dist2spread_instance_error(bad_input):
    with pytest.raises(ValueError):
        cal_dist2spread(bad_input, 1.0)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_cal_dist2spread_nan_inf_error(bad_value):
    dist = pd.DataFrame([[0.0, bad_value], [1.0, 0.0]])
    with pytest.raises(ValueError):
        cal_dist2spread(dist, 1.0)


@pytest.mark.parametrize("t", [0, -10.0, np.inf, np.nan])
def test_cal_dist2spread_t_error(t):
    dist = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        cal_dist2spread(dist, t=t)


def test_cal_dist2spread_diag_warning():
    dist = pd.DataFrame([[0.5, 1.0], [1.0, 1.0]])
    with pytest.warns(UserWarning):
        spread = cal_dist2spread(dist, 1.0)
    assert isinstance(spread, float)


def test_cal_dist2spread_shape_warning():
    dist = pd.DataFrame([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    with pytest.warns(UserWarning):
        spread = cal_dist2spread(dist, 1.0)
    assert isinstance(spread, float)


def test_cal_dist2spread_symmetry_warning():
    dist = pd.DataFrame([[1.0, 0.5], [0.0, 1.0], [1.0, 1.0]])
    with pytest.warns(UserWarning):
        spread = cal_dist2spread(dist, 1.0)
    assert isinstance(spread, float)