import numpy as np
import pytest
from funcs.core import *

def test_create_dissim_amy():
    # 入力、出力はpd
    # NaNを含んでいる場合warn
    # 行列の要素がdissimの正しい場所に移されている
    # 対称行列でない場合error
    # dim=2
    assert 


def test_cal_dissim2dist():

    # 入力、出力はpd
    # NaNを含んでいない
    # 行列の要素がdist_matの正しい場所に移されている
    # dim=2
    # 入力の対角成分が0でなかった場合にwarn
    # 行列が正方行列でなかった場合にwarn


def test_cal_dist2sim
    # 入力、出力はpd
    # NaN, infを含んでいない
    # 行列の要素がsim_matの正しい場所に移されている
    # dim=2
    # 入力の対角成分が0でなかった場合にwarn
    # 行列が正方行列でなかった場合にwarn
    # t>0でない場合にerror


def test_cal_sim2genmag
    # 入力はpd, 出力はfloat
    # NaN, infを含んでいない
    # 行列の要素がsim_matの正しい場所に移されている
    # dim=2
    # 対角成分が1でなかった場合にwarn
    # 対称行列でなかった場合にwarn


def test_cal_dist2spread
    # 入力はpd, 出力はfloat
    # NaN, infを含んでいない
    # 行列の要素がsim_matの正しい場所に移されている
    # dim=2
    # 対角成分が1でなかった場合にwarn (Spreadが定義される距離空間の定義から外れる)
    # 対称行列でなかった場合にwarn (spreadの定義から外れる)
    # 行列が正方行列でなかった場合にwarn (spreadの定義から外れる)
    # t>0でない場合にerror