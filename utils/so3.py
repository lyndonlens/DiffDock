import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# 结果是个2维数组，横坐标代表eps的取值，纵坐标代表w的取值
MIN_EPS, MAX_EPS, N_EPS = 0.01, 2, 1000
X_N = 2000


"""
    ref to Equation3 in Diffdock paper
    Preprocessing for the SO(3) sampling and score computations, truncated infinite series are computed and then
    cached to memory, therefore the precomputation is only run the first time the repository is run on a machine
"""

omegas = np.linspace(0, np.pi, X_N + 1)[1:] # angles


def _compose(r1, r2):  # R1 @ R2 but for Euler vecs
    return Rotation.from_matrix(Rotation.from_rotvec(r1).as_matrix() @ Rotation.from_rotvec(r2).as_matrix()).as_rotvec()


def _expansion(omega, eps, L=2000):  # f(w), the summation term only
    p = 0
    for l in range(L):
        p += (2 * l + 1) * np.exp(-l * (l + 1) * eps**2) * np.sin(omega * (l + 1 / 2)) / np.sin(omega / 2)
    return p


def _density(expansion, omega, marginal=True):  # if marginal, density over [0, pi], else over SO(3)
    if marginal:
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        return expansion / 8 / np.pi ** 2  # the constant factor doesn't affect any actual calculations though


def _score(exp, omega, eps, L=2000):  # score of density over SO(3)
    dSigma = 0
    for l in range(L):
        hi = np.sin(omega * (l + 1 / 2))
        dhi = (l + 1 / 2) * np.cos(omega * (l + 1 / 2))
        lo = np.sin(omega / 2)
        dlo = 1 / 2 * np.cos(omega / 2)
        dSigma += (2 * l + 1) * np.exp(-l * (l + 1) * eps**2) * (lo * dhi - hi * dlo) / lo ** 2 # derivative of f(w)
    return dSigma / exp # f(w)导数除以f(w)，也就是d(ln(f(w)))/dw


if os.path.exists('.so3_omegas_array2.npy'):
    _omegas_array = np.load('.so3_omegas_array2.npy')
    _cdf_vals = np.load('.so3_cdf_vals2.npy')
    _score_norms = np.load('.so3_score_norms2.npy')
    _exp_score_norms = np.load('.so3_exp_score_norms2.npy')
else:
    print("Precomputing and saving to cache SO(3) distribution table")
    _eps_array = 10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS)
    _omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]

    _exp_vals = np.asarray([_expansion(_omegas_array, eps) for eps in _eps_array]) # f(eps, w), n_eps*n_w
    _pdf_vals = np.asarray([_density(_exp, _omegas_array, marginal=True) for _exp in _exp_vals]) # p(w), n_eps*n_w
    _cdf_vals = np.asarray([_pdf.cumsum() / X_N * np.pi for _pdf in _pdf_vals]) # 在w方向上求累积和，归一化。最大=1. n_eps*n_w
    _score_norms = np.asarray([_score(_exp_vals[i], _omegas_array, _eps_array[i]) for i in range(len(_eps_array))])

    _exp_score_norms = np.sqrt(np.sum(_score_norms**2 * _pdf_vals, axis=1) / np.sum(_pdf_vals, axis=1) / np.pi)
    # 像是求了一下二阶矩的开根号。即对每个sigma，对应一个score均值。要根据sigma计算整个S均值

    np.save('.so3_omegas_array2.npy', _omegas_array)
    np.save('.so3_cdf_vals2.npy', _cdf_vals) # 1000*2000
    np.save('.so3_score_norms2.npy', _score_norms)
    np.save('.so3_exp_score_norms2.npy', _exp_score_norms)


def sample(eps): # 随机采样一个eps（sigma），并基于此再采样一个
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    x = np.random.rand() # uniform sampling in [0, 1]
    return np.interp(x, _cdf_vals[eps_idx], _omegas_array) # 采样出w，0-pi


def sample_vec(eps):
    x = np.random.randn(3)
    x /= np.linalg.norm(x) # 旋转轴矢量，首先需要归一化
    return x * sample(eps) # 欧拉矢量，x是（归一化的）旋转轴，sample(eps)指定了旋转角的大小。欧拉矢量使用3个参数表征一个SO3旋转


def score_vec(eps, vec):
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    om = np.linalg.norm(vec) # 欧拉向量的模就是旋转角，0-pi
    return np.interp(om, _omegas_array, _score_norms[eps_idx]) * vec / om
    # vec / om 就是归一化的旋转轴。这里是采出一个score得分后将其向量化到旋转轴上


def score_norm(eps):
    eps = eps.numpy()
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS-1)
    return torch.from_numpy(_exp_score_norms[eps_idx]).float()

