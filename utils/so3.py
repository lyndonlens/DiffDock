import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation


MIN_EPS, MAX_EPS, N_EPS = 0.01, 2, 1000
X_N = 2000

"""
    ref to Equation3 in Diffdock paper
    Preprocessing for the SO(3) sampling and score computations, truncated infinite series are computed and then
    cached to memory, therefore the precomputation is only run the first time the repository is run on a machine
    # 此函数目的是预计算并存储SO3群旋转扩散kernel的采样密度，即文献中方程3，IGSO(3)分布。由于该kernel没有显式表达式，所以需要预先计算存储. 且只需要在模型第一次运行的时候计算一次即可。
    # p(w)是sigma和w的函数，所以结果是个2维数组，横坐标代表eps的取值（0.01-2之间），纵坐标代表w的取值，即旋转角度（0-pi）
    # eps 就是 sigma
"""

omegas = np.linspace(0, np.pi, X_N + 1)[1:] # angles


def _compose(r1, r2):  # R1 @ R2 but for Euler vecs
    return Rotation.from_matrix(Rotation.from_rotvec(r1).as_matrix() @ Rotation.from_rotvec(r2).as_matrix()).as_rotvec()


def _expansion(omega, eps, L=2000):  # 方程3，f(w), the summation term only
    p = 0
    for l in range(L):
        p += (2 * l + 1) * np.exp(-l * (l + 1) * eps**2) * np.sin(omega * (l + 1 / 2)) / np.sin(omega / 2)
    return p


# 方程3的第一部分，总和
def _density(expansion, omega, marginal=True):  # if marginal, density over [0, pi], else over SO(3)
    if marginal:
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        return expansion / 8 / np.pi ** 2  # the constant factor doesn't affect any actual calculations though


# 文献方程3下面那行的delta(ln_p(R'/R))=d_log(f(w))/dw
def _score(exp, omega, eps, L=2000):  # score of density over SO(3)
    dSigma = 0
    for l in range(L):
        hi = np.sin(omega * (l + 1 / 2)) # 公式3分子
        dhi = (l + 1 / 2) * np.cos(omega * (l + 1 / 2))
        lo = np.sin(omega / 2) # 公式3分子
        dlo = 1 / 2 * np.cos(omega / 2)
        dSigma += (2 * l + 1) * np.exp(-l * (l + 1) * eps**2) * (lo * dhi - hi * dlo) / lo ** 2 # derivative of f(w)。 方程三对w的导数
    return dSigma / exp # d(f(w))/dw/f(w)，也就是d(ln(f(w)))/dw


if os.path.exists('.so3_omegas_array2.npy'):
    _omegas_array = np.load('.so3_omegas_array2.npy')
    _cdf_vals = np.load('.so3_cdf_vals2.npy')
    _score_norms = np.load('.so3_score_norms2.npy')
    _exp_score_norms = np.load('.so3_exp_score_norms2.npy')
else:
    print("Precomputing and saving to cache SO(3) distribution table")
    _eps_array = 10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS) # sigma序列在0.01-2之间
    _omegas_array = np.linspace(0, np.pi, X_N + 1)[1:] # w在0-pi上

    _exp_vals = np.asarray([_expansion(_omegas_array, eps) for eps in _eps_array]) # f(eps, w), 结果为n_eps*n_w。一个sigma产生一个w分布
    _pdf_vals = np.asarray([_density(_exp, _omegas_array, marginal=True) for _exp in _exp_vals]) # p(w), n_eps*n_w。一个sigma产生一个密度分布
    _cdf_vals = np.asarray([_pdf.cumsum() / X_N * np.pi for _pdf in _pdf_vals]) # 在w方向上求累积和，归一化。最大=1*pi. n_eps*n_w
    _score_norms = np.asarray([_score(_exp_vals[i], _omegas_array, _eps_array[i]) for i in range(len(_eps_array))]) # 一个sigma对应一个f(w)。对每个sigma求对应的转移核，i是对sigma遍历

    _exp_score_norms = np.sqrt(np.sum(_score_norms**2 * _pdf_vals, axis=1) / np.sum(_pdf_vals, axis=1) / np.pi) # 对应每个sigma的expansion score_norm，n_sigma*1个数字
    # 像是求了一下二阶矩的开根号。即对每个sigma，对应一个score模值。要根据sigma计算整个S均值

    np.save('.so3_omegas_array2.npy', _omegas_array)
    np.save('.so3_cdf_vals2.npy', _cdf_vals) # 1000*2000
    np.save('.so3_score_norms2.npy', _score_norms)
    np.save('.so3_exp_score_norms2.npy', _exp_score_norms)


def sample(eps): # 给定sigma，从该sigma对应的w密度分布采样一个w
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    x = np.random.rand() # uniform sampling in [0, 1]
    return np.interp(x, _cdf_vals[eps_idx], _omegas_array) # 采样出w，0-pi


def sample_vec(eps): # 采样旋转轴矢量：归一化的矢量即为旋转轴，该矢量的模即为旋转角
    x = np.random.randn(3) # 1*3
    x /= np.linalg.norm(x) # 旋转轴矢量，首先需要归一化
    return x * sample(eps) # 欧拉矢量，x是（归一化的）旋转轴，sample(eps)指定了旋转角的大小。欧拉矢量使用3个参数表征一个SO3旋转


# 将一个vec(其归一化向量代表欧拉轴，其模长代表旋转角度)。这里保持轴不变，但是把角度重新插值计算一下
def score_vec(eps, vec):
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    om = np.linalg.norm(vec) # 欧拉向量的模就是旋转角，0-pi。下一步将该角度在eps对应的p(w)上插值计算对应的预存模长，而不是使用vec自己的原始模长
    return np.interp(om, _omegas_array, _score_norms[eps_idx]) * vec / om # np.interp(x, xp, fp)
    # vec / om 就是归一化的旋转轴。


def score_norm(eps): # 输入一个eps数值，0.01-2之间，插值计算对应的score_norm值，采出的是一个score(eps,w)值
    eps = eps.numpy()
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS # 0.01-2间的位置，总共1000个点
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS-1) # 0-1000之间
    return torch.from_numpy(_exp_score_norms[eps_idx]).float()

