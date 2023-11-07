import numpy as np
import tqdm
import os

"""
    Preprocessing for the SO(2)/torus sampling and score computations, truncated infinite series are computed and then
    cached to memory, therefore the precomputation is only run the first time the repository is run on a machine
"""

# x是theta角与前一个theta角的差值，以pi为单位
def p(x, sigma, N=10): # sum(exp(-(x+2*pi*i)^2/2/sigma^2))
    p_ = 0
    for i in tqdm.trange(-N, N + 1): # N是
        p_ += np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_

# gradient of p
def grad(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += (x + 2 * np.pi * i) / sigma ** 2 * np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi

x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi # 在10-5pi到pi间的delta_theta插值。类比于SO3的w
sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi #

if os.path.exists('.p.npy'):
    p_ = np.load('.p.npy')
    score_ = np.load('.score.npy')
else:
    print("Precomputing and saving to cache torus distribution table")
    p_ = p(x, sigma[:, None], N=100)
    np.save('.p.npy', p_)

    score_ = grad(x, sigma[:, None], N=100) / p_ # d(ln(p))
    np.save('.score.npy', score_)

# x是弧度值，sigma是方差，这里将x和sigma根据规则进行插值重新计算score
def score(x, sigma): #
    x = (x + np.pi) % (2 * np.pi) - np.pi # -pi->-pi; -pi/2->-pi/2; 0->0; pi/2->pi/2; pi->-pi。 将所有x变换到0-pi之间
    sign = np.sign(x)
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int) # 获得在离散表格上的位置
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return -sign * score_[sigma, x]


def p(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return p_[sigma, x]


def sample(sigma): # 根据sigma采出对应的角度
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out


score_norm_ = score(
    sample(sigma[None].repeat(10000, 0).flatten()),
    sigma[None].repeat(10000, 0).flatten()
).reshape(10000, -1)
score_norm_ = (score_norm_ ** 2).mean(0)


def score_norm(sigma):
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return score_norm_[sigma]
