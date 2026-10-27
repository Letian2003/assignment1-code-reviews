import numpy as np


def matrix_operations(A=None, B=None):
    A = np.array([[3, 4], [2, 1]] if A is None else A, dtype=float)
    B = np.array([[1, 2], [0, -1]] if B is None else B, dtype=float)
    return A @ B, B @ A, (A + B).T, A @ B.T, np.trace(A), np.trace(A @ B)


def eigensystem(M=None):
    M = np.array([[2, -1], [-1, 2]] if M is None else M, dtype=float)
    return np.linalg.eigh(M)


def matrix_inverse(C=None):
    C = np.array([[4, 7], [2, 6]] if C is None else C, dtype=float)
    return np.linalg.inv(C)


def distribution_moments(rate=3.):
    return np.exp(-rate), rate, rate


def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    p = p_a_given_b * p_b
    return p, p / p_a, p_a + p_b - p


def coin_estimates(heads=3, tosses=5):
    return heads / tosses, (heads + 1) / (tosses + 1)


def gaussian_mle(data=None):
    x = np.array([1, 3, 5, 7] if data is None else data, dtype=float)
    return x.mean(), x.var()


def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    v, m, t = observation_variance, prior_mean, prior_variance
    return (x / v + m / t) / (1 / v + 1 / t)


def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    m = [2, -1] if mean is None else mean
    c = [[3, 1.2], [1.2, 1]] if covariance is None else covariance
    return np.random.default_rng(seed).multivariate_normal(m, c, n_samples)


def pca(X):
    x = np.asarray(X, dtype=float)
    m = x.mean(axis=0)
    z = x - m
    c = z.T @ z / (len(x) - 1)
    w, u = np.linalg.eigh(c)
    i = np.argsort(w)[::-1]
    w, u = np.maximum(w[i], 0), u[:, i]
    for j in range(u.shape[1]):
        if u[np.argmax(abs(u[:, j])), j] < 0:
            u[:, j] *= -1
    return dict(mean=m, centered=z, covariance=c, eigenvalues=w,
                components=u, scores=z @ u)


def explained_variance(eigenvalues):
    w = np.asarray(eigenvalues, dtype=float)
    return w / w.sum()


def make_plots(X, result):
    import matplotlib.pyplot as plt

    x = np.asarray(X, dtype=float)
    m, w, u = result['mean'], result['eigenvalues'], result['components']
    p = explained_variance(w)
    colors = ['tab:orange', 'tab:green']

    f1, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    ax.scatter(*x.T, s=16, alpha=.4, label='Samples')
    ax.scatter(*m, marker='x', c='black', s=70, label='Mean')
    for j, c in enumerate(colors):
        d = 2 * np.sqrt(w[j]) * u[:, j]
        ax.plot([m[0]-d[0], m[0]+d[0]], [m[1]-d[1], m[1]+d[1]], '--', c=c)
        ax.quiver(*m, *d, angles='xy', scale_units='xy', scale=1,
                  color=c, label=f'PC{j+1} ({p[j]:.1%})')
    ax.set(xlabel='Feature 1', ylabel='Feature 2', title='Data and PCs', aspect='equal')
    ax.legend()

    f2, ax = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    b = ax[0].bar(['PC1', 'PC2'], p * 100, color=colors)
    ax[0].bar_label(b, labels=[f'{v:.2%}' for v in p], padding=5)
    ax[0].set(ylim=(0, 105), ylabel='Explained variance (%)', title='Explained variance')
    z = result['scores'][:, :1] @ u[:, :1].T + m
    ax[1].scatter(*x.T, s=13, alpha=.3, label='Original')
    ax[1].scatter(*z.T, s=14, alpha=.6, c=colors[0], label='PC1 reconstruction')
    ax[1].set(xlabel='Feature 1', ylabel='Feature 2', title='2D to 1D', aspect='equal')
    ax[1].legend()
    return f1, f2


if __name__ == '__main__':
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = generate_data()
    r = pca(x)
    p = Path(__file__).resolve().parent / 'Assignment1_figures'
    p.mkdir(exist_ok=True)
    for f, name in zip(make_plots(x, r), ['data_pca.png', 'variance_reconstruction.png']):
        f.savefig(p / name, dpi=180)
        plt.close(f)
    print(r['eigenvalues'])
    print(explained_variance(r['eigenvalues']))
