import networkx as nx
import numpy as np
from scipy.special import expit as sigmoid
def simulate_dag(d, s0):
    """Simulate random DAG with some expected number of edges.
    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    
    Function based xunzheng/notears
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    G_und = nx.gnm_random_graph(d, s0)
    B_und = nx.to_numpy_matrix(G_und)
    B = _random_acyclic_orientation(B_und)
    B_perm = _random_permutation(B)

    assert nx.is_directed_acyclic_graph(nx.from_numpy_matrix(B_perm, create_using = nx.DiGraph))
    return B_perm

def simulate_nonlinear_sem(B, n, noise_scale=None):
    """Simulate samples from nonlinear SEM.
    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones
    Returns:
        X (np.ndarray): [n, d] sample matrix

    Function based on xunzheng/notears
    """
    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
          return z
        hidden = 100
        W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
        W1[np.random.rand(*W1.shape) < 0.5] *= -1
        W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
        W2[np.random.rand(hidden) < 0.5] *= -1
        x = sigmoid(X @ W1) @ W2 + z
        return x

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = nx.from_numpy_matrix(B, create_using = nx.DiGraph)
    ordered_vertices = nx.topological_sort(G)
    #assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j)
        X[:, j] = _simulate_single_equation(X[:, list(parents)], scale_vec[j])
    return X
