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
    G = nx.from_numpy_matrix(B.transpose(), create_using = nx.DiGraph)
    ordered_vertices = nx.topological_sort(G)
    #assert len(ordered_vertices) == d
    for j in reversed(list(ordered_vertices)):
        parents = G.neighbors(j)
        X[:, j] = _simulate_single_equation(X[:, list(parents)], scale_vec[j])
    return X

def simulate_linear_sem(B, n, noise_scale=None):
    """Simulate samples from linear SEM.
    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones
    Returns:
        X (np.ndarray): [n, d] sample matrix

    Function based on xunzheng/notears
    """
    def _simulate_single_equation(X, b, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
          return z
        x = X @ b + z
        return x

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = nx.from_numpy_matrix(B.transpose(), create_using = nx.DiGraph)
    ordered_vertices = nx.topological_sort(G)
    #assert len(ordered_vertices) == d
    for j in reversed(list(ordered_vertices)):
        parents = G.neighbors(j)
        parents = list(parents)
        X[:, j] = _simulate_single_equation(X[:, parents], B[parents, j], scale_vec[j])
    return X

def count_accuracy(G_true: nx.DiGraph,
                   G: nx.DiGraph,
                   G_und: nx.DiGraph = None) -> tuple:
    """Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.
    Args:
        G_true: ground truth graph
        G: predicted graph
        G_und: predicted undirected edges in CPDAG, asymmetric
    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    B_true = nx.to_numpy_array(G_true) != 0
    B = nx.to_numpy_array(G) != 0
    B_und = None if G_und is None else nx.to_numpy_array(G_und)
    d = B.shape[0]
    # linear index of nonzeros
    if B_und is not None:
        pred_und = np.flatnonzero(B_und)
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    if B_und is not None:
        B_lower += np.tril(B_und + B_und.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return fdr, tpr, fpr, shd, pred_size
