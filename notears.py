import numpy as np
import scipy.linalg as slin

def notears_linear(W, alpha, c, tol, threshold):
    '''
    Function that apply the NOTEARS algorithm in a linear model
    Inputs:
    W - np.array: initial adjacency matrix
    alpha 
    c - int in (0,1): learning rate
    tol - int in (0, :]: tolerance
    threshold - int in (0, :]: threhsold 
    '''
    def _h(W):
        '''
        Acyclicity constraint for the graph
        Args: 
            W - numpy.array((d,d)): graph
        Output:
            h - int: restriction value 
            G_h - int: restriction gradient
        '''
        #(Zheng et al. 2018 NOTEARS)
        # h(W) = tr[exp(W*W)] - d
        #E = slin.expm(W*W)
        #h = np.trace(E) - d
        #(Ye et al. 2019 DAG-GNN)
        # h(w) = tr[(I + kA*A)^d] - d
        M = np.eye(d) + W*W/d
        E = np.linalg.matrix_power(M, d-1)
        h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def loss(W, alpha):
        pass
    
    for _ in range(max_iter):
        while 