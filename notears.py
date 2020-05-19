import numpy as np
import scipy.linalg as slin

def notears_linear(X, h_tol, threshold, lambda1, c = 0.25, rho_max = 1e16, max_iter = 100):
    '''
    Function that apply the NOTEARS algorithm in a linear model
    
    Args:
        X (numpy.matrix) : [n_samples, n_variables] samples matrix 
        h_tol (float) : tolerance for constraint, exit condition 
        threshold (float) : threshold for W_est edge values
        lambda1 (float) : L1 regularization parameter 
        rho_max (float) : max value for rho in augmented lagrangian
        max_iter (int) : max number of iterations

    Outputs:
        W_est (numpy.matrix): [n_variables, n_variables] estimated graph
    '''
    def _h(W, d):
        '''
        Acyclicity constraint for the graph

        Args: 
            W (numpy.array): [n_variables, n_variables] graph

        Output:
            h (float) : restriction value 
            G_h (float) : restriction gradient
        '''
        #(Zheng et al. 2018 NOTEARS)
        # h(W) = tr[exp(W*W)] - d
        #E = slin.expm(W*W)
        #h = np.trace(E) - d
        #(Yu et al. 2019 DAG-GNN)
        # h(w) = tr[(I + kA*A)^d] - d
        M = np.eye(d) + W*W/d
        E = np.linalg.matrix_power(M, d-1)
        h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _fun(W, n):
        '''
        Scoring function to be minimized, the loss is mean squared error with L1 regularization
        Args:
            W (numpy.matrix) : [n_variables, n_variables] graph

        Output:
            Scoring function value
        '''
        y = np.matmul(X, W)
        MSE = ((X - y)**2)/(2*n)
        L1 = lambda1*np.abs(W).sum()
        return MSE + L1

    n_samples, n_variables = X.shape
    W_est, alpha, rho, h = np.zeros((n_variables, n_variables)), 0.0, 1.0, np.inf
    for _ in range(max_iter):
        while rho < rho_max:
            W_est = sol()
            h_new, _ = _h(W_est, n_variables)

            #Ascent alpha
            alpha = alpha + rho*h_new

            #Updating rho constraint parameter
            if h_new > h*c:
                rho*=10
            else:
                break
        
        #Verifying constraint tolerance
        if h <= h_tol or rho >= rho_max:
            break
    W_est = W_est * (np.abs(W_est) > threshold)
    return W_est
