import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt

def notears_linear(X,  lambda1, threshold = 0.01, h_tol = 1e-10, c = 0.25, rho_max = 1e16, max_iter = 100):
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

    def _adj(W):
        '''Transforms an array [2 * n^2] to matrix [n, n]'''
        return (W[:n_variables*n_variables] - W[n_variables*n_variables:]).reshape([n_variables,n_variables])

    def _h(W):
        ''' Acyclicity constraint value for the graph and gradient '''
        #(Zheng et al. 2018 NOTEARS)
        # h(W) = tr[exp(W*W)] - n_variable
        #E = slin.expm(W*W)
        #h = np.trace(E) - n_variables
        #(Yu et al. 2019 DAG-GNN)
        # h(w) = tr[(I + kA*A)^n_variables] - n_variables
        M = np.eye(n_variables) + W*W/n_variables
        E = np.linalg.matrix_power(M, n_variables-1)
        h = (E.T * M).sum() - n_variables
        G_h = E.T * W * 2
        return h, G_h

    def _loss(W):
        '''Calculation of loss value and gradient'''
        R = X - np.matmul(X, W)
        # Mean squared error loss  : (1/2n)*(X - XW)^2
        loss = (R**2).sum()/(2*n_samples)
        G_loss = - np.matmul(X.T, R)/n_samples
        return loss, G_loss

    def _func(W):
        ''' Evaluate value and gradient of augmented Lagrangian '''
        W  = _adj(W)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        #F(w) = L(W) + rho/2 * h² + alpha * h + lambda1*|W|
        f = loss + 0.5 * rho * h**2 + alpha * h + lambda1 * W.sum()
        #dF/dW = dL/dW(W) + rho * h * dh/dW + alpha * dh/dW
        G_f = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_f + lambda1, - G_f + lambda1), axis=None)
        return f, g_obj


    ########################
    # Optimization process #
    ########################

    n_samples, n_variables = X.shape
    W_est, alpha, rho, h = np.zeros([ 2 * n_variables**2]), 0.0, 1.0, np.inf
    bnd = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(n_variables) for j in range(n_variables)]
    for _ in range(max_iter): 
        W_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, W_est, method = "L-BFGS-B", jac = True, bounds= bnd)
            W_new = sol.x
            h_new, _ = _h(_adj(W_new))

            #Updating rho constraint parameter
            if h_new > h*c:
                rho*=10
            else:
                break
        W_est, h = W_new, h_new
        
        #Ascent alpha
        alpha = alpha + rho*h

        #Verifying constraint tolerance
        if h <= h_tol or rho >= rho_max:
            break

    #Applying threshold    
    W_est = W_est * (W_est > threshold)
    W_est = _adj(W_est)
    return W_est