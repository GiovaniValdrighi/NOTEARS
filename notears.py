import numpy as np
import scipy.linalg as slin
import tensorflow as tf
import tensorflow_probability as tfp 

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
    def _h(W):
        '''
        Acyclicity constraint for the graph

        Args: 
            W (numpy.array): [n_variables, n_variables] graph

        Output:
            h (float) : restriction value 
            G_h (float) : restriction gradient
        '''
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
        '''
        Calculation of Loss
        Args:
            W (numpy.matrix) : [n_variables, n_variables] graph

        Output:
            loss (float) : Loss value
            G_loss (float) : Loss gradient value
        '''
        y = np.matmul(X, W)
        R = X - y
        # Mean squared error loss  : (1/2n)*(X - XW)^2
        loss = ((R**2)/(2*n_samples)).sum()
        G_loss = - np.matmul(X.T, R)/n_samples
        return loss, G_loss


    def _func(W, n):
        '''
        Evaluate value and gradient of augmented Lagrangian
        Args:
            W (numpy.matrix) : [n_variables, n_variables] graph

        Output:
            Scoring function value
        '''
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        #F(w) = L(W) + rho/2 * h² + alpha * h + lambda1*|W|
        f = loss + 0.5 * rho * h**2 + alpha * h + lambda1 * W.sum()
        #dF/dW = dL/dW(W) + rho * h * dh/dW + alpha * dh/dW
        G_f = G_loss + (rho * h + alpha) * G_h
        return tfp.math.value_and_gradient(f, G_f)

    n_samples, n_variables = X.shape
    W_est, alpha, rho, h = np.zeros((n_variables, n_variables)), 0.0, 1.0, np.inf
    for _ in range(max_iter):
        while rho < rho_max:
            W_est = tfp.optimizer.lbfgs_minimize(_func, W_est)
            h_new, _ = _h(W_est)

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
