import tensorflow as tf
import numpy as np
import scipy.optimize as sopt

class Notears_MLP(tf.keras.models.Model):
  '''
  Class for the neural network used on NOTEARS non-linear model
  
  Inputs:
    dims [int] - list of dimensions of hidden layers, the last dimension must be 1
    batch_size [int] - size of the batch for training
    bias [bool] - use of bias in model
  '''
  def __init__(self, dims, bias = True):
    super(Notears_MLP, self).__init__()
    self.dims = dims
    self.n_variables = dims[0]

    #fc1 layer [d * m0]
    self.fc1_pos = tf.keras.layers.Dense(dims[0] * dims[1], use_bias = bias, bias_initializer='glorot_uniform')
    self.fc1_neg = tf.keras.layers.Dense(dims[0] * dims[1], use_bias = bias, bias_initializer='glorot_uniform')

    #fc2 layers [d, m1, m2]
    self.locally = []
    for i in range(len(dims) - 2):
      self.locally.append(tf.keras.layers.LocallyConnected1D(dims[i + 2], 1, use_bias = bias, bias_initializer='glorot_uniform'))
    
    #activation
    self.sigmoid = tf.keras.activations.sigmoid

  def layers_shape(self):
    '''Utility function for val_and_grad'''
    shapes = []
    for l in self.trainable_variables:
      shapes.append(l.shape)
    return shapes

  def flat_params(self):
    '''Return flat vector of params in float64 to Scipy minimize method'''
    params = []
    for l in self.trainable_variables:
      params.append(tf.reshape(l, -1))
    return tf.concat(params, axis = 0).numpy().astype(np.float64)
  
  def flat_bounds(self):
    '''Return flat vector of bounds to Scipy minimize method'''
    bounds = []
    bounds_fc1 = []
    for i in range(self.n_variables):
      for m in range(self.dims[1]):
        for j in range(self.n_variables):
          if i == j:
            bounds_fc1.append((0.,0.))
          else:
            bounds_fc1.append((0., None))
    for _ in range(2):
      bounds.append(bounds_fc1)
      bounds.append([(None, None) for _ in range(tf.math.reduce_prod(self.fc1_pos.weights[1].shape))])
    
    for layer in self.locally:
      bounds.append([(None, None) for _ in range(tf.math.reduce_prod(layer.weights[0].shape))])
      bounds.append([(None, None) for _ in range(tf.math.reduce_prod(layer.weights[1].shape))])
    return sum(bounds, [])
  
  def flat_gradients(self, grad):
    grad_list = []
    for g in grad:
      if g.shape == (self.n_variables, self.dims[0] * self.dims[1]):
        grad_list.append(tf.transpose(tf.reshape(g, -1)))
      else:
        grad_list.append(tf.reshape(g, -1))
    return tf.concat(grad_list, axis = 0)

  def update_params(self, weights, flat = True):
    '''Function to update parameters from a flat params list'''
    if not flat:
      j = 0
      for layer in [self.fc1_pos, self.fc1_neg] + self.locally:
        layer.set_weights([weights[j], weights[j+1]])
        j+=2
      return
    #getting weights index
    shapes = [tf.math.reduce_prod(l) for l in self.layers_shape()]
    s_flat = [0] + [sum(shapes[:i+1]) for i in range(len(shapes))]
    #updating weights
    i = 0
    for layer in [self.fc1_pos, self.fc1_neg] + self.locally:
      layer.set_weights([np.reshape(weights[s_flat[i]:s_flat[i+1]], layer.weights[0].shape),
                         np.reshape(weights[s_flat[i+1]:s_flat[i+2]], layer.weights[1].shape)])
      i+=2

  def call(self, inputs):
    '''
    Forward procedure in the neural network, pass the inputs trought fc1 and fc2 layers

    Inputs:
      inputs [tensor] - tensor of samples with shape [batch_size, n_variables]

    Outputs:
      out [tensor] - tensor with shape [batch_size, n_variables]
    '''

    hid = self.fc1_pos(inputs) - self.fc1_neg(inputs) #[n, d * m0]
    out = tf.reshape(hid, (-1, self.n_variables, self.dims[1])) #[n, d, m0]
    for layer in self.locally:
      out = self.sigmoid(out)
      out = layer(out)
    out = tf.squeeze(out, 2) #[n, d, 1] -> [n, d]
    return out

  def _h(self):
    '''Calculate the constraint of fc1 to ensure that it's a DAG'''
    fc1_weights = tf.transpose(self.fc1_pos.weights[0]) - tf.transpose(self.fc1_neg.weights[0]) #[d, d * m0]
    fc1_weights = tf.reshape(fc1_weights, (self.n_variables, -1, self.n_variables)) #[d, m0, d]
    A = tf.transpose(tf.math.reduce_sum(tf.pow(fc1_weights, 2), axis = 1)) #[d, d]
    #(Yu et al. 2019 DAG-GNN)
    # h(w) = tr[(I + kA*A)^n_variables] - n_variables
    M = tf.eye(self.n_variables, num_columns = self.n_variables) + A/self.n_variables
    E = M
    for _ in range(self.n_variables - 2):
      E = tf.linalg.matmul(E, M)
    h = tf.math.reduce_sum(tf.transpose(E) * M) - self.n_variables
    return h
  
  def _l2_loss(self):
    '''Calculate L2 loss from model parameters'''
    loss = 0
    fc1_weights = self.fc1_pos.weights[0] - self.fc1_neg.weights[0]
    loss +=  tf.math.reduce_sum(tf.pow(fc1_weights, 2))
    for layer in self.locally:
      loss += tf.math.reduce_sum(tf.pow(layer.weights[0], 2))
    return loss

  def _l1_loss(self):
    '''Calculate L1 loss from fc1 parameters'''
    return tf.math.reduce_sum(self.fc1_pos.weights[0] + self.fc1_neg.weights[0])

  def _square_loss(self, X, Y):
    return 0.5 * tf.math.reduce_sum(tf.pow(X - Y, 2)) / X.shape[0]
    

  def loss(self, params, X, rho, alpha, lambda1, lambda2):
    '''Calculate model loss'''
    self.update_params(params.astype(np.float32))  #scipy send params as a np.float64
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.trainable_variables)
      Y = self(X)
      mse_ = self._square_loss(X, Y)
      h_ = self._h()
      h_loss = 0.5 * rho * h_ * h_ + alpha * h_
      l1_ = lambda1 * self._l1_loss()
      l2_ = 0.5 * lambda2 * self._l2_loss()
      loss = mse_ + h_loss + l1_ + l2_
    grad = tape.gradient(loss, self.trainable_variables)
    flat_grad = self.flat_gradients(grad)
    #scipy must recieve loss and grad as arrays np.float64
    return loss.numpy().astype(np.float64), flat_grad.numpy().astype(np.float64)

  def to_adj(self):
    '''Reshape fc1 to an adjacency matrix'''
    fc1_weights = tf.transpose(self.fc1_pos.weights[0]) - tf.transpose(self.fc1_neg.weights[0]) #[d * m0, d]
    fc1_weights = tf.reshape(fc1_weights, (self.n_variables, -1, self.n_variables)) #[d, m0, d]
    A = tf.transpose(tf.math.reduce_sum(fc1_weights * fc1_weights, axis = 1)) #[d, d]
    return tf.math.sqrt(A).numpy()
  
def notears_nonlinear(dims, X,  h_tol = 1e-8, threshold = 0.3, lambda1 = 0.5, lambda2 = 0.5, rho_max = 1e+16, max_iter = 100, init_params = None):
  '''
    Function that apply the NOTEARS algorithm in a non linear model
    
    Args:
        dims (int) : list of dimensions for neural network
        X (numpy.matrix) : [n_samples, n_variables] samples matrix 
        h_tol (float) : tolerance for constraint, exit condition 
        threshold (float) : threshold for W_est edge values
        lambda1 (float) : L1 regularization parameter
        lambda2 (float) : L2 regularization parameter 
        rho_max (float) : max value for rho in augmented lagrangian
        max_iter (int) : max number of iterations
    Outputs:
        W_est (numpy.matrix): [n_variables, n_variables] estimated graph
    '''
    
  ########################
  # Optimization process #
  ########################
  model = Notears_MLP(dims, True)
  model.build(X.shape)
  if init_params != None:
    model.update_params(init_params, flat = False)
  rho, alpha, h = 1., 0., np.inf
  for ind in range(int(max_iter)):
    h_new = None
    while rho < rho_max:
      sol = sopt.minimize(model.loss, model.flat_params(), args = (X, rho, alpha, lambda1, lambda2), 
                          method = "L-BFGS-B", jac = True, bounds = model.flat_bounds())
      new_params = sol.x
      model.update_params(new_params.astype(np.float32))
      h_new = model._h().numpy()

      #Updating rho constraint parameter
      if h_new > h * 0.25:
        rho = rho * 10
      else:
        break
    
    h = h_new

    #Ascent alpha
    alpha += rho * h

    #Verifying constraint tolerance
    if h < h_tol or rho >= rho_max:
      break
  #Applying threshold   
  W_est = model.to_adj()
  W_est[np.abs(W_est) < threshold] = 0
  return W_est  
