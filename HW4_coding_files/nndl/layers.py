import numpy as np
import pdb

"""
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.
"""

def affine_forward(x, w, b):
    x_flat = x.reshape(x.shape[0], -1)
    out = x_flat @ w + b
    return out, (x, w, b)

def affine_backward(dout, cache):
    x, w, b = cache
    x_flat = x.reshape(x.shape[0], -1)
    dx = (dout @ w.T).reshape(x.shape)
    dw = x_flat.T @ dout
    db = np.sum(dout, axis=0)
    return dx, dw, db

def relu_forward(x):
    return np.maximum(0, x), x

def relu_backward(dout, cache):
    return dout * (cache > 0)

def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    running_mean = bn_param.get('running_mean', np.zeros(x.shape[1], dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(x.shape[1], dtype=x.dtype))
    cache = None

    if mode == 'train':
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        centered = x - mean
        inv_std = 1 / np.sqrt(var + eps)
        x_hat = centered * inv_std
        out = gamma * x_hat + beta
        cache = (centered, x_hat, inv_std, gamma)
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
    elif mode == 'test':
        out = gamma * (x - running_mean) / np.sqrt(running_var + eps) + beta
    else:
        raise ValueError('Invalid batchnorm mode: %s' % mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    return out, cache

def batchnorm_backward(dout, cache):
    centered, x_hat, inv_std, gamma = cache
    N = dout.shape[0]
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)

    # Go backward through the scale, normalization, variance, and mean.
    dx_hat = dout * gamma
    dvar = np.sum(dx_hat * centered, axis=0) * (-0.5) * inv_std**3
    dmean = -np.sum(dx_hat * inv_std, axis=0) - 2 * dvar * centered.mean(axis=0)
    dx = dx_hat * inv_std + 2 * centered * dvar / N + dmean / N
    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']
    if not 0 < p <= 1:
        raise ValueError('p is the keep probability and must be in (0, 1].')
    mask = None
    if mode == 'train':
        # A fixed local RNG makes gradient checks repeatable.
        rng = np.random.RandomState(dropout_param['seed']) if 'seed' in dropout_param else np.random
        mask = (rng.rand(*x.shape) < p).astype(x.dtype) / p
        out = x * mask
    elif mode == 'test':
        out = x
    else:
        raise ValueError('Invalid dropout mode: %s' % mode)
    # Save the mode used for this pass, even if the caller changes it later.
    cache = (dropout_param.copy(), mask)
    return out.astype(x.dtype, copy=False), cache

def dropout_backward(dout, cache):
    dropout_param, mask = cache
    if dropout_param['mode'] == 'train':
        return dout * mask
    return dout

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
    # Work with log probabilities so large scores don't cause log(0).
    shifted = x - x.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    N = x.shape[0]
    loss = -log_probs[np.arange(N), y].mean()
    dx = np.exp(log_probs)
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
