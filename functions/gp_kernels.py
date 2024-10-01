import jax.numpy as jnp
from jax.scipy import linalg

################################################################################
################################################################################

def RBF_kernel(X, Xp, hyp):
    '''
    Return the RBF kernel given X and kernel hyperparameters.
    Input: 
        - X: np.array of shape (N, d)
        - hyp: (length_scale, variance)
    '''
    # unpack hyperparameters
    length_scale, variance = hyp[0], hyp[1]
    # compute the distance matrix
    dist_matrix = X[:, jnp.newaxis, :] - Xp[jnp.newaxis, :, :]
    # jnp.sum(dist_matrix**2, axis=-1)/l**2 is 
    # equivalent to cdists(X/l,X/l,metric='sqeuclidean') 
    # compute the kernel
    K = variance * jnp.exp(-0.5 * jnp.sum(dist_matrix**2, axis=-1) / length_scale**2)
    # K = jnp.exp(-jnp.sum(dist_matrix**2, axis=-1) / length_scale**2) 
    return K

################################################################################
################################################################################

def matern_kernel(X, Xp, hyp):
    '''
    Return the Matern kernel given X and kernel hyperparameters.
    Input: 
        - X: np.array of shape (N, d)
        - hyp: (length_scale, variance)
    '''
    # unpack hyperparameters
    length_scale, variance = hyp[0], hyp[1]
    # compute the distance matrix
    dist_matrix = X[:, jnp.newaxis, :] - Xp[jnp.newaxis, :, :]
    # compute the kernel
    K = variance * jnp.exp(-jnp.sqrt(3) * jnp.sum(jnp.abs(dist_matrix), axis=-1) / length_scale)
    return K

################################################################################
################################################################################