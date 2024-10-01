import jax
import jax.numpy as jnp
from jax.scipy import linalg
from .gp_kernels import RBF_kernel, matern_kernel

################################################################################
################################################################################

# @jax.jit
class GPModel():
    """
    GP class for fitting implemented in JAX.
    This is a GPModel for a supplied kernel, with white noise.
    Very much inspired by the simple Rasmussen and Williams implementation.
    This has been checked against the scikit-learn implementation which also
    follows R&W.

    White noise kernel:
        - k(x_i, x_j) = noise\_level \text{ if } x_i == x_j \text{ else } 0
    """

    ############################################################################
    ############################################################################

    def __init__(self):
        """ 
        Setup a GP regressor but keep it real light at this stage and 
        then we can implement felixibility later
        """
        # setup an RBF kernel 
        self.kernel = matern_kernel
        self.hyp = (1,1)
        self.Kchol = None # cholesky decomposition
        self.alpha = None # alpha vector
        self.X = None # store the training data
        self.white_noise = True # add white noise to the kernel where xi==xj
        self.y_scale = True # scale the y data

    ############################################################################
    ############################################################################
 
    def fit(self, X, Y, hyp):
        # store the hyperparameters
        self.hyp = hyp

        # scale the y data if requested
        if self.y_scale:
            self.y_mean = jnp.mean(Y)
            self.y_std = jnp.std(Y)
            y_scale = (Y-self.y_mean)/self.y_std
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
            y_scale = Y

        # compute the kernel matrix with white noise
        Kxx = self.kernel(X, X, self.hyp) 
        if self.white_noise:
           # k(x_i, x_j) = noise\_level \text{ if } x_i == x_j \text{ else } 0
           Kxx += (self.hyp[2]**2)*jnp.eye(X.shape[0])
        # compute the cholesky decomposition
        self.Kchol = linalg.cholesky(Kxx, lower=True)
        # compute the alpha vector
        # self.alpha = linalg.cho_solve((self.Kchol.T, False),linalg.cho_solve((self.Kchol, True), Y))
        self.alpha = linalg.cho_solve((self.Kchol, True), y_scale)
        # store the X
        self.X = X

    ############################################################################
    ############################################################################
 
    def covariance_matrix(self,Xp):
        # calculate the kernels
        Kxp = self.kernel(Xp, self.X, self.hyp)
        # otherwise cholesky issues when Xp = self.X (need to be different for actual GP)
        if self.white_noise:
           # k(x_i, x_j) = noise\_level \text{ if } x_i == x_j \text{ else } 0
           Kxp += (self.hyp[2]**2)*jnp.eye(Xp.shape[0]) 
        return Kxp

    ############################################################################
    ############################################################################
 
    def predict(self,Xp):
        # calculate the kernels
        Kxp = self.kernel(Xp, self.X, self.hyp)
        # calculate the mean and covariance
        y_mean = Kxp @ self.alpha
        # rescale mean
        y_mean = y_mean*self.y_std + self.y_mean
        return y_mean
    
    ############################################################################
    ############################################################################
 
    def predict_grad(self,Xp):
        jac_mat = jax.jacfwd(self.predict)(Xp)
        # much easier than....
        # grad_logKeff_m = jax.vmap(jax.grad(gp_mod.predict, argnums=0),(1),0)(jXb[jnp.newaxis,:,0:1])
        return jac_mat.diagonal()

    ############################################################################
    ############################################################################
 
    def predict_cov(self,Xp):
        # calculate the kernels
        Kxp = self.kernel(Xp, self.X, self.hyp)
        Kpp = self.kernel(Xp, Xp, self.hyp)
        if self.white_noise:
           # k(x_i, x_j) = noise\_level \text{ if } x_i == x_j \text{ else } 0
           Kpp += (self.hyp[2]**2)*jnp.eye(Xp.shape[0])
        # calculate the covariance (L\Kxp.T)
        v = linalg.solve_triangular(self.Kchol, Kxp.T,lower=True)
        # kpp - v.T . v
        y_cov = Kpp - v.T @ v
        # rescale covariance
        y_cov = jnp.outer(y_cov,self.y_std**2).reshape(*y_cov.shape,-1).squeeze(axis=2)
        # Lreg = 1e-8
        # y_cov += Lreg*jnp.eye(y_cov.shape[0])*jnp.mean(y_cov)
        return y_cov
         
    ############################################################################
    ############################################################################
 