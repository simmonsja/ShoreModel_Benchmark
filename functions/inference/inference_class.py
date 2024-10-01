import pickle

# NumPyro for proabilistic programming
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.diagnostics import hpdi
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value, init_to_sample, init_to_feasible
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

import arviz as az

############################################################
############################################################

# numpyro model class
class NumPyroSampler():
    '''
    A class that represents a Bayesian Neural Network using NumPyro and Flax.

    Attributes:
        width (int): The number of neurons in each layer.
        depth (int): The number of layers in the neural network.
        activation_fn (object): The activation function to use in each layer.
        flax_module (object): The Flax module representing the neural network.
        model (object): The NumPyro model.
        samples (object): The samples from the posterior distribution.
    '''

    ############################################################
    ############################################################
 
    # init with width, depth and activation functin holders
    def __init__(self, X, y, add, model=None, ci=0.89, seed=1991):
        '''
        Initializes the class for inference

        Args:
            width (int): The number of neurons in each layer.
            depth (int): The number of layers in the neural network.
            activation_fn (object): The activation function to use in each layer.
        '''
        self.seed = seed
        self.input_dim = 1
        self.output_dim = 1
        self.ci = ci
        self.model = model
        self.arviz = None
        self.guide = None
        self.samples = None

        # store the arrays
        self.train_X = X
        self.train_Y = y
        self.train_add = add       

    ############################################################
    ############################################################
    
    def prior_predictive_check(self, num_samples=100, test=False):
        # JAX requires a key for random number generation
        rng_key_ = random.PRNGKey(self.seed)
        # here take 100 samples from our priors and make predictions on x_log
        # linear_hmodel or bnn_model
        prior_samples = Predictive(self.model, num_samples=num_samples)(
            rng_key_, X=self.train_X, add=self.train_add, Y=None, test=test
        )
        prior_samples = {k: jnp.expand_dims(v,axis=0) for k, v in prior_samples.items()}
        # and put this into arviz for easy plotting
        arviz_priors = az.from_dict(
            prior=prior_samples
        )

        # get the mean model prediciton and CI
        mean_mu_prior = jnp.mean(arviz_priors.prior['mu'].values.squeeze(), axis=0)
        hpdi_mu_prior = hpdi(arviz_priors.prior['mu'].values.squeeze(), self.ci)
        hpdi_sim_prior = hpdi(arviz_priors.prior['obs'].values.squeeze(), self.ci)

        return (arviz_priors, {'mean_mu_prior': mean_mu_prior, 'hpdi_mu_prior': hpdi_mu_prior, 'hpdi_sim_prior': hpdi_sim_prior})

    ############################################################
    ############################################################

    def model_predict(self, rng_key, X, class_var=None, num_samples=1000, prior=True):
        '''
        Predicts the output for the given input data.

        Args:
            rng_key (object): The random number generator key.
            X (jnp.array): The input data.
            num_samples (int): The number of samples to draw from the posterior.
            prior (bool): Whether to use the prior or the posterior for prediction.

        Returns:
            jnp.array: The predicted output.
        '''

        rng, _ = jax.random.split(rng_key)
        if prior:
            samples = Predictive(
                self.model, num_samples=num_samples, return_sites=['mu']
            )(
                rng, X=jnp.array(X), Y=None
            )
        else:
            samples = Predictive(
                self.model, num_samples=num_samples, return_sites=['mu'],
                posterior_samples=self.samples
            )(
                rng, X=jnp.array(X), Y=None,
            ) 
        return samples['mu'].squeeze().T
    
    ############################################################
    ############################################################
        
    def run_sampler(self,rng_key,num_samples,num_warmup,num_chains,max_tree,extract_vars=None):
        '''
        Trains the model with the given input and output data.

        Args:
            rng_key (object): The random number generator key.
            X (jnp.array): The input data.
            Y (jnp.array): The output data.
            num_samples (int): The number of samples to draw from the posterior.
            num_warmup (int): The number of warmup steps before sampling.
            num_chains (int): The number of Markov chains to run in parallel.
            max_tree (int): The maximum tree depth for the NUTS sampler.
        '''
        nuts = NUTS(
            self.model,
            max_tree_depth=max_tree
        )
        mcmc_obj = MCMC(
            nuts, 
            num_warmup=num_warmup, 
            num_samples=num_samples,
            num_chains=num_chains
        )
        mcmc_obj.run(
            rng_key,
            X = self.train_X,
            add = self.train_add,
            Y = self.train_Y
        )

        self.mcmc_obj = mcmc_obj
        # get the samples which will form our posterior
        samples = mcmc_obj.get_samples()

        # get the samples for predictive uncertainty (our linear model + error)
        posterior_predictive = Predictive(
            self.model, posterior_samples=samples, 
            return_sites=extract_vars
        )(
            rng_key, 
            X = self.train_X, add = self.train_add, Y = None
        )

        self.arviz = az.from_numpyro(
            mcmc_obj,
            posterior_predictive = posterior_predictive
        )

        mean_mu_train, hpdi_mu_train, hpdi_sim_train = calc_mean_hpdi(
            self.arviz, ci=self.ci, y_scaler=None, mu_var='mu', sim_var='obs'
        )

        # posterior_predictive = Predictive(
        #     self.model, posterior_samples=samples, 
        #     return_sites=extract_vars
        # )(
        #     rng_key, 
        #     X = self.test_X, add = self.test_add, Y = None
        # )

        # predictive_arviz = az.from_dict(
        #     posterior_predictive={k: jnp.expand_dims(v, 0) for k, v in posterior_predictive.items()}
        # )

        # mean_mu_test, hpdi_mu_test, hpdi_sim_test = calc_mean_hpdi(
        #     predictive_arviz, ci=self.ci, y_scaler=None, mu_var='mu', sim_var='obs'
        # )
    
        self.samples = samples

        results = {
            'mean_mu_train': mean_mu_train,
            'hpdi_mu_train': hpdi_mu_train,
            'hpdi_sim_train': hpdi_sim_train,
            # 'mean_mu_test': mean_mu_test,
            # 'hpdi_mu_test': hpdi_mu_test,
            # 'hpdi_sim_test': hpdi_sim_test
        }
        self.results = results

    ############################################################
    ############################################################
        
    def run_sampler_SVI(self,rng_key,num_samples,num_chains,step_size=0.01,extract_vars=None):
        '''
        Trains the model with the given input and output data.

        Args:
            rng_key (object): The random number generator key.
            X (jnp.array): The input data.
            Y (jnp.array): The output data.
            num_samples (int): The number of samples to draw from the posterior.
            num_warmup (int): The number of warmup steps before sampling.
            num_chains (int): The number of Markov chains to run in parallel.
            max_tree (int): The maximum tree depth for the NUTS sampler.
        '''
        rng, _ = jax.random.split(rng_key)
        elbo = Trace_ELBO(num_particles=num_chains)
        step_size = step_size
        optimizer = numpyro.optim.ClippedAdam(step_size=step_size, clip_norm=10.0)
        #Clipped,clip_norm=10.0)

        guide = AutoNormal(self.model)
        svi = SVI(
            model=self.model, guide=guide, optim=optimizer, loss=elbo,  
            X = self.train_X, add = self.train_add, Y = self.train_Y
        )

        svi_result = svi.run(rng_key, num_samples)

        self.params = svi_result.params
        self.losses = svi_result.losses
        self.guide = guide

        # SVI get samples using the guide
        predictive = Predictive(
            guide,
            params=self.params,
            num_samples=num_samples
        )
        samples = predictive(rng,
            X = self.train_X, add = self.train_add, Y = None
        )
        posterior_predictive = Predictive(
            self.model, posterior_samples=samples, 
            return_sites=extract_vars
        )(
            rng, 
            X = self.train_X, add = self.train_add, Y = None
        )

        self.samples = samples

        self.arviz = az.from_dict(
            {k: jnp.expand_dims(v, 0) for k, v in samples.items()},
            posterior_predictive={k: jnp.expand_dims(v, 0) for k, v in posterior_predictive.items()},
        )

        mean_mu_train, hpdi_mu_train, hpdi_sim_train = calc_mean_hpdi(
            self.arviz, ci=self.ci, y_scaler=None, mu_var='mu', sim_var='obs'
        )
    
        results = {
            'mean_mu_train': mean_mu_train,
            'hpdi_mu_train': hpdi_mu_train,
            'hpdi_sim_train': hpdi_sim_train,
            # 'mean_mu_test': mean_mu_test,
            # 'hpdi_mu_test': hpdi_mu_test,
            # 'hpdi_sim_test': hpdi_sim_test
        }
        self.results = results


    ############################################################
    ############################################################
    
    def predict_forward(self, rng_key, extract_vars=None):
        '''
        Predicts the output for the given input data - training and test data.
        '''

        posterior_predictive = Predictive(
            self.model, posterior_samples=self.samples, 
            return_sites=extract_vars
        )(
            rng_key, 
            X = self.train_X, add = self.train_add, Y = None, test = True
        )
        predictive_arviz = az.from_dict(
            posterior_predictive={k: jnp.expand_dims(v, 0) for k, v in posterior_predictive.items()}
        )

        mean_mu_train, hpdi_mu_train, hpdi_sim_train = calc_mean_hpdi(
            predictive_arviz, ci=self.ci, y_scaler=self.yScaler, mu_var='mu', sim_var='obs'
        )

        posterior_predictive = Predictive(
            self.model, posterior_samples=self.samples, 
            return_sites=extract_vars
        )(
            rng_key, 
            X = self.test_X, add = self.test_add, Y = None, test = True
        )
        predictive_arviz = az.from_dict(
            posterior_predictive={k: jnp.expand_dims(v, 0) for k, v in posterior_predictive.items()}
        )

        mean_mu_test, hpdi_mu_test, hpdi_sim_test = calc_mean_hpdi(
            predictive_arviz, ci=self.ci, y_scaler=self.yScaler, mu_var='mu', sim_var='obs'
        )

        results = {
            'mean_mu': mean_mu_train,
            'hpdi_mu': hpdi_mu_train,
            'hpdi_sim': hpdi_sim_train,
            'mean_mu_test': mean_mu_test,
            'hpdi_mu_test': hpdi_mu_test,
            'hpdi_sim_test': hpdi_sim_test
        }
        return results

    ############################################################
    ############################################################    

def calc_mean_hpdi(arviz_post, ci=0.89, y_scaler=None, mu_var='mu', sim_var='obs'):
    """
    Calculate the mean and highest posterior density interval (HPDI) for the 'mu' and 'obs' variables in the ArviZ posterior
    and posterior predictive objects.

    Parameters
    ----------
    arviz_post : ArviZ InferenceData object
        The posterior and posterior predictive samples.
    ci : float, optional
        The probability mass of the HPDI. Default is 0.89.
    y_scaler : Scaler object, optional
        A Scaler object to unscale the mean and HPDI values if the data was scaled before fitting the model. Default is None.
    mu_var : str, optional
        The name of the 'mu' variable in the ArviZ posterior object. Default is 'mu'.
    sim_var : str, optional
        The name of the 'obs' variable in the ArviZ posterior predictive object. Default is 'obs'.

    Returns
    -------
    mean_mu : ndarray
        The mean of the 'mu' variable.
    hpdi_mu : ndarray
        The HPDI of the 'mu' variable.
    hpdi_sim : ndarray
        The HPDI of the 'obs' variable.
    """
    # Define the dimensions that are common to both the posterior and posterior predictive objects
    base_dims = ['chain','draw']
    
    # Get the mean of the 'mu' variable in the posterior object
    mean_mu = arviz_post.posterior_predictive[mu_var].mean(dim=base_dims).values
    
    # Define the dimensions for the 'mu' and 'obs' variables that are not in the base dimensions
    mu_dims = [_ for _ in arviz_post.posterior_predictive[mu_var].coords.dims if not _ in base_dims]
    sim_dims = [_ for _ in arviz_post.posterior_predictive[sim_var].coords.dims if not _ in base_dims]
    
    # Calculate the HPDI for the 'mu' and 'obs' variables using the arviz.hdi() function
    hpdi_mu = az.hdi(
        arviz_post.posterior_predictive, hdi_prob=ci, var_names=[mu_var]
    ).transpose('hdi',*mu_dims)[mu_var].values
    hpdi_sim = az.hdi(
        arviz_post.posterior_predictive, hdi_prob=ci, var_names=[sim_var]
    ).transpose('hdi',*sim_dims)[sim_var].values

    # If a scaler object is provided, unscale the mean and HPDI values and reverse the log transform if necessary
    if not y_scaler is None:
        # Unscale the mean and HPDI values
        mean_mu = y_scaler.inverse_transform(mean_mu.reshape(-1,1)).squeeze()
        hpdi_mu[0,...] = y_scaler.inverse_transform(hpdi_mu[0,...].reshape(-1,1)).squeeze()
        hpdi_mu[1,...] = y_scaler.inverse_transform(hpdi_mu[1,...].reshape(-1,1)).squeeze()
        hpdi_sim[0,...] = y_scaler.inverse_transform(hpdi_sim[0,...].reshape(-1,1)).squeeze()
        hpdi_sim[1,...] = y_scaler.inverse_transform(hpdi_sim[1,...].reshape(-1,1)).squeeze()
    
    # Return the mean and HPDI values for the 'mu' and 'obs' variables
    return mean_mu, hpdi_mu, hpdi_sim
        
################################################################################
################################################################################
