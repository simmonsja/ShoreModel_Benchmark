# NumPyro for proabilistic programming
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

################################################################################
################################################################################
# Models
################################################################################
################################################################################

def build_shladj_linear_model():
    def time_scan(tcarry,tparams):
        # use actual or predicted shoreline depending on test
        orig_shl = tcarry['shl_prev']

        static_shl = jax.lax.cond(tcarry['test'],lambda car,par: car['shl_prev'],lambda car,par: par['shl_prev_static'], tcarry, tparams)

        Hsig_beta = tcarry['beta_Hsig_0'] + static_shl * tcarry['beta_shl'] + tparams['Hsig_max'] * tcarry['beta_Hsig_max'] + tcarry['beta_Tp_0'] * tparams['Tp_mean'] + tparams['Dir_mean'] * tcarry['beta_Hsig_dir']

        # calc dval
        dval = Hsig_beta * tparams['Hsig_mean'] + jnp.take_along_axis(tcarry['beta_month'],tparams['month'][None,:],axis=0).squeeze()
        #  + tcarry['beta_0'] don't need as unidentifiable with beta_month

        # carry shl
        tcarry['shl_prev'] = orig_shl * tcarry['beta_ar1'] + dval

        return tcarry, orig_shl * tcarry['beta_ar1'] + dval

    def adj_linear_model(X, add, Y=None, test=False):
        '''
        Define linear hierarchical model with priors for the parameters and model error
        Inputs:
            energy: storm energy
            dshl: observed shoreline change
        '''
        # Counts
        tran_num = X.shape[1]
        n_months = 12

        Hsig_mean = X[:,:,0]
        Hsig_max = X[:,:,1]
        Tp_mean = X[:,:,2]
        Tp_max = X[:,:,3]
        Dir_mean = X[:,:,4]
        month = X[:,:,5].astype(int)

        # Hyperpriors for our multi-level model
        # sample the beta params
        tau = 2.0
        # AR1 term
        alpha_Hsig_0 = numpyro.sample("alpha_Hsig_0", dist.Normal(0, tau))
        alpha_Hsig_max = numpyro.sample("alpha_Hsig_max", dist.Normal(0, tau))
        alpha_Hsig_dir = numpyro.sample("alpha_Hsig_dir", dist.Normal(0, tau))
        alpha_Tp_0 = numpyro.sample("alpha_Tp_0", dist.Normal(0, tau))
        alpha_0 = numpyro.sample("alpha_0", dist.Normal(0, tau))
        with numpyro.plate("months", n_months, dim=-2):
            alpha_month = numpyro.sample("alpha_month", dist.Normal(0, tau))
        alpha_shl = numpyro.sample("alpha_shl", dist.Normal(0, tau))

        tau_Hsig_0 = numpyro.sample("tau_Hsig_0", dist.Exponential(1))
        tau_Hsig_max = numpyro.sample("tau_Hsig_max", dist.Exponential(1))
        tau_Hsig_dir = numpyro.sample("tau_Hsig_dir", dist.Exponential(1))
        tau_Tp_0 = numpyro.sample("tau_Tp_0", dist.Exponential(1))
        tau_0 = numpyro.sample("tau_0", dist.Exponential(1))
        tau_month = numpyro.sample("tau_month", dist.Exponential(1))
        tau_shl = numpyro.sample("tau_shl", dist.Exponential(1))

        with numpyro.plate("transects", tran_num, dim=-1):
            beta_Hsig_0 = numpyro.sample("beta_Hsig_0", dist.Normal(alpha_Hsig_0, tau_Hsig_0))
            beta_Hsig_max = numpyro.sample("beta_Hsig_max", dist.Normal(alpha_Hsig_max, tau_Hsig_max))
            beta_Hsig_dir = numpyro.sample("beta_Hsig_dir", dist.Normal(alpha_Hsig_dir, tau_Hsig_dir))
            beta_Tp_0 = numpyro.sample("beta_Tp_0", dist.Normal(alpha_Tp_0, tau_Tp_0))
            beta_0 = numpyro.sample("beta_0", dist.Normal(alpha_0, tau_0))
            beta_month = numpyro.sample("beta_month", dist.Normal(alpha_month, tau_month))
            beta_ar1 = numpyro.sample("beta_ar1", dist.Uniform(0,1))
            beta_shl = numpyro.sample("beta_shl", dist.Normal(alpha_shl, tau_shl))

        sigma = numpyro.sample("sigma_meas", dist.Exponential(1))

        carry = {
            'beta_Hsig_0': beta_Hsig_0,
            'beta_Hsig_max': beta_Hsig_max,
            'beta_Hsig_dir': beta_Hsig_dir,
            'beta_Tp_0': beta_Tp_0,
            'beta_0': beta_0,
            'beta_month': beta_month,
            'beta_ar1': beta_ar1,
            'beta_shl': beta_shl,
            'shl_prev': add[0,:],
            'test': test
        }
        scan_covariates = {
            'Hsig_mean': Hsig_mean,
            'Hsig_max': Hsig_max,
            'Tp_mean': Tp_mean,
            'Tp_max': Tp_max,
            'Dir_mean': Dir_mean,
            'month': month,
            'shl_prev_static': add
        }

        _, mu = jax.lax.scan(
            time_scan,
            carry,
            scan_covariates
        )

        numpyro.deterministic("mu",mu)

        numpyro.sample("obs", dist.Normal(mu, sigma), obs=Y)

    return adj_linear_model

################################################################################
################################################################################