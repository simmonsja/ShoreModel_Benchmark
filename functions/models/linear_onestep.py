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

def build_base_linear_model(n_variables, yscale_mean, yscale_scale, addscale_mean, addscale_scale):
    def time_scan(tcarry,tparams):
        # use actual or predicted shoreline depending on test
        orig_shl = jax.lax.cond(tcarry['test'],lambda car,par: car['shl_prev'],lambda car,par: par['add'], tcarry, tparams)

        # calc dval
        dval = orig_shl * tcarry['beta_ar1'] + jnp.sum(tcarry['beta'] * tparams['x']) + tcarry['beta_0']

        
        # jax.debug.print('orig shl: {}',orig_shl)
        # jax.debug.print('orig shl T: {}',(orig_shl * addscale_scale) + addscale_mean)
        # jax.debug.print('dval: {}',dval)
        # jax.debug.print('dval T: {}',(dval * yscale_scale) + yscale_mean)
        
        # unscale all 
        new_shl = ((orig_shl * addscale_scale) + addscale_mean) + (dval.squeeze() * yscale_scale + yscale_mean)
        tcarry['shl_prev'] = (new_shl - addscale_mean) / addscale_scale

        # jax.debug.print('new_shl: {}',new_shl)
        # jax.debug.print('new_shl T: {}',tcarry['shl_prev'])

        # return dShoreline or Shoreline pending test
        dval = jax.lax.cond(tcarry['test'],lambda dv,car: car['shl_prev'],lambda dv,car: dv, dval, tcarry)

        return tcarry, dval

    def base_linear_model(X, add, Y=None, test=False):
        '''
        Define linear hierarchical model with priors for the parameters and model error
        Inputs:
            energy: storm energy
            dshl: observed shoreline change
        '''
        # Counts
        vars_num = X.shape[1]

        # Hyperpriors for our multi-level model
        # sample the beta params
        tau = 2.0
        # AR1 term
        beta_ar1 = numpyro.sample("beta_ar1", dist.Normal(0, tau))
        # beta terms
        with numpyro.plate("features", vars_num,dim=-1):
            beta = numpyro.sample("beta", dist.Normal(0, tau))
        # intercept
        beta_0 = numpyro.sample("beta_0", dist.Normal(0, tau))

        sigma = numpyro.sample("sigma", dist.Exponential(1))

        carry = {
            'beta_ar1': beta_ar1,
            'beta': beta,
            'beta_0': beta_0,
            'shl_prev': add[0],
            'test': test
        }
        scan_covariates = {
            'x': X,
            'add': add
        }

        _, mu = jax.lax.scan(
            time_scan,
            carry,
            scan_covariates
        )

        numpyro.deterministic("mu",mu)

        numpyro.sample("obs", dist.Normal(mu, sigma), obs=Y)

    return base_linear_model

################################################################################
################################################################################

def build_shladj_linear_model():
    def time_scan(tcarry,tparams):
        # use actual or predicted shoreline depending on test
        orig_shl = tcarry['shl_prev']

        Hsig_beta = tcarry['beta_Hsig_0'] + tparams['Hsig_max'] * tcarry['beta_Hsig_max'] + tparams['Dir_mean'] * tcarry['beta_Hsig_dir']

        # calc dval
        dval = Hsig_beta * tparams['Hsig_mean'] + tcarry['beta_Tp_0'] * tparams['Tp_mean'] + tcarry['beta_0'] + jnp.take_along_axis(tcarry['beta_month'],tparams['month'][None,:],axis=0).squeeze()

        # carry shl
        tcarry['shl_prev'] = orig_shl + dval

        return tcarry, orig_shl + dval

    def adj_linear_model(X, add, Y=None, test=False):
        '''
        Define linear hierarchical model with priors for the parameters and model error
        Inputs:
            energy: storm energy
            dshl: observed shoreline change
        '''
        # Counts
        vars_num = X.shape[1]
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

        tau_Hsig_0 = numpyro.sample("tau_Hsig_0", dist.Exponential(1))
        tau_Hsig_max = numpyro.sample("tau_Hsig_max", dist.Exponential(1))
        tau_Hsig_dir = numpyro.sample("tau_Hsig_dir", dist.Exponential(1))
        tau_Tp_0 = numpyro.sample("tau_Tp_0", dist.Exponential(1))
        tau_0 = numpyro.sample("tau_0", dist.Exponential(1))
        tau_month = numpyro.sample("tau_month", dist.Exponential(1))


        with numpyro.plate("transects", tran_num, dim=-1):
            beta_Hsig_0 = numpyro.sample("beta_Hsig_0", dist.Normal(alpha_Hsig_0, tau_Hsig_0))
            beta_Hsig_max = numpyro.sample("beta_Hsig_max", dist.Normal(alpha_Hsig_max, tau_Hsig_max))
            beta_Hsig_dir = numpyro.sample("beta_Hsig_dir", dist.Normal(alpha_Hsig_dir, tau_Hsig_dir))
            beta_Tp_0 = numpyro.sample("beta_Tp_0", dist.Normal(alpha_Tp_0, tau_Tp_0))
            beta_0 = numpyro.sample("beta_0", dist.Normal(alpha_0, tau_0))
            beta_month = numpyro.sample("beta_month", dist.Normal(alpha_month, tau_month))    

        sigma = numpyro.sample("sigma_meas", dist.Exponential(1))

        carry = {
            'beta_Hsig_0': beta_Hsig_0,
            'beta_Hsig_max': beta_Hsig_max,
            'beta_Hsig_dir': beta_Hsig_dir,
            'beta_Tp_0': beta_Tp_0,
            'beta_0': beta_0,
            'beta_month': beta_month,
            'shl_prev': add
        }
        scan_covariates = {
            'Hsig_mean': Hsig_mean,
            'Hsig_max': Hsig_max,
            'Tp_mean': Tp_mean,
            'Tp_max': Tp_max,
            'Dir_mean': Dir_mean,
            'month': month
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

################################################################################
################################################################################

def build_seasonal_linear_model():
    def time_scan(tcarry,tparams):
        # use actual or predicted shoreline depending on test
        orig_shl = tcarry['shl_prev']

        # calc dval
        dval = tcarry['beta_0'] + jnp.take_along_axis(tcarry['beta_month'],tparams['month'][None,:],axis=0).squeeze()

        # carry shl
        tcarry['shl_prev'] = orig_shl + dval

        return tcarry, orig_shl + dval

    def adj_linear_model(X, add, Y=None, test=False):
        '''
        Define linear hierarchical model with priors for the parameters and model error
        Inputs:
            energy: storm energy
            dshl: observed shoreline change
        '''
        # Counts
        vars_num = X.shape[1]
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

        tau_Hsig_0 = numpyro.sample("tau_Hsig_0", dist.Exponential(1))
        tau_Hsig_max = numpyro.sample("tau_Hsig_max", dist.Exponential(1))
        tau_Hsig_dir = numpyro.sample("tau_Hsig_dir", dist.Exponential(1))
        tau_Tp_0 = numpyro.sample("tau_Tp_0", dist.Exponential(1))
        tau_0 = numpyro.sample("tau_0", dist.Exponential(1))
        tau_month = numpyro.sample("tau_month", dist.Exponential(1))


        with numpyro.plate("transects", tran_num, dim=-1):
            beta_Hsig_0 = numpyro.sample("beta_Hsig_0", dist.Normal(alpha_Hsig_0, tau_Hsig_0))
            beta_Hsig_max = numpyro.sample("beta_Hsig_max", dist.Normal(alpha_Hsig_max, tau_Hsig_max))
            beta_Hsig_dir = numpyro.sample("beta_Hsig_dir", dist.Normal(alpha_Hsig_dir, tau_Hsig_dir))
            beta_Tp_0 = numpyro.sample("beta_Tp_0", dist.Normal(alpha_Tp_0, tau_Tp_0))
            beta_0 = numpyro.sample("beta_0", dist.Normal(alpha_0, tau_0))
            beta_month = numpyro.sample("beta_month", dist.Normal(alpha_month, tau_month))    

        sigma = numpyro.sample("sigma_meas", dist.Exponential(1))

        carry = {
            'beta_Hsig_0': beta_Hsig_0,
            'beta_Hsig_max': beta_Hsig_max,
            'beta_Hsig_dir': beta_Hsig_dir,
            'beta_Tp_0': beta_Tp_0,
            'beta_0': beta_0,
            'beta_month': beta_month,
            'shl_prev': add
        }
        scan_covariates = {
            'Hsig_mean': Hsig_mean,
            'Hsig_max': Hsig_max,
            'Tp_mean': Tp_mean,
            'Tp_max': Tp_max,
            'Dir_mean': Dir_mean,
            'month': month
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


################################################################################
################################################################################
