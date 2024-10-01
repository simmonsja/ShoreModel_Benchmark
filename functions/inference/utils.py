import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

################################################################################
################################################################################

def calc_skill(curr_obs,prev_obs,mean_mu):
    good_idx = np.where(~np.isnan(curr_obs) & ~np.isnan(prev_obs))[0]
    if np.any(np.isnan(mean_mu)) or np.any(np.abs(mean_mu)>1e20):
        print('Mean mu has NaNs')
        return np.nan, np.nan
    # calc RMSE
    rmse = np.sqrt(mean_squared_error(curr_obs.iloc[good_idx],mean_mu[good_idx]))
    # calc R
    dx = curr_obs.values[1:] - curr_obs.values[:-1] 
    dx_pred = mean_mu[1:] - mean_mu[:-1]
    r2 = r2_score(dx[good_idx[1:]-1],dx_pred[good_idx[1:]-1])
    r = np.corrcoef(curr_obs.iloc[good_idx],mean_mu[good_idx])[0,1]
    # calc BSS vs persistence
    bss = 1 - (np.sum((curr_obs.iloc[good_idx] - mean_mu[good_idx])**2)/np.sum((curr_obs.iloc[good_idx] - prev_obs.iloc[good_idx])**2))
    return rmse, r2, bss, r

################################################################################

def print_skill(models,df, mu_var='mean_mu'):
    # Calculate BSS, RMSE and R and print each for the models in models
    model_avg_bss = {_:[] for _ in models.keys()}
    model_avg_rmse = {_:[] for _ in models.keys()}
    model_avg_r2 = {_:[] for _ in models.keys()}
    model_avg_r = {_:[] for _ in models.keys()}
    for ii, this_tran in enumerate(df.columns):
        print('# Site {}: {}'.format(ii, this_tran))
        for this_mod in models.keys():
            rmse, r2, bss, r = calc_skill(df[this_tran],df[this_tran].shift(1),models[this_mod][mu_var][:,df.columns.get_loc(this_tran)])
            print('{} - {}: BSS: {:.2f} | RMSE: {:.2f} | R2: {:.2f} | r: {:.2f}'.format(this_mod,this_tran,bss,rmse,r2,r))
            model_avg_bss[this_mod].append(bss)
            model_avg_rmse[this_mod].append(rmse)
            model_avg_r2[this_mod].append(r2) 
            model_avg_r[this_mod].append(r)
    print('# Overall')
    for this_mod in models.keys():
        print('Model - {}: BSS: {:.2f} | RMSE: {:.2f} | R2: {:.2f} | r: {:.2f}'.format(this_mod,np.nanmean(model_avg_bss[this_mod]),np.nanmean(model_avg_rmse[this_mod]),np.nanmean(model_avg_r2[this_mod]), np.nanmean(model_avg_r[this_mod])))
    return model_avg_bss

################################################################################
################################################################################