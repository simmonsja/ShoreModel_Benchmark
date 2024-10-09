import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import jax.numpy as jnp

################################################################################
################################################################################

def transform_data_to_jax(data, x_vars, y_vars, transect_var, standardise=True, scaler_x=None, scaler_y=None, interp_y=True):
    # ensure data are sorted by transect then date
    data = data.sort_values([transect_var, 'date'])
    # Transform data to jax format
    if standardise:
        if scaler_x is None:
            scaler_x = StandardScaler()
            scaler_x.fit(data[x_vars])
        if scaler_y is None:
            scaler_y = StandardScaler()
            scaler_y.fit(data[y_vars])
        standardised_data = data.copy()
        standardised_data[x_vars] = scaler_x.transform(data[x_vars])
        if 'month' in x_vars:
            standardised_data.loc[:,'month'] = data.loc[:,'month'].values
        standardised_data[y_vars] = scaler_y.transform(data[y_vars])
    else:
        standardised_data = data.copy()
    # standardised_data = tabular_data['df_obs'].copy()
    # bend it into a 3D frame
    print('Unique transects: {}'.format(standardised_data[transect_var].unique()))
    jnp_X = jnp.array(np.stack(
        [
            standardised_data.query('Transect == @trans_id')[x_vars].values for trans_id in standardised_data[transect_var].unique()
        ], axis=1
    ))

    df_Y = pd.pivot_table(standardised_data, index='date', columns='Transect', values=y_vars[0], dropna=False)

    jnp_Y = np.stack(
        [
            standardised_data.query('Transect == @trans_id')[y_vars].values for trans_id in standardised_data[transect_var].unique()
        ],
        axis=1
    ).squeeze()

    # # fill nan with mean value
    # jnp_Y = np.where(np.isnan(jnp_Y), np.nanmean(jnp_Y), jnp_Y)

    if interp_y:
        # fill nan with linear interpolated value
        for i in range(jnp_Y.shape[1]):
            jnp_Y[:,i] = np.where(np.isnan(jnp_Y[:,i]), np.interp(np.arange(jnp_Y.shape[0]), np.arange(jnp_Y.shape[0])[~np.isnan(jnp_Y[:,i])], jnp_Y[:,i][~np.isnan(jnp_Y[:,i])]), jnp_Y[:,i])

    jnp_Y = jnp.array(jnp_Y)
    jnp_Ym1 = jnp_Y[:-1,:]

    # store jnp_T as an int by category
    jnp_T = jnp.array(standardised_data[transect_var].astype('category').cat.codes.values)

    jnp_X = jnp_X[1:,:]
    jnp_Y = jnp_Y[1:,:]
    df_Y = df_Y.iloc[1:,:]

    print('jnp_X.shape: {}, isnan: {}'.format(jnp_X.shape, np.isnan(jnp_X).sum()))
    print('jnp_Y.shape: {}, isnan: {}'.format(jnp_Y.shape, np.isnan(jnp_Y).sum()))
    print('jnp_T.shape: {}, isnan: {}'.format(jnp_T.shape, np.isnan(jnp_T).sum()))
    print('jnp_Ym1.shape: {}, isnan: {}'.format(jnp_Ym1.shape, np.isnan(jnp_Ym1).sum()))
    return jnp_X, jnp_Y, jnp_Ym1, df_Y, {'scaler_x':scaler_x, 'scaler_y':scaler_y}

################################################################################
################################################################################

def prepare_resample_monthly_data(data):
    # split per Transect then average to monthly values with mean and peak for Hs and Tp then recombine into a reasonable dataframe
    resampled_data = data.copy()
    resampled_data = pd.concat(
        [
            resampled_data.query('Transect == @trans_id').drop(columns=['Transect']).set_index('date').resample('MS').agg({'Hs':['mean','max'],'Tp':['mean','max'],'Dir':['mean'],'shoreline':['mean']}).reset_index().assign(Transect=trans_id) for trans_id in resampled_data['Transect'].unique()
        ], axis=0
    )
    # combine the column names to make one level
    resampled_data.columns = [
        '_'.join(col).strip() if '' != col[1] else col[0] for col in resampled_data.columns.values]
    resampled_data = resampled_data.rename(columns={'shoreline_mean':'shoreline'})
    # now add month predictor
    resampled_data = resampled_data.assign(month=resampled_data['date'].dt.month-1)
    resampled_data = resampled_data.reset_index(drop=True)

    # log Hs_mean and Hs_max
    resampled_data['Hs_mean'] = np.log(resampled_data['Hs_mean'])
    resampled_data['Hs_max'] = np.log(resampled_data['Hs_max'])

    # now interpolate the missing values per transect
    # resampled_data.loc[:,'shoreline'] = resampled_data.groupby('Transect')['shoreline'].apply(lambda x: x.interpolate(method='linear')).values

    return resampled_data

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