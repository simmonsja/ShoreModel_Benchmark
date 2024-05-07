import os
import pandas as pd

################################################################################
################################################################################
##############################   MAIN FUNCTIONS   ##############################
################################################################################
################################################################################

def load_modelling_data(basedir='.'):
    '''
    Handy function mirroring the load in 1.data_explore.ipynb that converts 
    all the painfull data loading into a one line call. Hard coded paths and file names to mirror the shoreshop main repo structure.
    
    Inputs:
    - basedir: str, path to the directory where the datasets are stored

    Returns:
    dict: A dictionary containing the following pandas DataFrames:
        - 'df_tran': Transect data.
        - 'df_gt': Groundtruth shoreline positions.
        - 'df_targ_short': Short-term target shoreline positions.
        - 'df_targ_medium': Medium-term target shoreline positions.
        - 'df_obs': Observed shoreline positions.
        - 'dfs_wave': Wave data for different parameters ('Hs', 'Tp', 'Dir').
        - 'df_SLR_obs': Observed sea level rise data.
        - 'df_SLR_proj': Projected sea level rise data.
    '''
    # Set inputs
    # Transect info
    fp = 'datasets' #File path
    fn_tran =  'transects_coords.csv' #File name for transects
    target_trans = ['Transect2', 'Transect5', 'Transect8'] # Target transects for evaluation

    # Shoreline data
    fn_obs =  'shorelines_obs.csv' # File name for shoreline observation
    fn_targ_short =  'shorelines_target_short.csv' # File name for short-term shoreline prediction target
    fn_targ_medium =  'shorelines_target_medium.csv' # File name for medium-term shoreline prediction target
    fn_gt =  'shorelines_groundtruth.csv' #File name for groudtruth

    # Wave data
    WAVE_PARAMS = ['Hs', 'Tp', 'Dir'] 

    ################################################################################
    # Read transect info
    df_tran = pd.read_csv(os.path.join(basedir,fp, fn_tran), index_col='ID')
    df_tran
    
    print('df_tran: Loaded {} transects...'.format(len(df_tran)))

    ################################################################################
    # Read shoreline data
    df_gt = pd.read_csv(os.path.join(basedir,fp, 'shorelines', fn_gt), index_col='Datetime', parse_dates=True, date_format='%d/%m/%Y')
    # df_gt.index = pd.to_datetime(df_gt.index)
    print('df_gt: Loaded {} (dates,transects) groundtruth shoreline positions...{} to {}'.format(df_gt.shape, df_gt.index.min().strftime('%Y-%m-%d'), df_gt.index.max().strftime('%Y-%m-%d')))

    df_targ_short = pd.read_csv(os.path.join(basedir,fp, 'shorelines', fn_targ_short), index_col='Datetime', parse_dates=True, date_format='%Y-%m-%d %H:%M:%S')
    # df_targ_short.index = pd.to_datetime(df_targ_short.index)
    print('df_targ_short: Loaded {} (dates,transects) short-term target shoreline positions...{} to {}'.format(df_targ_short.shape, df_targ_short.index.min().strftime('%Y-%m-%d'), df_targ_short.index.max().strftime('%Y-%m-%d')))

    # something a bit weird here - med should match the others (or I think the others should be rounded to a day)
    df_targ_medium = pd.read_csv(os.path.join(basedir,fp, 'shorelines', fn_targ_medium), index_col='Datetime', parse_dates=True, date_format='%Y-%m-%d')
    # df_targ_medium.index = pd.to_datetime(df_targ_medium.index)
    print('df_targ_medium: Loaded {} (dates,transects) medium-term target shoreline positions...{} to {}'.format(df_targ_medium.shape, df_targ_medium.index.min().strftime('%Y-%m-%d'), df_targ_medium.index.max().strftime('%Y-%m-%d')))

    df_obs = pd.read_csv(os.path.join(basedir,fp, 'shorelines', fn_obs), index_col='Datetime', parse_dates=True, date_format='%Y-%m-%d %H:%M:%S')
    # df_obs.index = pd.to_datetime(df_obs.index)
    print('df_obs: Loaded {} (dates,transects) observed shoreline positions...{} to {}'.format(df_obs.shape, df_obs.index.min().strftime('%Y-%m-%d'), df_obs.index.max().strftime('%Y-%m-%d')))
    
    ################################################################################
    # Read wave data

    # Hs: Significant wave height
    # Tp: Peak wave period
    # Dir: Mean wave direction

    dfs_wave = {}
    for wave_param in WAVE_PARAMS:
        df_wave = pd.read_csv(
            os.path.join(basedir,fp, 'hindcast_waves' ,'{}.csv'.format(wave_param)),
            index_col = 'Datetime', parse_dates=True, date_format='%Y-%m-%d'
        )
        # df_wave.index = pd.to_datetime(df_wave.index)
        dfs_wave[wave_param] = df_wave
        print('dfs_wave["{}"]: Loaded {} (dates,transects) {} wave data...'.format(wave_param, df_wave.shape, wave_param))
    print('Wave data spans: {} to {} at {} day(s) interval'.format(df_wave.index.min().strftime('%Y-%m-%d'), df_wave.index.max().strftime('%Y-%m-%d'), (df_wave.index[1]-df_wave.index[0]).days))
    
    ################################################################################
    # Read SLR data
    df_SLR_obs = pd.read_csv(
        os.path.join(basedir,fp, 'sealevel', 'SLR_obs.csv'),
        index_col = 'Year', parse_dates=True, date_format='%Y')
    df_SLR_proj = pd.read_csv(
        os.path.join(basedir,fp, 'sealevel', 'SLR_proj.csv'),
        index_col = 'Year', parse_dates=True, date_format='%Y')

    print('df_SLR_obs: Loaded {} (years, {}) observed sea level rise data...{} to {}'.format(df_SLR_obs.shape, df_SLR_obs.columns.tolist(), df_SLR_obs.index.min().strftime('%Y'), df_SLR_obs.index.max().strftime('%Y')))
    print('df_SLR_proj: Loaded {} (years, {}) projected sea level rise data...{} to {}'.format(df_SLR_proj.shape, df_SLR_proj.columns.tolist(), df_SLR_proj.index.min().strftime('%Y'), df_SLR_proj.index.max().strftime('%Y')))

    ################################################################################
    # combine the data for output - nothing fancy 
    # stick to the names already decided on
    combined_data = {
        'df_tran': df_tran,
        'df_gt': df_gt,
        'df_targ_short': df_targ_short,
        'df_targ_medium': df_targ_medium,
        'df_obs': df_obs,
        'dfs_wave': dfs_wave,
        'df_SLR_obs': df_SLR_obs,
        'df_SLR_proj': df_SLR_proj
    }

    ################################################################################ 
    # return data
    return combined_data


################################################################################
################################################################################

def tabularise_raw_data(combined_data):
    # prepare the observed data for modelling
    df_obs = combined_data['df_obs'].copy()
    # rounding to the day for now - limit assigning to the nearest day
    df_obs.index = df_obs.index.round('D')
    df_obs = df_obs.resample('D').mean().reset_index(names='date').melt(id_vars='date',var_name='Transect', value_name='shoreline')

    # prepare the targ_short
    df_targ_short = combined_data['df_targ_short'].copy()
    df_targ_short.index = df_targ_short.index.round('D')
    df_targ_short = df_targ_short.resample('D').mean().reset_index(names='date').melt(id_vars='date',var_name='Transect', value_name='shoreline')

    # prepare the targ_medium
    df_targ_medium = combined_data['df_targ_medium'].copy()
    df_targ_medium.index = df_targ_medium.index.round('D')
    df_targ_medium = df_targ_medium.resample('D').mean().reset_index(names='date').melt(id_vars='date',var_name='Transect', value_name='shoreline')
    
    # join the wave data - not an elegant solution..
    wave_data_raw = {
        key: val.reset_index(names='date').melt(id_vars='date',var_name='Transect', value_name=key) for key, val in combined_data['dfs_wave'].items()
    }
    wave_data = wave_data_raw['Hs'].merge(
        wave_data_raw['Tp'],
        on=['Transect','date']
    )
    wave_data = wave_data.merge(
        wave_data_raw['Dir'],
        on=['Transect','date']
    )

    # combine the data for output
    tabular_data = {
        'df_obs': df_obs.merge(wave_data, on=['Transect','date']),
        'df_targ_short': df_targ_short.merge(wave_data, on=['Transect','date']),
        'df_targ_medium': df_targ_medium.merge(wave_data, on=['Transect','date'])
    }
    return tabular_data



################################################################################
################################################################################



################################################################################
################################################################################



################################################################################
################################################################################


