import argparse

import datetime
import itertools
import pickle
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.signal import convolve
from scipy.signal import detrend
from sklearn.metrics import r2_score, mean_squared_error, brier_score_loss
from scipy.spatial.distance import euclidean
import os
# from fastdtw import fastdtw
import plot_params
import scipy
#from datetime import datetime

from datetime import timedelta as timedelta
import shorefor



def getCmdargs():
    p = argparse.ArgumentParser()
    p.add_argument("-fp_in", "--filepath_input", type=str, 
                    default="../../datasets",
                    help="Input file directory")
    p.add_argument("-fp_out", "--filepath_output", type=str, 
                    default="../../submissions/ShoreFor",
                    help="Output file directory")
    p.add_argument("-STT", "--start_time_train", type=str, 
                    default='1999-01-01',
                    help="Start time for model prediction")
    p.add_argument("-ETT", "--end_time_train", type=str, 
                    default='2018-12-31',
                    help="End time for training")
    p.add_argument("-STSP", "--start_time_short_pred", type=str, 
                    default='2018-12-31',
                    help="Start time for short-term prediction")        
    p.add_argument("-ETSP", "--end_time_short_pred", type=str, 
                    default='2023-12-31',
                    help="End time for short-term prediction")    
    p.add_argument("-STMP", "--start_time_medium_pred", type=str, 
                    default='1951-05-01',
                    help="Start time for medium-term prediction")        
    p.add_argument("-ETMP", "--end_time_medium_pred", type=str, 
                    default='1998-12-31',
                    help="End time for medium-term prediction")
    p.add_argument("-STLP", "--start_time_long_pred", type=str, 
                    default='2018-12-31',
                    help="Start time for long-term prediction")        
    p.add_argument("-ETLP", "--end_time_long_pred", type=str, 
                    default='2100-12-31',
                    help="End time for long-term prediction")   
    p.add_argument("-retrain", action='store_true',
                    help="End time for long-term prediction")   

    cmdargs = p.parse_args()
    return cmdargs

class ShoreFor(object):
    def __init__(self, g=9.81, ro=1025, gamma=0.78):
        """
        Initialize our ShoreFor model with wave timeseries and shoreline
        measurements. The provided data should contain both training and testing
        periods.

        Args:
            Hs: Pandas time series of signficant wave heights in meters
            Tp: Pandas time series of peak wave periods in seconds
            shoreline_x: Pandas time series of shoreline measurements at some
            constant elevation in meters
            fall_velocity: Pandas time series of fall (settling) velocity, usually
            dependant on grain-size in meters per seconds
            water_depth: Pandas time series of water depth of wave measurements. Used
            for reverse shoalling of wave parameters in meters
            g: Acceleration due to gravity in meters per second squared (defaults to
            9.81)
            ro: Seawater density in kilograms per cubic meter (defaults to 1025)
            gamma: Wave breaking parameter (defaults to 0.78)
        """

        # Get our constants
        self._g = g
        self._ro = ro
        self._gamma = gamma

    def fit(
        self, Hs, Tp, shoreline_x, d50, water_depth, start_time, end_time
    ):
        """
        Using the provided wave and shoreline data, find the best values for the
        response factor (phi), rate parameter (c) and linear term (b) for the
        ShoreFor model which best matches our training data.

        Args:
            start_time: Datetime of the start of the training period
            end_time: Datetime of the end of the training period
            plot: Boolean to enable plotting of training period
        """
        
        d50 = pd.Series(d50, index = Hs.index)
        water_depth = pd.Series(water_depth, index = Hs.index)
        
        
        #Obtain dimensionless fall velocity from d50
        fall_velocity = self._fall_velocity_calc(D = d50/1000, Tw = 15)
        
        
        # Join columns into a data frame
        df = pd.concat(
            [
                Hs.rename("Hs"),
                Tp.rename("Tp"),
                shoreline_x.rename("shoreline_x"),
                fall_velocity.rename("fall_velocity"),
                water_depth.rename("water_depth"),
            ],
            axis=1,
        )
        

        df.index = pd.to_datetime(df.index)
        # Before resampling to hourly records, we want to keep track of which
        # shoreline measurements are our actual measurements. This is so we can test
        # on only these points (rather than the linearly interpolated resampled
        # shorelines).
        df["shoreline_measurement_flag"] = False
        df.loc[~df.shoreline_x.isna(), "shoreline_measurement_flag"] = True

        # Now, resample to hourly records and linearly interpolate over missing records.
        df = df.resample("H").interpolate(method="time")

        # Calculate additional parameter columns in the dataframe
        
        if water_depth.values[0] != np.inf: #i.e. waves are inshore at some finite depth
            df["Hs0"] = self._reverse_shoal(df=df, g=self._g)
        else: #waves are offshore (unknown depth or depth where we filled with reanlys datata)
            df["Hs0"] = df["Hs"]
            
        df["Hsb"] = self._wave_breaking(df=df, g=self._g)
            
            
        
        
        
        df["omega"] = self._non_dimensional_fall_velocity(df=df)
        df["P"] = self._wave_power(df=df, g=self._g, ro=self._ro, gamma=self._gamma)
        # Set the range of timelag values to check.
        phi_ranges = (
            [x for x in range(5, 100, 5)]
             +[x for x in range(100, 500, 20)]
            + [x for x in range(500, 1050, 50)]
        )
        

        # Store fit results in a list
        fit_result = None

        
        
        print("Testing phi values:")
        for phi in phi_ranges:

            df_copy = df.copy()

            # Ensure we remove times where there is no wave information.. Needed to
            # properly calculate omega eq
            try:
                df_copy = df_copy.iloc[np.where(df_copy.Hs.isna())[0][-1] + 1:]
                print("\r"+'phi={:5.1f}'.format(phi),end="")
            except:
                print("\r"+'phi={:5.1f}'.format(phi),end="")
            #df_copy = df_copy.interpolate()

            df_copy = self._time_varying_equilibrium_beach_state(phi, df_copy)
            

            

            #EDIT RAI
            #Modify X time to match the max between last shoreline and inserted period
            dates = [end_time,shoreline_x.index[-1]]
            #print(dates)
            end_time = min(dt for dt in dates)
            #print(end_time)
            
            df_copy = self._trim_to_times(df_copy, start_time, end_time)


            param_fit = self._fit_params(df_copy)

            # Continue if we couldn't fit parameters
            if not param_fit:
                print('Skipping {}'.format(phi))
                continue

            A = param_fit.params[0]
            b = param_fit.params[1]
            c = param_fit.params[2]

            # df_copy = self._time_varying_equilibrium_beach_state(phi, df)
            df_fit = self._fit_shoreline(A = A , b = b, c = c, r = None, df =  df_copy)


            
            r = df_fit.r[0]

            m = df_fit.shoreline_measurement_flag == True
            shoreline_fit = self._ols(df_fit[m].shoreline_x_modelled,
                                      df_fit[m].shoreline_x)
            shoreline_nmse = self._nmse(
                df_fit[m].shoreline_x, df_fit[m].shoreline_x_modelled
            )
            shoreline_rsquared = shoreline_fit.rsquared
            shoreline_rmse = np.sqrt(shoreline_fit.mse_resid)

            # print(df_fit[m].shoreline_x.values)
            # print(df_fit[m].shoreline_x_modelled.values)
            # dtw_distance, _ = fastdtw(df_fit[m].shoreline_x.values,
            #                       df_fit[m].shoreline_x_modelled.values,
            #                       dist=euclidean)

            result = {
                "phi": phi,
                'A' : A,
                "b": b,
                "c": c,
                "r": r,
                "param_fit": param_fit,
                "shoreline_fit": shoreline_fit,
                "df_fit": df_fit,
                "shoreline_nmse": shoreline_nmse,
                "shoreline_correlation": np.sqrt(shoreline_rsquared),
                "shoreline_rmse": shoreline_rmse
                #"distance":dtw_distance
            }

            # print("\r"+'phi={:5.1f}, NMSE={:8.5f}, RMSE={:6.2f}, R={:5.2f}, '
            #       'dist={:5.2f}'.format(phi, shoreline_nmse, shoreline_rmse,
            #                           np.sqrt(shoreline_rsquared),dtw_distance),end="")
            


            if not fit_result:
                fit_result = result
                continue

            elif shoreline_nmse < fit_result["shoreline_nmse"]:
                fit_result = result
                continue
            
            

        return fit_result

    @staticmethod
    def _fall_velocity_calc(D,Tw=15):
            

        """
        Calculates dimensionless fall velocity assuming T = 15 deg C
        Args:
            D: d50 in [m]
            g: Acceleration due to gravity
        Returns:
            dimensionless fall velocity in m/s
        """
        D=D*100
        ROWs=2.75 #Density of sand 
        g=981     #Gravity n cm**2/s
        
        T=np.array([5 ,10, 15, 20, 25])
        v   =np.array([0.0157, 0.0135, 0.0119, 0.0105, 0.0095])
        ROW =np.array([1.028, 1.027, 1.026, 1.025, 1.024])
        
        vw=np.interp(Tw,T,v)
        ROWw=np.interp(Tw,T,ROW)    
        
        A=((ROWs-ROWw)*g*(D**3))/(ROWw*(vw**2))
        w= pd.Series(0,index=A.index)
        
        mask1 =  A < 39
        w[mask1] = ((ROWs-ROWw)*g*(D**2))/(18*ROWw*vw)
        
        mask2 = (A < 10**4) & (A > 39)
        if any(mask2):
            w[mask2] = ((((ROWs-ROWw)*g/ROWw)**0.7)*(D**1.1))/(6*(vw**0.4))   
        mask3 = A > 10**4
        if any(mask3):
            w[mask3] = np.sqrt(((ROWs-ROWw)*g*D)/(0.91*ROWw))
        w=w/100 #convert to SI (m/s)
        
        return w
    

    def predict(
        self,
        Hs,
        Tp,
        d50,
        water_depth,
        start_time,
        end_time,
        fit_result,        
        shoreline_x,
        b_zero
    ):
        """
        Using the fitted ShoreFor parameters (self._b, self._c and self._phi),
        predict the shoreline for the defined time period.

        Args:
            start_time: Datetime of the start of the prediction period
            end_time: Datetime of the end of the prediction period
        """
        #Obtain dimensionless fall velocity from d50
        d50 = pd.Series(d50, index = Hs.index)
        water_depth = pd.Series(water_depth, index = Hs.index)
        
        fall_velocity = self._fall_velocity_calc(D = d50/1000, Tw = 15)
        
        # If fit_parameters are not defined, look for them in the class. If the model
        # has been fitted, they should be available

        # Join columns into a data frame
        df = pd.concat(
            [
                Hs.rename("Hs"),
                Tp.rename("Tp"),
                fall_velocity.rename("fall_velocity"),
                water_depth.rename("water_depth"),
                shoreline_x.rename('shoreline_x')
            ],
            axis=1,
        )

        # Before resampling to hourly records, we want to keep track of which
        # shoreline measurements are our actual measurements. This is so we can test
        # on only these points (rather than the linearly interpolated resampled
        # shorelines).
        df["shoreline_measurement_flag"] = False
        df.loc[~df.shoreline_x.isna(), "shoreline_measurement_flag"] = True

        # Now, resample to hourly records and linearly interpolate over missing records.
        df = df.resample("H").interpolate(method="time")

        # Calculate additional parameter columns in the dataframe
        if water_depth.values[0] != np.inf: #i.e. waves are inshore at some finite depth
            df["Hs0"] = self._reverse_shoal(df=df, g=self._g)
        else: #waves are offshore (unknown depth or depth where we filled with reanlys datata)
            df["Hs0"] = df["Hs"]

        df["Hsb"] = self._wave_breaking(df=df, g=self._g)
        df["omega"] = self._non_dimensional_fall_velocity(df=df)
        df["P"] = self._wave_power(df=df, g=self._g, ro=self._ro, gamma=self._gamma)
        
        #EDIT Rai
        #Create an auxiliar b term that will be used in case we want to forecast without the hindcasted b
        b_cero =  pd.Series(0, index = df.index) #will be zero during forecasting period
        b_cero[b_cero.index<=fit_result['df_fit'].index[-1]] = fit_result['b']

        # Ensure we remove times where there is no wave information.. Needed to
        # properly calculate omega eq
        try:
            df = df.iloc[np.where(df.Hs.isna())[0][-1] + 1:]
        except:
            pass

        # Note that we need to calculate the time varying equilibirum beach state
        # BEFORE trimming to the prediction times. This is time varying equilibrium
        # beach state is dependant on the previous time steps, and if we trim them,
        # we won't be able to calculate the value.
        df = self._time_varying_equilibrium_beach_state(fit_result['phi'], df)
        df = self._trim_to_times(df, start_time, end_time)
        b_cero  = self._trim_to_times(b_cero, start_time, end_time)

        # Get predicted shorelines
        
        if b_zero:
            #Rai; We allocate b values of hindcasted magnitud and then 0 for forecasting period
            fit_result['A'] = fit_result['df_fit']['shoreline_x_modelled'].iloc[0] # to start from last modeled shoreline

            df_predict = self._fit_shoreline(fit_result['A'], b_cero , fit_result[
                'c'], fit_result[
                'r'] , df) #Predicting using the r from calib
        else:
            df_predict = self._fit_shoreline(fit_result['A'],fit_result['b'], fit_result[
                'c'],fit_result['r'], df) #Predicting using the r from calib
            

        return df_predict


    @classmethod
    def plot_fit(cls, fit_result, df_pred=None,output_path = None, output_file=None, start_time=None,
                 end_time=None):

        # Setup default plot parameters
        plot_params.setup()

        # Get fit dataframe
        df = fit_result["df_fit"]

        # Trim dataframe if required
        #We will trim until the last observed shoreline
        if df_pred is None:
            end_time = fit_result['df_fit']['shoreline_x'].index[-1]
        
        df = cls._trim_to_times(df, start_time, end_time)

        # Initalize the plot and add subplots
        #fig,axs = plt.subplots(4,1, dpi=300)
        fig = plt.figure(figsize=(4.5, 3.5), dpi=300)
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = fig.add_subplot(gs[0, 0])  # shorelines
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # non-dimensional fall velocity

        # Axis 1 - shorelines
        m = df.shoreline_measurement_flag == True
        ax1.plot(
            df.index, df.shoreline_x_modelled, label="Calibration", ls="-",
            linewidth=1,c='#1f78b4'
        )
        ax1.plot(
            df[m].index,
            df[m].shoreline_x,
            marker="o",
            label="Measured",
            c="#000000",
            markersize=0.5,
            linestyle="None",
        )
        ax1.plot(df[m].index,
                 df[m].shoreline_x,
                 alpha=0.2,
                 c="#000000",
                 linewidth=1)

        if df_pred is not None:
            
            #EDIT RAI: to plot only after the calibration period
            #df_pred =  df_pred[df_pred.index > fit_result['df_fit'].index[-1]]
            
            m = df_pred.shoreline_measurement_flag == True
            
            ax1.plot(df_pred.index,
                     df_pred.shoreline_x_modelled,
                     label='Predicted',
                     linestyle='--',
                     linewidth=1,
                     c='#ff7f00')
            
            ax1.plot(df_pred[m].index,
                     df_pred[m].shoreline_x,
                     marker="o",
                     c="#000000",
                     markersize=0.5,
                     linestyle="None")
            ax1.plot(df_pred[m].index,
                     df_pred[m].shoreline_x,
                     alpha=0.2,
                     c="#000000",
                     linewidth=1)

        ax1.set_ylabel("Shoreline (m)")
        ax1.legend(loc="upper right", fontsize = 5)

        # Axis 2 - Omega
        ax2.plot(df.index, df.omega, label="$\Omega$", c="#000000", linewidth=0.1)
        ax2.plot(df.index, df.omega_eq, label="$\Omega_{eq}$", c="#000000",ls="--",
                 linewidth=1)        
        ax2.set_ylabel("$\Omega$ (-)")

        if df_pred is not None:
            ax2.plot(df_pred.index, df_pred.omega, c='#ff7f00', linewidth=0.1)
            ax2.plot(df_pred.index, df_pred.omega_eq, c='#ff7f00', ls="--", linewidth=1)

        ax2.legend(loc="upper right", fontsize = 5)
        # Remove ticks from shared axes
        ax1.tick_params(labelbottom=False)
        
        ax1.grid(True)
        ax2.grid(True)

        # Add parameters in text box to axis 1
        textstr = "\n".join(['CALIBRATION RESULTS',
              '$\overline{\Omega}$ = '+ '%.2f' % df.omega.mean() + '(-)',  
             "$b$ = {:.2f} m/yr".format(fit_result["b"]*60*60*24*365),
                 '$c$ =' + '%.2e' % fit_result["c"] + ' $m^{1.5}s^{-1}W^{-0.5}$',
                 "$r$ = {:.2f} (-) ".format(fit_result["r"]),
                 "$\phi$ = {:.0f} days".format(fit_result["phi"]),
                 "RMSE = {:.2f} m".format(fit_result["shoreline_rmse"]),
                 "NMSE = {:.2f}".format(fit_result["shoreline_nmse"]),
                 "$R$"+"= {:.2f}".format(fit_result["shoreline_correlation"]),
             ])
        
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax1.text(
        0.02,
        0.70,
        textstr,
        transform=ax1.transAxes,
        fontsize=4,
        verticalalignment="bottom",
        bbox=props,
        )
        
        plt.tight_layout()

        # if output_path:
        #         fig.savefig(os.path.join(output_path,output_file), dpi=300, bbox_inches="tight", pad_inches=0.01)
        #         print('\nSaved plot to {}'.format(os.path.join(output_path,output_file)))
        # else:
        #         fig.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.01)
        #         print('Saved plot to {}'.format(output_file))

        return fig

    @classmethod
    def _fit_params(cls, df):

        df = df.dropna(subset=["omega_eq"])
        df = cls._forcing_terms(df)

        #### TODO Fix this.

        m = df.shoreline_measurement_flag==True
        #print(m)

        # Calculate time between shoreline measurements and convert to seconds
        dt = df[m].index.to_series().diff()
        dt = dt / np.timedelta64(1, "s")
        #print(dt)
        dt.iloc[0] = 0
        dt = dt.cumsum()
        dt = dt.rename('dt')

        X = pd.concat([dt, df.forcing.cumsum()[m]*3600],axis=1)
        X = sm.add_constant(X)
        Y = df.shoreline_x[m]

        model_rls = sm.RLM(Y,X)
        param_fit_rls = model_rls.fit()



        return param_fit_rls

    @classmethod
    def _fit_shoreline(cls, A, b, c, r, df):
        
        


        df = df.dropna(subset=["omega_eq"])
        df = cls._forcing_terms(df)

        # Change in shoreline position (in m/s)
        if r is None: #for calibration
            df["dx_dt"] = c * (df.F_acc + df.r * df.F_ero) + b
        else:  #for prediction using r obtained from calibration period
            print('Using r from calib period to predict future shoreline change')
            df["dx_dt"] = c * (df.F_acc + r * df.F_ero) + b
            

        # Start at original shoreline, then calculate modelled shoreline.
        # Remember that dx_dt is given in m/s

 #       if shoreline_x_start == None:
#            #shoreline_x_start = df.iloc[0].shoreline_x
        shoreline_x_start = A

        df["shoreline_x_modelled"] = df.dx_dt.shift(1) * 3600
        df.loc[df.iloc[0].name, "shoreline_x_modelled"] = shoreline_x_start 
        df.shoreline_x_modelled = df.shoreline_x_modelled.cumsum()

        return df

    def _ols(self,X, Y):
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        return model.fit()

    @classmethod
    def _forcing_terms(cls, df):

        # Need to make a copy
        df = df.copy()

        # Disequilibrium term (Splinter et al, 2014 Eqn 7)
        df["d_omega"] = df.omega_eq - df.omega
        
        # Forcing term (Splinter et al, 2014 Eqn 5)
        F = (df.P ** 0.5) * df.d_omega / np.nanstd(df.d_omega)
       
        
        # We want to split the forcing term by if it's positive or negative
        df["F_ero"] = 0
        df["F_acc"] = 0
        mask_positive_omega = df.d_omega > 0
        df.loc[mask_positive_omega, "F_acc"] = F[mask_positive_omega]
        df.loc[~mask_positive_omega, "F_ero"] = F[~mask_positive_omega]

        # Erosion ratio
        df["r"] = cls._erosion_ratio(df.F_acc, df.F_ero)

        # Create the forcing term
        df["forcing"] = df.F_acc + df.r * df.F_ero

        return df

    @classmethod
    def _shoreline_rate_of_change(self, df):

        # Create mask for rows where the actual shoreline was measured (i.e. not
        # interpolated values).
        m = df.shoreline_measurement_flag == True

        # Calculate change in shoreline between measurements
        dx = df[m].shoreline_x.diff()

        # Calculate time between shoreline measurements and convert to seconds
        dt = df[m].index.to_series().diff()
        dt = dt / np.timedelta64(1, "s")

        # Measured dx/dt values are thus given by:
        dx_dt = (dx / dt).rename("dx_dt")

        return dx_dt

    @classmethod
    def _forcing_summation(cls, df, index):

        # We want to sum the forcings between the times where we have shoreline
        # measurements, so we can fit a linear regression with respect to dx/dt

        # Iterate through each pair of indicies, getting a mask for the start and end
        # time so we can do the summation
        masks = [(df.index >= x1) & (df.index <= x2) for x1, x2 in cls.pairwise(index)]

        # # Get number of seconds in each mask. Need to do this to make sure units are
        # # consistent with dx/dt
        # secs = [(df[m].iloc[-1].name - df[m].iloc[0].name) / np.timedelta64(1, "s") for m in masks]

        # Sum the forcing terms for each mask
        # TODO This line could probably be faster
        sums = [df[m].forcing.sum(min_count=1) for m in masks]

        # sums = [df[m].forcing.mean() for m in masks]
        #
        # from scipy.integrate import trapz
        # sum = [trapz(df[m].forcing) for m in masks]

        # Define the index for the new forcings dataframe to be the endtimes for the
        # given index
        idx = index[1:]

        # Create the new dataframe
        forcings = pd.DataFrame(sums, index=idx)
        forcings.columns = ["forcings"]

        return forcings

    @staticmethod
    def _nmse(y_true, y_pred):
        # Equation written in Splinter et al. (2014) doesn't show th denominator
        # having the y_true.mean() term in it, but Rai's port does.
        return ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean())** 2).sum()

    @staticmethod
    def _reverse_shoal(df, g):
        """
        Calculates the deep water wave height given an inshore wave height at a
        particular depth

        Args:
            df: Input dataframe with column for water_depth, Tp, and Hs
            g: Acceleration due to gravity

        Returns:
            Pandas series of reverse shoalled wave heights

        """
        y = 4.03 * df.water_depth / (df.Tp ** 2)
        kd2 = y ** 2 + y / (
            1
            + (0.666 * y)
            + (0.355 * y ** 2)
            + (0.161 * y ** 3)
            + (0.0632 * y ** 4)
            + (0.0218 * y ** 5)
            + (0.00564 * y ** 6)
        )
        kh = np.sqrt(kd2)
        Cg = (
            g
            * df.Tp
            / (2 * np.pi)
            * (np.tanh(kh))
            * (0.5 * (1 + 2 * kh / np.sinh(2 * kh)))
        )
        Cgo = 1 / 4 * g * df.Tp / np.pi
        Ks = np.sqrt(Cgo / Cg)
        Hs0 = df.Hs / Ks
        return Hs0

    @staticmethod
    def _wave_breaking(df, g):
        """
        Calculates the breaking wave height from deep water wave height,
        in accordance with Komar (1974).

        Args:
            df: Pandas dataframe with columns for Tp, Hs0
            g: Acceleration due to gravity

        Returns:
            Pandas series of breaking wave height
        """
        Hsb = 0.39 * g ** (1 / 5) * (df.Tp * df.Hs0 ** 2) ** (2 / 5)
        return Hsb

    @staticmethod
    def _non_dimensional_fall_velocity(df):
        """
        Calculates the non-dimensional fall velocity (omega) for a dataframe.

        Args:
            df: Pandas dataframe with columns for Hsb, fall_velocity and Tp

        Returns:
            Pandas series of non-dimensional fall velocity
        """
        omega = df.Hsb / (df.fall_velocity * df.Tp)
        return omega

    @staticmethod
    def _wave_power(df, g, ro, gamma):
        """
        Calculates wave power for a dataframe

        Args:
            df: Pandas dataframe with columns for Hsb
            g: Acceleration due to gravity
            ro: Density of seawater
            gamma: Wave breaking parameter

        Returns:
            Pandas series of wave power
        """
        E = 1 / 16 * ro * g * df.Hsb ** 2
        Cg = np.sqrt(g * df.Hsb / gamma)
        P = E * Cg
        return P

    @staticmethod
    def _time_varying_equilibrium_beach_state(phi, df):
        """
        Calculates the time varying equilibrium beach state (omegea_eq). Refer to
        Splinter et al. (2014), Eqn 8 for the definition of this term.
        Args:
            phi: Reponse factor
            df: Pandas dataframe with column for omega

        Returns:
            Dataframe with a new column for omega_eq
        """

        df = df.copy()
        


        
        

        # Calculate the weights to multiply the omega term in the numerator
        weights = np.power(10, -np.arange(int(2 * phi * 24), 0, -1) / int(phi * 24))
        # i = np.arange(1, int(2 * phi * 24) + 1)
        # weights = np.power(10.0, -i / int(phi * 24))

        # The numerator term is just the convolution (rolling sum) of omega term
        # multipyled by the weights. We only want to return the values where we have
        # the full rolling sum returned (which means some portion of the starting
        # values will get discarded because there aren't enough values).

        # This calculation can be quite slow - the quickest way I found was to use
        # scipy's convolve function (even faster than numpy!).
        numerator = convolve(df.omega, weights, mode="valid")

        # Denominator is just the sum of the weights
        denominator = weights.sum()
        omega_eq = numerator / denominator

        # Since we can't return a valid value for each row (because we need a certain
        # number of observations before we can start return values), we need to
        # assign our new values to the end of the dataframe.
        df.loc[df.iloc[: -len(omega_eq)].index, "omega_eq"] = np.nan
        df.loc[df.iloc[-len(omega_eq) :].index, "omega_eq"] = omega_eq
        return df

    @staticmethod
    def _erosion_ratio(F_acc, F_ero):
        """
        Calculate erosion ratio as defined by Splinter et al (2014) Eqn 4

        Args:
            F_acc: Pandas series of accretion forcing term
            F_ero: Pandas series of erosion forcing term

        Returns:
            Float of the erosion ratio
        """
        # (Splinter et al, 2014 Eqn 4)

        # F = pd.concat([F_acc[F_acc!=0],F_ero[F_ero!=0]]).sort_index()
        # F_detrended = detrend(F) + np.mean(F)
        # r = np.abs(F_detrended[F_detrended>0].sum() / F_detrended[F_detrended<0].sum())
        # return r
        
        
        numerator = np.sum(detrend(F_acc.dropna()) + np.nanmean(F_acc))
        
        denominator = np.sum(detrend(F_ero.dropna()) + np.nanmean(F_ero))
        r = np.abs(numerator / denominator)
        return r

    @staticmethod
    def _trim_to_times(df, start_time, end_time):
        """
        Trims a given dataframe based on the times in the index.

        Args:
            df: Dataframe to trim
            start_time: Remove rows before this time
            end_time: Remove rows after this time

        Returns:
            Dataframe with the rows removed.
        """
        # Drop times outside the time limits
        if start_time:
            df = df[df.index >= start_time]
        if end_time:
            df = df[df.index <= end_time]

        if len(df) == 0:
            raise ValueError(
                "Got an empty dataframe for the selected times. Check "
                "start_time and end_time, and make sure you have wave & shoreline "
                "data provided for this period."
            )

        return df

    @staticmethod
    def _rmse(y_true, y_pred):
        """
        Returns the root mean squared error of array of predictions

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Float of RMSE
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def _r2(y_true, y_pred):
        """
        Returns the R2 score of an array of predictions

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Float of R2 score
        """
        return r2_score(y_true, y_pred)

    @staticmethod
    def _bss(y_true, y_pred):
        """
        Returns the Brier Skill Score of an array of predictions

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Float of the BSS
        """
        return 1 - brier_score_loss(y_true, y_pred)

    @staticmethod
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)


if __name__ == "__main__":
    
    cmdargs = getCmdargs()
    
    fp_in = cmdargs.filepath_input
    fp_out = cmdargs.filepath_output
    
    START_TIME_TRAIN = cmdargs.start_time_train
    END_TIME_TRAIN = cmdargs.end_time_train
    START_TIME_SHORT_PRED = cmdargs.start_time_short_pred
    END_TIME_SHORT_PRED = cmdargs.end_time_short_pred
    START_TIME_MEDIUM_PRED = cmdargs.start_time_medium_pred
    END_TIME_MEDIUM_PRED = cmdargs.end_time_medium_pred
    START_TIME_LONG_PRED = cmdargs.start_time_long_pred
    END_TIME_LONG_PRED = cmdargs.end_time_long_pred
    
    SITE = 'Beach_X'
    WAVE_PARAMS = ['Hs', 'Tp', 'Dir']
    
    retrain = cmdargs.retrain
    
    # fn_target = 'shorelines/shorelines_target_short.csv'
    #fn_predict = 'shorelines_prediction.csv'
    
    
    # #Loading shoreline positions
    shorelines_obs = pd.read_csv(
        os.path.join(fp_in, 'shorelines', 'shorelines_obs.csv'), 
        index_col='Datetime')
    shorelines_obs.index = pd.to_datetime(shorelines_obs.index)
    shorelines_obs.index = shorelines_obs.index.round('H')
    
    shorelines_medium_targ = pd.read_csv(
        os.path.join(fp_in, 'shorelines/shorelines_target_medium.csv'), 
        parse_dates=['Datetime'], index_col='Datetime')
    # shorelines_targ.index = pd.to_datetime(shorelines_targ.index)
    

    
    
    dfs_wave_hindcast = {}
    dfs_wave_RCP45 = {}
    dfs_wave_RCP85 = {}
    for wave_param in WAVE_PARAMS:
        df_wave_hindcast = pd.read_csv(
            os.path.join(fp_in, 'hindcast_waves','{}.csv'.format(wave_param)),
            index_col = 'Datetime'
        )
        df_wave_RCP45 = pd.read_csv(
            os.path.join(fp_in, 'forecast_waves', 'RCP45', '{}.csv'.format(wave_param)),
            index_col = 'Datetime'
        )
        df_wave_RCP85 = pd.read_csv(
            os.path.join(fp_in, 'forecast_waves', 'RCP85', '{}.csv'.format(wave_param)),
            index_col = 'Datetime'
        )
        df_wave_hindcast.index = pd.to_datetime(df_wave_hindcast.index)
        df_wave_RCP45.index = pd.to_datetime(df_wave_RCP45.index)
        df_wave_RCP85.index = pd.to_datetime(df_wave_RCP85.index)
        dfs_wave_hindcast[wave_param] = df_wave_hindcast
        dfs_wave_RCP45[wave_param] = df_wave_RCP45
        dfs_wave_RCP85[wave_param] = df_wave_RCP85
     
    # Set params for train and pred
    time_scales = ['short', 'medium', 'RCP45', 'RCP85']
    #time_scales = ['medium']
    preds = dict(zip(time_scales, [[] for _ in range(len(time_scales))]))
    start_time_train = datetime.datetime.strptime(START_TIME_TRAIN, '%Y-%m-%d')
    end_time_train = datetime.datetime.strptime(END_TIME_TRAIN, '%Y-%m-%d')
    start_time_preds = {
        'short': shorelines_obs.index[~shorelines_obs.isna().any(axis=1)][-1], # Date for last obs without nan
        'medium': shorelines_medium_targ.index[0],
        'RCP45': shorelines_obs.index[~shorelines_obs.isna().any(axis=1)][-1], # Date for last obs without nan
        'RCP85': shorelines_obs.index[~shorelines_obs.isna().any(axis=1)][-1] # Date for last obs without nan
        }
    end_time_preds = {
        'short': datetime.datetime.strptime(END_TIME_SHORT_PRED, '%Y-%m-%d'), 
        'medium': datetime.datetime.strptime(END_TIME_MEDIUM_PRED, '%Y-%m-%d'), 
        'RCP45': datetime.datetime.strptime(END_TIME_LONG_PRED, '%Y-%m-%d'),
        'RCP85': datetime.datetime.strptime(END_TIME_LONG_PRED, '%Y-%m-%d')
        }
    
    refs = {
        'short': shorelines_obs[~shorelines_obs.isna().any(axis=1)].iloc[-1], # Last obs without nan
        'medium': shorelines_medium_targ.iloc[0], # Context data provided in medium target
        'RCP45': shorelines_obs[~shorelines_obs.isna().any(axis=1)].iloc[-1], # Last obs without nan
        'RCP85': shorelines_obs[~shorelines_obs.isna().any(axis=1)].iloc[-1]
        }

    
    # Iterate over transects
    
    for i, tran_id in enumerate(shorelines_obs.columns):
        shoreline_x = pd.Series(shorelines_obs[tran_id])
        
        
        dict_waves_hindcast = {}
        dict_waves_RCP45 = {}
        dict_waves_RCP85 = {}
        for wave_param in WAVE_PARAMS:
            dict_waves_hindcast[wave_param] = dfs_wave_hindcast[wave_param][tran_id]
            dict_waves_RCP45[wave_param] = dfs_wave_RCP45[wave_param][tran_id]
            dict_waves_RCP85[wave_param] = dfs_wave_RCP85[wave_param][tran_id]
        df_waves_hindcast = pd.DataFrame(dict_waves_hindcast)
        df_waves_RCP45 = pd.DataFrame(dict_waves_RCP45)
        df_waves_RCP85 = pd.DataFrame(dict_waves_RCP85)
    
        # #There are some negative values around
        df_waves_hindcast.loc[df_waves_hindcast['Tp'] < 0, 'Tp'] = np.nan
        df_waves_RCP45.loc[df_waves_RCP45['Tp'] < 0, 'Tp'] = np.nan
        df_waves_RCP85.loc[df_waves_RCP85['Tp'] < 0, 'Tp'] = np.nan
        # #############################
        df_waves_hindcast = df_waves_hindcast.interpolate(method ='linear')
        df_waves_RCP45 = df_waves_RCP45.interpolate(method ='linear')
        df_waves_RCP85 = df_waves_RCP85.interpolate(method ='linear')
        
        depth = 10 #from gold coast buoy
        d50 = 0.3 # mm
        # Initalize ShoreFor model
        model = shorefor.ShoreFor()
        
        
        #======================================================================
        # Model fit
        #======================================================================
        # Fit model based on data within the start and end times
        fit_result_hindcast = model.fit(
            Hs=df_waves_hindcast.Hs,
            Tp=df_waves_hindcast.Tp,
            shoreline_x=shoreline_x,
            d50=d50,
            water_depth=depth,
            start_time=start_time_train,
            end_time=end_time_train)
        
        if ('RCP45' in time_scales)&(retrain):
        
            fit_result_RCP45 = model.fit(
                Hs=df_waves_RCP45.Hs,
                Tp=df_waves_RCP45.Tp,
                shoreline_x=shoreline_x,
                d50=d50,
                water_depth=depth,
                start_time=start_time_train,
                end_time=end_time_train)
        
        if ('RCP85' in time_scales)&(retrain):
        
            fit_result_RCP85 = model.fit(
                Hs=df_waves_RCP85.Hs,
                Tp=df_waves_RCP85.Tp,
                shoreline_x=shoreline_x,
                d50=d50,
                water_depth=depth,
                start_time=start_time_train,
                end_time=end_time_train)
        
        #======================================================================
        # Model predict
        #======================================================================
        for time_scale in time_scales:
            start_time_pred = start_time_preds[time_scale]      
            end_time_pred = end_time_preds[time_scale]      
            
            #use b_zero = True if you want the forecast to be cross-shore dominated only (i.e. no long-term trend)
            if time_scale == 'RCP45':
                df_waves = df_waves_RCP45
                if retrain:
                    fit_result = fit_result_RCP45
                else:
                    fit_result = fit_result_hindcast
                    
            elif time_scale == 'RCP85':
                df_waves = df_waves_RCP85
                if retrain:
                    fit_result = fit_result_RCP85
                else:
                    fit_result = fit_result_hindcast
                    
            else:
                df_waves = df_waves_hindcast
                fit_result = fit_result_hindcast
            
            df_pred = model.predict(
                Hs=df_waves.Hs,
                Tp=df_waves.Tp,
                d50=d50,
                water_depth=depth,
                start_time=start_time_pred,
                end_time= end_time_pred,
                shoreline_x = shoreline_x,
                fit_result=fit_result,
                b_zero = True)
            
            df_pred['shoreline_x_modelled'] = df_pred['shoreline_x_modelled'] - df_pred['shoreline_x_modelled'].iloc[0] + refs[time_scale][tran_id]
            pred = df_pred['shoreline_x_modelled'].resample('D').interpolate().rename(tran_id)
            # Calibrate pred to ref
            preds[time_scale].append(pred)
        
            fig = shorefor.ShoreFor.plot_fit(fit_result, df_pred=df_pred)
            if retrain:
                fig.savefig('figures/retrain/{}_{}.jpg'.format(time_scale, tran_id),
                            dpi=300)
            else:
                fig.savefig('figures/{}_{}.jpg'.format(time_scale, tran_id),
                            dpi=300)
            plt.close(fig)
        
        # shorelines_cali[tran_id] = df_pred['shoreline_x_modelled'].reindex(
        #     shorelines_cali.index, method='Nearest')
        # shorelines_targ[tran_id] = df_pred['shoreline_x_modelled'].reindex(
        #     shorelines_targ.index, method='Nearest')
        
    for time_scale in time_scales:
        df_preds = pd.concat(preds[time_scale], axis=1)
        if not os.path.exists(fp_out):
            os.makedirs(fp_out)
        #shorelines_targ.to_csv(os.path.join(fp_out, fn_predict))
        #shorelines_cali.to_csv(os.path.join(fp_out, 'shorelines_calibration.csv'))
        #df_pred[df_pred.index<=end_time_train].to_csv(os.path.join(fp_out, 'shorelines_calibration.csv'))
        if retrain:
            df_preds.to_csv(
                os.path.join(fp_out, 'retrain', 'shorelines_prediction_{}.csv'.format(time_scale))
                )
        else:
            df_preds.to_csv(
                os.path.join(fp_out, 'shorelines_prediction_{}.csv'.format(time_scale))
                )
    
    
    
    
    
