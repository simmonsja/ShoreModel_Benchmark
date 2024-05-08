import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

################################################################################
################################################################################
##############################   MAIN FUNCTIONS   ##############################
################################################################################
################################################################################

def plot_prediction(df_Y, mean_mu, hpdi_mu, hpdi_sim, tran_id = 0, ci=0.89, mix=None, hpdi_mix=None, save_loc=None):
    """
    Plot the modelled and observed y values with the modelled and simulated confidence intervals.

    Parameters
    ----------
    df_Y : pandas DataFrame
        The observed y values.
    mean_mu : ndarray
        The mean of the 'mu' variable.
    hpdi_mu : ndarray
        The HPDI of the 'mu' variable.
    hpdi_sim : ndarray
        The HPDI of the 'obs' variable.
    ci : float, optional
        The probability mass of the HPDI. Default is 0.89.
    save_loc : str, optional
        The file path to save the plot. Default is None.

    Returns
    -------
    None
    """
    # Set the seaborn style and context
    sns.set_style('ticks')
    sns.set_context('paper')
    
    # Create a new figure with the specified size
    fig = plt.figure(figsize=(7,3))
    # Create a new subplot for the plot
    ax1 = plt.subplot(111)
    # Plot the modelled y values
    ax1.plot(df_Y.index, mean_mu[:,tran_id], label='Modelled')
    # Shade the area between the upper and lower bounds of the simulated confidence interval
    ax1.fill_between(df_Y.index, hpdi_sim[0,:,tran_id], hpdi_sim[1,:,tran_id], alpha=0.25, color='C1', label='Simulated {:.0f}% CI'.format(ci*100))
    # Shade the area between the upper and lower bounds of the modelled confidence interval
    ax1.fill_between(df_Y.index, hpdi_mu[0,:,tran_id], hpdi_mu[1,:,tran_id], alpha=0.5, color='C0', label='Modelled {:.0f}% CI'.format(ci*100))
    # Plot the observed y values
    ax1.plot(df_Y.index, df_Y.iloc[:,tran_id], label='Observed')
    if not mix is None:
        ax2 = ax1.twinx()
        if len(mix.shape) > 1:
            for i in range(mix.shape[0]):
                ax2.plot(df_Y.index, mix[i,:], color='C{}'.format(i+2), alpha=0.65, label='Mix {}'.format(i))
        else:
            ax2.plot(df_Y.index, mix, color='C2', alpha=0.65, label='Mix')
        if not hpdi_mix is None:
            ax2.fill_between(
                df_Y.index, hpdi_mix[0,:], hpdi_mix[1,:], alpha=0.1, color='C2', label='Mix {:.0f}% CI'.format(ci*100)
            )
    # Set the title, x-axis label, and y-axis label for the plot
    ax1.set_ylabel('Values')
    ax1.set_xlabel('Index')
    # ax1.set_ylim([df_Y.min(), df_Y.max()])
    # Add a legend to the plot
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    # Save the plot if a file path is specified
    if save_loc is not None:
        plt.savefig(save_loc, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # Show the plot
    plt.show()

################################################################################
################################################################################

def plot_prediction_scatter(df_Y, mean_mu, hpdi_mu, hpdi_sim, tran_id=0, ci=0.89):
    """
    Plot the modelled and observed y values as a scatter plot with error bars.

    Parameters
    ----------
    df_Y : pandas DataFrame
        The observed y values.
    mean_mu : ndarray
        The mean of the 'mu' variable.
    hpdi_mu : ndarray
        The HPDI of the 'mu' variable.
    hpdi_sim : ndarray
        The HPDI of the 'obs' variable.
    ci : float, optional
        The probability mass of the HPDI. Default is 0.89.

    Returns
    -------
    None
    """
    # Set the seaborn style and context
    sns.set_style('ticks')
    sns.set_context('paper')

    # Create a new figure with the specified size
    fig = plt.figure(figsize=(4, 4))
    # Create a new subplot for the plot
    ax1 = plt.subplot(111)

    # Constrain x and y lims to be the same and equal to min max between modelled and observed
    ax1.set_xlim([np.nanmin([df_Y.min(), mean_mu.min()]), np.nanmax([df_Y.max(), mean_mu.max()])])
    ax1.set_ylim([np.nanmin([df_Y.min(), mean_mu.min()]), np.nanmax([df_Y.max(), mean_mu.max()])])
    # plot a 1:1 line red dashed
    ax1.plot([np.nanmin([df_Y.min(), mean_mu.min()]), np.nanmax([df_Y.max(), mean_mu.max()])], [np.nanmin([df_Y.min(), mean_mu.min()]), np.nanmax([df_Y.max(), mean_mu.max()])], '--', color='red')

    # Plot error bars for the modelled y values
    if not hpdi_sim is None: 
        ax1.errorbar(df_Y, mean_mu, yerr=[mean_mu - hpdi_sim[0,:], hpdi_sim[1,:] - mean_mu], fmt='none', alpha= 0.1, color='C0')
    if not hpdi_mu is None: 
        ax1.errorbar(df_Y, mean_mu, yerr=[mean_mu - hpdi_mu[0,:], hpdi_mu[1,:] - mean_mu], fmt='none', alpha= 0.6, color='C1', elinewidth=2)
    # Plot the modelled y values as a scatter plot
    ax1.plot(df_Y, mean_mu, 'o', markersize=2, color='C0',label='Modelled')

    # Set the title, x-axis label, and y-axis label for the plot
    ax1.set_ylabel('Modelled Values')
    ax1.set_xlabel('Observed Values')
    # Show the plot
    plt.show()

################################################################################
################################################################################



################################################################################
################################################################################


################################################################################
################################################################################
