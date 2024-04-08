import numpy as np
from scipy.special import factorial
import pandas as pd
import matplotlib.pyplot as plt

# create zscore function with dimension parameter
def zscore(x, axis=None):
    '''
    Z-score data along specified axis
    
    Parameters
    ----------
    x : array
        Data to z-score. Can be N-dimensional.
    axis : int, optional
        Axis along which to z-score data. If None, z-score over all dimensions
    
    Returns
    -------
    zscored_data : array
        Z-scored data
    '''
    return (x - np.mean(x, axis=axis, keepdims=True))/np.std(x, axis=axis, keepdims=True)


# create function to epoch data
def epoch_data(data, event_idxs=None, pre_len=0, post_len=1):
    '''
    Epoch data around events
    
    Parameters
    ----------
    data : array
        Data to epoch. Can be N-dimensional, with last dimension corresponding to time
    event_idxs : array
        Array of event times in samples. If None, return empty array
    pre_len : float
        Samples include before events. Default is 0
    post_len : float
        Samples to include after events. Default is 1
        
    Returns
    -------
    rel_idxs : array
        Array of event indices relative to the event_idx of the epoch
    epoched_data : array
        Data epoched around event times. Will have one additional dimension appended to the beginning,
        corresponding to the epoch number

    '''
    
    # if no event indices are provided, return empty array
    if event_idxs is None:
        return np.array([])

    # ensure event indices are a flattened numpy array of integers
    event_idxs = np.array(event_idxs).reshape(-1,1).astype(int)

    # initialize epoched data array
    epoched_data = np.zeros([event_idxs.size, *data.shape[:-1], pre_len+post_len])
    rel_idxs = np.arange(-pre_len, post_len)

    # loop through events and epoch data
    for i, event in enumerate(event_idxs):
        # indexing with '...,' means to take all indices in all dimensions except the last one
        epoched_data[i] = data[..., event+rel_idxs]# (event-pre_len):(event+post_len)]
    
    return rel_idxs, epoched_data


# plot time series data as stacked lines
def plot_stacked_signals(sig, t, sep=5, ax=None, **kwargs):
    '''
    Plot stacked time series data

    Parameters
    ----------
    sig : array
        Array of signals to plot. Each column will be plotted as a separate signal
    t : array
        Array of time points
    sep : float, optional
        Separation between signals
    ax : matplotlib axis, optional
        Axis on which to plot
    **kwargs : optional
        Additional keyword arguments to pass to matplotlib plot function

    Returns
    -------
    ax : matplotlib axis
        Axis on which data was plotted
    '''

    if ax is None:
        ax = plt.gca()
    for i in range(sig.shape[1]):
        ax.plot(t, sig[:, i] + i*sep, **kwargs)
    ax.set_xlabel('Time (s)')
    ax.legend()
    return ax


def poisson_pdf(k, lamb):
    """
    Computes the Poisson probability mass function

    Parameters
    ----------
    k : array
        Array of counts to evaluate the Poisson PMF
    lamb : float
        Poisson parameter, mean number of counts

    Returns
    -------
    pdf : array
        Array of Poisson probabilities
    """

    # ensure k is an integer
    k = k.astype(int)

    # compute the Poisson probability mass function
    pdf = (np.power(lamb,k)*np.exp(-lamb)) / factorial(k)

    return pdf