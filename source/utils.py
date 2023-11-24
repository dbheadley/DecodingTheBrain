import numpy as np
import matplotlib.pyplot as plt

# create zscore function with dimension parameter
def zscore(x, axis=None):
    '''
    Z-score data along specified axis
    
    Parameters
    ----------
    x : array
        Data to z-score. Can be N-dimensional, with last dimension corresponding to time
    axis : int, optional
        Axis along which to z-score data. If None, z-score over all dimensions
    
    Returns
    -------
    zscored_data : array
        Z-scored data
    '''
    return (x - np.mean(x, axis=axis, keepdims=True))/np.std(x, axis=axis, keepdims=True)


# create function to epoch data
def epoch_data(data, fs, event_t, pre_t, post_t):
    '''
    Epoch data around events
    
    Parameters
    ----------
    data : array
        Data to epoch. Can be N-dimensional, with last dimension corresponding to time
    fs : int
        Sampling rate of data
    event_t : array
        Array of event times in seconds
    pre_t : float
        Time in seconds to include before event time
    post_t : float
        Time in seconds to include after event time
        
    Returns
    -------
    epoched_data : array
        Data epoched around event times. Will have one additional dimension appended to the beginning,
        corresponding to the epoch number

    '''
    # convert pre/post event times to samples
    pre_len = int(pre_t*fs)
    post_len = int(post_t*fs)
    event_ind = np.round(event_t*fs).astype(int)
    
    # initialize epoched data array
    epoched_data = np.zeros([event_t.size, *data.shape[:-1], pre_len+post_len])

    # loop through events and epoch data
    for i, event in enumerate(event_ind):
        # indexing with '...,' means to take all indices in all dimensions except the last one
        epoched_data[i] = data[..., (event-pre_len):(event+post_len)]
    
    return epoched_data


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