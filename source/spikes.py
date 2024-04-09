import numpy as np
import pandas as pd

# class for calculating receptive fields
class rf():
    def __init__(self, stims, window):
        """
        Initializes the class
        
        Parameters
        ----------
        stims : DataFrame
            DataFrame of stimulus presentations
        window : [float, float]
            Window around the event times, [start, end]
        """

        self.window = window
        self.stims = stims
        self.stim_counts = None
        self.spikes = None
        self.dims = None
    
    def _count_spikes(self, spikes, events):
        """
        Counts spike times around each event time
        
        Parameters
        ----------
        spikes : array
            Array of spike times
        events : array
            Array of event times
        
        Returns
        -------
        counts : array of int
            Spike count in each window around events
        """

        # initialize list to store spike times
        counts = np.zeros(len(events))
        
        # loop through each event time
        for i, et in enumerate(events):
            # get the spike times within the window
            start = et+self.window[0] # start of the window
            end = et+self.window[1] # end of the window
            counts[i] = np.sum((spikes > start) & (spikes <= end)) # count the spikes
            
        return counts

    def compute(self, spikes, dims=None):
        """
        Computes the spike counts for each stimulus presentation
        
        Parameters
        ----------
        spikes : array
            Array of spike times
        dims : list of str, optional
            List of dimensions to group stimuli by. If None, all unique stimuli are used
            
        """

        if dims is None:
            # use all unique stimuli
            dims = self.stims.columns
        
        # for each stimulus compute the spike counts
        stim_counts = self.stims.groupby(dims)['start_time'].apply(lambda x: self._count_spikes(spikes, x))
        stim_counts.name = 'spike_counts'
        self.stim_counts = stim_counts.to_frame()
        
        # store spikes and dims for later use
        self.spikes = spikes
        self.dims = dims
        
        # return object so it can be easily used with apply method in dataframe
        return self

    def rf(self, type='count'):
        """
        Returns the spike counts as an array
        
        Parameters
        ----------
        type : str, optional
            Type of receptive field to return. 
            'count' returns the spike counts
            'rate' returns the firing rate
        
        Returns
        -------
        counts : array
            Array of spike counts
        dims : list of str
            List of dimensions
        vals : list of lists
            List of stimulus values for each dimension
        """

        # get the counts as an array
        counts = self.stim_counts['spike_counts'].apply(np.mean).values

        # get the dimensions
        dims = self.stim_counts.index.names

        # get the values for each dimension
        vals = [self.stim_counts.index.get_level_values(d).unique().values.T for d in dims]

        # reshape the counts to match the stimulus dimension values
        counts = counts.reshape([len(v) for v in vals])

        # compute the firing rate if requested
        if type == 'rate':
            counts = counts / (self.window[1] - self.window[0])
        
        return counts, dims, vals
    

# get dataframe of spike times from Allen Institute cache
def spikes_to_df(sess):
    """
    Converts spike_times from session object to a dataframe

    Parameters
    ----------
    sess: session object

    Returns
    -------
    spikes: dataframe of spike times
    """

    # get all spike times in a session
    spike_times = sess.spike_times

    # Create a dataframe from dictionary of spike_times and restrict to units in VISp
    spikes = pd.Series(spike_times) # convert to series since we only have one column of data right now
    spikes.index.name = 'unit_id' # name the index column, same as the index column in units_visp
    spikes.name = 'times' # name the data column
    spikes = spikes.to_frame() # convert to dataframe so we can add additional columns later

    return spikes