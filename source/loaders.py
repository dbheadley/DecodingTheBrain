import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import spectrogram
import scipy.ndimage as sn
from nilearn import plotting
from nimare import utils
from source.utils import zscore, epoch_data

class EcogFingerData():
    def __init__(self, fpath, fs=1000, finger_lut=None, spec_win=0.25, spec_step=0.05):
        """
        Loads ECoG finger data.

        Parameters
        ----------
        fpath : str
            Path to data file
        fs : float
            Sampling rate of data
        finger_lut : dict
            Dictionary mapping finger names to indices
        spec_win : float
            Window length for spectrogram in seconds
        spec_step : float
            Step size for spectrogram in seconds
        """

        self.fpath = fpath
        self.fs = fs
        
        if finger_lut is None:
            self.finger_lut = {'thumb':0, 'index':1, 'middle':2, 'ring':3, 'little':4}
        else:
            self.finger_lut = finger_lut
        
        data = loadmat(fpath)
        self.ecog = data['data'].T # channels x time
        self.regions = data['elec_regions']
        self.flex = data['flex'].T # fingers x time
        self.brain = data['brain']
        self.locs = data['locs']
        self.cue = data['cue']
        self._ch_num = self.ecog.shape[0]

        # calculate spectrogram
        win_len = int(spec_win*fs)
        overlap_len = int((spec_win-spec_step)*fs) # E.g. 250 ms - 50 ms = 200 ms overlap
        spec_temp = []
        for i in range(self._ch_num):
            f, t, s = spectrogram(self.ecog[i,:], fs=self.fs, window='hann', nperseg=win_len, noverlap=overlap_len)
            spec_temp.append(s)

        self.spec = np.stack(spec_temp, axis=0) # channels x freq x time
        self.spec_fs = 1/np.mean(np.diff(t)) # calculate spectrogram sampling rate
        self.spec_f = f
        self.spec_t = t

    def get_sig(self, chans=None, event_times=None, pre_t=None, post_t=None):
        """
        Gets a signal from the data.
        
        Parameters
        ----------
        chans : array_like
            Array of channels to get signal from. If None, all channels will be used.
        event_times : array_like
            Array of event times. If None, all times will be used.
        pre_t : float
            Time before event to include in signal (in seconds). Default is 0.
        post_t : float
            Time after event to include in signal (in seconds)
            
        Returns
        -------
        rel_t : array_like
            Array of times relative to event.
        sig : array_like
            Array of signal. Will have one additional dimension appended to the beginning,
            corresponding to the epoch number.
        """
        
        if chans is None: # if no channels specified, use all
            chans = np.arange(self._ch_num).reshape(-1,1)
        else:
            chans = np.array(chans).reshape(-1,1)
        
        if event_times is None: # if no event times specified, use entire session
            event_idxs = int(0)
            pre_len = int(0)
            post_len = self.ecog.shape[1]
        else:
            event_idxs = np.round(event_times*self.fs).astype(int)

        if pre_t is None: # if no pre time specified, start at event time
            pre_len = int(0)
        else:
            pre_len = np.round(pre_t*self.fs).astype(int)
        
        if post_t is None: # if no post time specified, use single time sample
            post_len = 1
        else:
            post_len = np.round(post_t*self.fs).astype(int)

        rel_idxs, sig = epoch_data(self.ecog[chans,:], event_idxs=event_idxs, 
                          pre_len=pre_len, post_len=post_len)

        rel_t = rel_idxs/self.fs
        return rel_t, sig
    
    def get_spec(self, chans=None, freq_min=None, freq_max=None, event_ts=None, pre_t=None, post_t=None):
        """
        Gets a spectrogram from the data.
        
        Parameters
        ----------
        chans : array_like
            Array of channels to get signal from. If None, all channels will be used.
        freq_min : float
            Minimum frequency to include in spectrogram
        freq_max : float
            Maximum frequency to include in spectrogram
        event_ts : array_like
            Array of event times in seconds. If None, entire session is returned
        pre_t : float
            Time before event to include in signal (in seconds). Default is 0.
        post_t : float
            Time after event to include in signal (in seconds). Default is None, which
            returns a single time sample at event_times.

            
        Returns
        -------
        f : array_like
            Array of frequencies.
        rel_t : array_like
            Array of times relative to event.
        spec : array_like
            Array of spectrogram. Will have one additional dimension appended to the beginning,
            corresponding to the epoch number.
        """
        

        if chans is None: # if no channels specified, use all
            chans = np.arange(self._ch_num).flatten()
        else: # otherwise convert, or ensure, it is a numpy array
            chans = np.array(chans).flatten()

        if freq_min is None: # if no min frequency specified, use lowest
            freq_min_idx = 0
        else:
            freq_min_idx = np.where(self.spec_f >= freq_min)[0][0]


        if freq_max is None: # if no max frequency specified, use highest
            freq_max_idx = len(self.spec_f) 
        else:
            freq_max_idx = np.where(self.spec_f <= freq_max)[0][-1]

        if event_ts is None: # if no event times specified, use entire session
            event_idxs = int(0)
            pre_len = int(0)
            post_len = self.spec.shape[2]
        else:
            # get differences between event times and spectrogram times
            event_diffs = np.abs(event_ts.reshape(-1,1) - self.spec_t.reshape(1,-1))
            # get indices of event times in spectrogram
            event_idxs = np.argmin(event_diffs, axis=1)

        if post_t is None: # if no post time specified, use single time sample
            post_len = 1
        else:
            post_len = int(post_t*self.spec_fs)

        if pre_t is None: # if no pre time specified, start at event time
            pre_len = 0
        else:
            pre_len = int(pre_t*self.spec_fs)
        
        rel_idxs, spec = epoch_data(self.spec[chans, freq_min_idx:freq_max_idx, :], event_idxs=event_idxs,
                            pre_len=pre_len, post_len=post_len)

        f = self.spec_f[freq_min_idx:freq_max_idx]
        rel_t = rel_idxs/self.spec_fs

        return f, rel_t, spec

    
    def detect_flex_events(self, thresh=1.5, min_duration=500, max_spacing=500):
        """
        Detects events in a flex signal.
        
        Parameters
        ----------
        thresh : float
            The threshold to use for event detection.
        min_duration : int
            The minimum duration of an event (in samples).
        max_spacing : int
            The maximum spacing between events (in samples).
            
        Returns
        -------
        events : array_like
            Binary array.
        """

        # 1. get the z-scored flex signal
        z_sig = zscore(self.flex, axis=1)

        # 2. detect when signal crosses a threshold
        events = z_sig > thresh

        # 3. combine events spaced too closely together
        events = sn.binary_closing(events, structure=np.ones([1,max_spacing]))

        # 4. remove events shorter than min duration
        events = sn.binary_opening(events, structure=np.ones([1,min_duration]))
        
        # 5. for each flexion epoch, keep only the channel with the largest amplitude
        epochs = np.any(events, axis=0).astype(int)
        epochs = np.concatenate(([0], epochs, [0])) # pad with False
        onsets = np.where(np.diff(epochs) == 1)[0]+1
        offsets = np.where(np.diff(epochs) == -1)[0]+1

        for onset, offset in zip(onsets, offsets):
            max_chan = np.argmax(np.mean(z_sig[:,onset:offset], axis=1))
            events[:, onset:offset] = 0
            events[max_chan, onset:offset] = 1

        return events.astype(int) # convert to int for numpy operations

    def detect_flex_onsets(self, finger='thumb', **kwargs):
        """
        Detects the onsets of flex events.
        
        Parameters
        ----------
        finger : str
            Finger to detect onsets for. Must be one of 'thumb', 'index', 'middle', 'ring', 'little'.
        **kwargs : dict
            Keyword arguments to pass to detect_flex_events
            
        Returns
        -------
        onsets : array_like
            Array of flex event onsets (in seconds).
        """

        events = self.detect_flex_events(**kwargs)
        finger_ind = self.finger_lut[finger]
        onsets = np.where(np.diff(events[finger_ind,:]) == 1)[0] + 1
        return onsets/self.fs
    
    def plot_ecog_surf(self):
        """
        Plots the electrode locations on a brain.
        """

        view = plotting.view_markers(utils.tal2mni(self.locs),
                                     marker_labels=['%d'%k for k in np.arange(self.locs.shape[0])],
                                     marker_color='purple',
                                     marker_size=5)
        return view
    
    def plot_ecog_data(self, data, ax=None, **kwargs):
        """
        Plots the electrode locations on a brain.

        Parameters
        ----------
        data : array_like
            Array of data to plot. Must have one row per electrode.
        ax : matplotlib axis, optional
            Axis on which to plot
        **kwargs : optional
            Additional keyword arguments to pass to matplotlib plot function

        Returns
        -------
        ax : matplotlib axis
            Axis on which data was plotted
        """

        if ax is None:
            ax = plt.gca()
        ax.scatter(self.locs[:, 0], self.locs[:, 1], c=data, **kwargs)
        return ax
    


