import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import spectrogram, convolve, butter, filtfilt, iirnotch, find_peaks
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
            chans = np.arange(self._ch_num).flatten()
        else:
            chans = np.array(chans).flatten()
        
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

        if post_t is None: # if no post time specified, use single time sample
            post_len = 1
        else:
            post_len = int(post_t*self.spec_fs)

        if pre_t is None: # if no pre time specified, start at event time
            pre_len = 0
        else:
            pre_len = int(pre_t*self.spec_fs)
            
        if event_ts is None: # if no event times specified, use entire session
            event_idxs = int(0)
            pre_len = int(0)
            post_len = self.spec.shape[2]
        else:
            # get differences between event times and spectrogram times
            event_diffs = np.abs(event_ts.reshape(-1,1) - self.spec_t.reshape(1,-1))
            # get indices of event times in spectrogram
            event_idxs = np.argmin(event_diffs, axis=1)
        
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
    


class EEG:
    def __init__(self, eeg_file, chan_file):
        """
        A class for loading and plotting EEG data

        Parameters
        ----------
        eeg_file : str, path to the .set file
        chan_file : str, path to the _channels.tsv file
        """

        # load the eeg data
        self._eeg = loadmat(eeg_file, appendmat=False)
        data = self._eeg['data']
        self.srate = self._eeg['srate'][0,0]
        self.data = data
        self.nchan = self.data.shape[0]
        self.nsamp = self.data.shape[1]
        self.dur = self.nsamp/self.srate

        # load the channel info and integrate with locations
        chan_info = self._eeg['chaninfo'][0,0][1]
        chan_names = [name.split()[0] for name in chan_info]
        chan_locs = np.array([name.split()[2:] for name in chan_info], dtype=float)
        chan_info = pd.DataFrame({'name': chan_names, 
                                    'ml': chan_locs[:,0], 
                                    'ap': chan_locs[:,1], 
                                    'dv': chan_locs[:,2]})
        chans = pd.read_csv(chan_file, sep='\t')
        chans = pd.merge(chans, chan_info, how='left', on='name')
        chans.index.name = 'idx'
        self.chans = chans

        # get reference electrode position
        ref_elec = self._eeg['ref'][0]
        near_ref_names = [ref for ref in ref_elec.split('_') if ref in chans['name'].tolist()]
        near_ref_chans = chans[chans['name'].isin(near_ref_names)]
        near_ref_coords = near_ref_chans[['ml', 'ap', 'dv']].to_numpy()
        self.ref_coord = np.mean(near_ref_coords, axis=0)

    def get_data(self, chans=None, start_t=0, dur_t=None, scale='absolute'):
        """
        Extract EEG data from the EEG object

        Parameters
        ----------
        chans : list of str, the channels to extract
        start_t : numeric array, the start times in seconds
        dur_t : float, the duration in seconds
        scale : str, 'absolute' or 'relative'

        Returns
        -------
        data_epochs : 3d array, the eeg data
        tpts : 1d array, the time vector
        chans : list of str, the channels extracted
        """

        # ensure proper formatting of inputs
        if not chans:
            chans = self.chans['name']
        elif chans == 'eeg' or chans == ['eeg']: # only extract eeg channels
            chans = self.chans[self.chans['type']=='EEG']['name'].values
        elif type(chans) == str:
            chans = [chans]
        
        if not dur_t:
            dur_t = self.dur-start_t
        start_t = np.array(start_t)
        start_t = start_t.ravel()
        epoch_num = start_t.size

        # convert times to indices
        start_idxs = (start_t*self.srate).astype(int)
        dur_idx = (dur_t*self.srate).astype(int)
        end_idxs = start_idxs + dur_idx

        # get the channel indices
        chan_idxs = [np.where(self.chans['name']==sel_ch)[0][0] for sel_ch in chans]

        # extract the eeg data, one epoch at a time
        data_epochs = np.zeros((dur_idx, len(chan_idxs), epoch_num)) # this also ensures changes to the data don't affect the original
        for i in range(epoch_num):
            data_epochs[:,:,i] = self.data[chan_idxs, start_idxs[i]:end_idxs[i]].T

        # get the time vector
        if scale == 'absolute':
            tpts = start_t + np.arange(0, dur_idx)/self.srate
        elif scale == 'relative':
            tpts = np.arange(0, dur_idx)/self.srate
        
        return data_epochs, tpts, chans

    def plot_scalp(self, ax=None, colors='b'):
        """
        Plot the channel locations on the scalp
        """

        # ensure proper formatting of inputs
        if not ax:
            fig, ax = plt.subplots()
        
        chans = self.chans[self.chans['type']=='EEG']

        # plot the channel locations
        ax.scatter(chans['ml'], chans['ap'], c=colors)
        for ind, name in enumerate(chans['name']):
            ax.text(chans['ml'][ind], chans['ap'][ind], name)
        
        # plot the reference electrode
        ax.scatter(self.ref_coord[0], self.ref_coord[1], c='k', s=100)
        
        ax.set_xlabel('Medial-lateral')
        ax.set_ylabel('Anterior-posterior')
        ax.set_aspect('equal')

        return ax
    
## Preprocessing code for EEG data
# a function to remove baseline drift
def remove_baseline_drift(eeg, w=8):
    # eeg: an EEG object
    # w: window size in seconds
    # returns: an EEG object with baseline drift removed

    # convert window size to number of samples
    w = int((w/2) * eeg.srate)

    # create convolution kernel
    kernel = np.ones((1, 2*w+1)) / (2*w+1)

    # determine which channels are EEG
    eeg_chans = eeg.chans['type'] == 'EEG'

    # pad data with edge values <-- HERE IS A CHANGE
    data_pad = np.pad(eeg.data[eeg_chans,:], ((0,0), (w,w)), 'edge')

    # convolve kernel with EEG data using scipy.signal.convolve
    # (mode='valid' to keep output the same size as the input after padding) <-- HERE IS A CHANGE
    baseline = convolve(data_pad, kernel, mode='valid')

    # subtract baseline from EEG data
    eeg.data[eeg_chans,:] = eeg.data[eeg_chans,:] - baseline

# function to remove EMG artifacts from EEG data
def remove_emg(eeg, cut_freq=60):
    # eeg: EEG data
    # cut_freq: cutoff frequency
    
    # create a bandpass filter
    b, a = butter(4, cut_freq, 'low', fs=eeg.srate)
    
    # determine which channels are EEG
    eeg_chans = eeg.chans['type'] == 'EEG'

    # apply the filter to the data
    eeg.data[eeg_chans,:] = filtfilt(b, a, eeg.data[eeg_chans,:], axis=1)

# function to remove AC noise from EEG data
def remove_ac(eeg, ac_freq=60): 
    # eeg: EEG data
    # ac_freq: frequency of the AC noise (default: 60 Hz because we're in the US)
    
    # create a bandpass filter
    b, a = iirnotch(ac_freq, 15, fs=eeg.srate)
    
    # determine which channels are EEG
    eeg_chans = eeg.chans['type'] == 'EEG'

    # apply the filter to the data
    eeg.data[eeg_chans,:] = filtfilt(b, a, eeg.data[eeg_chans,:], axis=1)

def detect_blinks(eeg, eog_chan='HEO', threshold=40, ibi=0.5):
    """
    Detect blinks in the EEG signal.

    Parameters
    ----------
    eeg : our EEG object
    eog_chan : str
        The name of the EOG channel. Defaults to 'HEO'.
    threshold : float
        The threshold for detecting blinks. Defaults to 40.
    ibi : float
        The minimum inter-blink interval (in seconds). Defaults to 0.5.

    Returns
    -------
    blink_times : array of floats
        The times of the blinks in seconds.
    blink_heights : array of floats
        The height of the blinks.   
    """
    # Format inputs
    ibi = ibi * eeg.srate

    # Find the EOG channel
    eog_data, eog_t, _ = eeg.get_data(chans=eog_chan)

    # Format EOG data for peak finding
    eog_data = -eog_data.squeeze()

    # Filter drifting baseline and EMG out of the EOG data
    b, a = butter(2, [1, 50], 'bandpass', fs=eeg.srate)
    eog_data = filtfilt(b, a, eog_data)

    # Find the blinks
    blink_times, blink_props = find_peaks(eog_data, height=threshold, distance=ibi)

    # Convert blink_times to seconds
    blink_times = eog_t[blink_times]

    # Get the blink heights
    blink_heights = -blink_props['peak_heights']

    return blink_times, blink_heights