import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.ndimage as sn
from nilearn import plotting
from nimare import utils
from source.utils import zscore, plot_stacked_signals, epoch_data

# default finger map for ecog finger data
finger_lut = {'thumb': 0, 'index': 1, 'middle': 2, 'ring': 3, 'little': 4}

class EcogFingerData():
    def __init__(self, fpath, fs=1000):
        self.fpath = fpath
        self.fs = fs
        
        data = loadmat(fpath)
		
        self.ecog = data['data']
        self.regions = data['elec_regions']
        self.flex = data['flex']
        self.brain = data['brain']
        self.locs = data['locs']
        self.cue = data['cue']
		
    def detect_flex_events(self, thresh=2, min_duration=200, max_spacing=500):
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
        z_sig = zscore(self.flex, dim=0)

        # 2. detect when signal crosses a threshold
        events = z_sig > thresh

        # 3. remove events shorter than min duration
        events = sn.binary_opening(events, structure=np.ones([min_duration,1]))

        # 4. combine events spaced too closely together
        events = sn.binary_closing(events, structure=np.ones([max_spacing,1]))
        
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
            Array of flex event onsets (in samples).
        """

        events = self.detect_flex_events(**kwargs)
        finger_ind = finger_lut[finger]
        return np.where(np.diff(events[:,finger_ind]) == 1)[0] + 1
    
    def plot_ecog_surf(self):
        """
        Plots the electrode locations on a brain.
        """

        view = plotting.view_markers(utils.tal2mni(self.locs),
                                     marker_labels=['%d'%k for k in np.arange(self.locs.shape[0])],
                                     marker_color='purple',
                                     marker_size=5)
        return view
    
    def plot_data_on_grid(self, data, ax=None, **kwargs):
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


data
	int32, (610040, 63)
elec_regions
	uint8, (63, 1)
flex
	uint16, (610040, 5)
brain
	[('vert', 'O'), ('tri', 'O')], (1, 1)
locs
	float64, (63, 3)
cue
	uint8, (610040, 1)
# load data files
subj = 'cc'
ecog_path = ['.', 'data', subj, '{}_fingerflex.mat'.format(subj)]
ecog_file = op.join(*ecog_path) # use * to unpack the list
data = loadmat(ecog_file)

# recording parameters
fs = 1000
t = np.arange(data['data'].shape[0]) / fs
ch = np.arange(data['data'].shape[1])


# PLOT ELECTRODE LOCATIONS
# code from https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/main/projects/ECoG/load_ECoG_fingerflex.ipynb#scrollTo=tD75j3xqzMh9
plt.figure(figsize=(8, 8))
locs = data['locs']
view = plotting.view_markers(utils.tal2mni(locs),
                             marker_labels=['%d'%k for k in np.arange(locs.shape[0])],
                             marker_color='purple',
                             marker_size=5)
view

# plot locs colored by erp
def erp_grid(locs, erp, t, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.scatter(locs[:, 0], locs[:, 1], c=erp[t,:], cmap='coolwarm', s=100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return ax

plt.figure(figsize=(12, 4))
plot_stacked_signals(zflex, t)
plt.legend(['Thumb', 'Index', 'Middle', 'Ring', 'Little'])


