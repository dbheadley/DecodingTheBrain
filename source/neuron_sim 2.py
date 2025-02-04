import numpy as np
import pandas as pd
import scipy.signal as sig

# Here are somme realistic parameters for a neuron
mem_r = 25e3 # ohm*cm^2, taken from Egger et al 2020
cell_radius = 30e-4 # cm, 30 microns, this is larger than the cell body itself to account for membrane from dendrites
cell_area = 4*np.pi*cell_radius**2 # cm^2
g_leak = (1/mem_r) * cell_area # S/cm^2
e_rest = -0.065 # V, 65 mV
ion_props = pd.DataFrame({'Ion': ['Na', 'K', 'Cl', 'Ca'],
                          'Out_mM': [140, 3, 140, 1.5],
                          'In_mM': [7, 140, 7, 0.0001],
                          'Charge': [1, 1, -1, 2]})
ion_props.set_index('Ion', inplace=True)

# calculate Nernst equilibrium potential for an ion
def nernst(conco=3, conci=140, z=1, t=310):
    """Calculate Nernst potential for an ion.
    
    Parameters
    ----------
    conco : float
        Concentration of ion outside cell (mM). Default is for K+.
    conci : float
        Concentration of ion inside cell (mM). Default is for K+.
    z : int
        Charge of ion.  Default is for K+.
    t : float
        Temperature (Kelvin). Default is body temperature.
    
    Returns
    -------
    float
        Nernst potential for ion (mV).
    """
    r = 8.314 # J/mol/K, gas constant
    f = 96485 # C/mol, Faraday's constant
    return (r*t)/(z*f) * np.log(conco/conci) * 1000

# code for solving for leak current
def ionic_current(v=0, g=g_leak, e=e_rest):
    """Calculate ionic current.
    
    Parameters
    ----------
    v : float
        Membrane potential (V). Default is 0.
    g : float
        Conductance (S). Default is leak conductance.
    e : float
        Equilibrium potential (V). Default is resting potential.
    
    Returns
    -------
    float
        Ionic current (A).
    """
    return g * (v - e)

# Function for generating spike trains
def spk_train(t, spk_times=0.1):
    dt = np.median(np.diff(t))
    spk_times = np.array([spk_times]) # ensures spk_times is a numpy array
    spk = np.zeros(len(t))
    spk[np.round(spk_times / dt).astype(int)] = 1
    return spk


### CLASSES FOR ACTIVE, PASSIVE, AND SYNAPTIC NEURONS ###

# create passive neuron class
class PassiveNeuron:
    def __init__(self, v_rest=-65, c_m=1, r_m=25, radius=30):
        self._vrest = v_rest/1000 # mV
        self._cm = c_m * 1e-6 # uF/cm^2
        self._rm = r_m * 1e3 # kOhm*cm^2
        self._radius = radius * 1e-4 # cm

        self._area = 4 * np.pi * self._radius**2 # cm^2
        self._c = self._cm * self._area # F  
        self._gleak = 1 / (self._rm / self._area) # S

        self._vm = self._vrest
        self._im = 0
        self._add = 0 # holds additional current to deliver to the neuron
        self._dt = 0.0001 # sec

    def get_tau(self):
        return self._cm * self._rm
    
    def set_input(self, inp=0):
        self._add = inp
    
    def reset_state(self):
        self._vm = self._vrest
        self._im = 0
        self._add = 0 # holds additional current to deliver to the neuron
    
    def get_state(self):
        # return membrane potential in mV, membrane current in nA, and spike status
        return self._vm*1e3, self._im*1e9
    
    def get_t_vec(self, dur = 0.1, dt = 0.0001):
        return np.arange(0, dur, dt)
    
    def update(self):
        # solve for transmembrane currents
        self._im = (self._gleak * (self._vm - self._vrest)) + self._add
        
        # update membrane potential
        self._vm = self._vm + -(self._im / self._c) * self._dt
    
    def run(self, dur=0.1, dt=0.0001, inp=0):

        self.reset_state() # reset state
        
        # initialize arrays to store values
        t = self.get_t_vec(dur, dt) # time array
        v = np.zeros(len(t)) # voltage array
        i = np.zeros(len(t)) # current array
        self._dt = dt # set time step

        # if input is scalar, make it an array
        if isinstance(inp, (int, float)):
            inp = np.ones(len(t)) * inp
        
        # run simulation
        for ind, _ in enumerate(t):
            self.set_input(inp[ind]) # set input current
            self.update() # update membrane potential and current
            v[ind], i[ind] = self.get_state() # store values

        return v, i

# create active neuron class that inheriting from the passive neuron model
class ActiveNeuron(PassiveNeuron):
    def __init__(self, v_thresh=-50, ena=50, ek=-90, gna=8e-8, gk=4e-8, **kwargs):
        super().__init__(**kwargs) # allows us to pass arguments for the passive properties of the neuron
        self._vthresh = v_thresh / 1000 # mV
        self._ena = ena / 1000 # mV
        self._gna = gna # S
        self._ek = ek / 1000 # mV
        self._gk = gk # S
        self._spk_timer = 0
        self._spk = False
    
    def reset_state(self):
        super().reset_state()
        self._spk_timer = 0
        self._spk = False

    def gen_ap(self):
        # action potential mechanism
        if (self._vm > self._vthresh) & (self._spk_timer <= 0):
            self._spk_timer = 0.004 # start countdown timer for duration of action potential
            self._spk = True
        elif self._spk_timer > 0.003: # open up sodium conductance for first 1 ms
            self._add = self._add + self._gna * (self._vm - self._ena)
            self._spk = False
            self._spk_timer -= self._dt
        elif self._spk_timer > 0: # open up potassium conductance for next 3 ms
            self._add = self._add + self._gk * (self._vm - self._ek)
            self._spk = False
            self._spk_timer -= self._dt

    def run(self, dur=0.1, dt=0.0001, inp=0):

        self.reset_state() # reset state
        
        # initialize arrays to store values
        t = self.get_t_vec(dur, dt) # time array
        v = np.zeros(len(t)) # voltage array
        i = np.zeros(len(t)) # current array
        self._dt = dt # set time step

        # if input is scalar, make it an array
        if isinstance(inp, (int, float)):
            inp = np.ones(len(t)) * inp
        
        # run simulation
        for ind, _ in enumerate(t):
            self.set_input(inp[ind]) # set input current
            self.gen_ap() # generate action potential <-- NEW
            self.update() # update membrane potential and current
            v[ind], i[ind] = self.get_state() # store values

        return v, i

# create an active neuron with synapse class that inherits from the active neuron model
class SynapticNeuron(ActiveNeuron):
    def __init__(self, esyn=50, gsyn=5e-9, asyn=900, bsyn=500, tdur=3, **kwargs):
        super().__init__(**kwargs) # allows us to pass arguments for the passive properties of the neuron
        self._esyn = esyn / 1000 # mV
        self._gsyn = gsyn # S
        self._asyn = asyn 
        self._bsyn = bsyn
        self._tdur = tdur / 1000 # ms, duration of transmitter release
        self._syn_timer = 0 # timer for transmitter release
        self._r = 0 # fraction of open channels

    def reset_state(self):
        super().reset_state()
        self._syn_timer = 0 # timer for transmitter release
        self._r = 0 # fraction of open channels

    def gen_syn(self, prespk=False):
        # synaptic mechanism
        if prespk:
            self._syn_timer = self._tdur
        elif self._syn_timer > 0:
            self._syn_timer -= self._dt
        
        # update fraction of open channels
        self._r = self._r + self._dt * ((self._asyn * (self._syn_timer>0) * (1-self._r)) \
                                        - (self._bsyn * (self._r)))
        
        # add synaptic current
        self._add = self._add + self._gsyn * self._r * (self._vm - self._esyn)

    def run(self, dur=0.1, dt=0.0001, inp=0, prespk=False): # added prespk to drive synapse

        self.reset_state() # reset state
        
        # initialize arrays to store values
        t = self.get_t_vec(dur, dt) # time array
        v = np.zeros(len(t)) # voltage array
        i = np.zeros(len(t)) # current array
        self._dt = dt # set time step

        # if input is scalar, make it an array
        if isinstance(inp, (int, float)):
            inp = np.ones(len(t)) * inp

        if isinstance(prespk, (bool, int, float)):
            prespk = np.ones(len(t)) * prespk
        
        # run simulation
        for ind, _ in enumerate(t):
            self.set_input(inp[ind]) # set input current
            self.gen_syn(prespk[ind]) # generate synaptic response <-- NEW
            self.gen_ap() # generate action potential
            self.update() # update membrane potential and current
            v[ind], i[ind] = self.get_state() # store values

        return v, i
    


class ExtracellRec():
    # class for extracellular recording of a sheet of neurons using the SynapticNeuron class
    # user can set the radius of the sheet, the density of neurons, the mean time and standard deviation 
    # around when a synaptic input arrives, the level of a shared slow input noise, and an individual gaussian noise
    # level for each neuron, electrode distance from the sheet
    def __init__(self, radius=2, density=1000, mean_t=0.1, std_t=0.005, slow_noise=25, \
                 noise=200, extra_cond=0.3, **kwargs):
        self._radius = radius # cm
        self._density = density # neurons per cm^2
        self._mean_t = mean_t # sec
        self._std_t = std_t # sec
        self._slow_noise = slow_noise * 1e-12 # slow shared noise standard deviation in pA
        self._noise = noise * 1e-12 # individual neuron gaussian standard deviation in pA
        self._extra_cond = extra_cond # extracellular conductivity in S/m
        self._v = [] # voltage array
        self._i = [] # current array
        self._t = [] # time array

        # calculate number of neurons
        self._n_neurons = int(np.round(np.pi * self._radius**2 * self._density))

        # calculate positions of neurons, uniformly distributed in the circular sheet
        # place each neuron by setting a random angle and radius from the center
        self._neuron_pos = np.zeros((self._n_neurons, 2))
        for ind in range(self._n_neurons):
            curr_ang = np.random.rand() * 2 * np.pi
            curr_rad = np.sqrt(np.random.rand()) * self._radius
            self._neuron_pos[ind, 0] = np.cos(curr_ang) * curr_rad
            self._neuron_pos[ind, 1] = np.sin(curr_ang) * curr_rad
        # force one neuron to be at the center, so it can be easily picked up by the electrode
        self._neuron_pos[0, :] = 0

        # create neurons
        self._neurons = []
        for ind in range(self._n_neurons):
            self._neurons.append(SynapticNeuron(**kwargs))

    # method to calculate extracellular potential based on currents from each neuron
    def calc_extracell(self, h=1):
        dists = np.sqrt(np.sum(self._neuron_pos**2,axis=1)+h**2)
        return 1/(4*np.pi*self._extra_cond) * np.sum((self._i.T * 1e-9) / dists, axis=1)


        
    # create run function that will run all neurons for a given duration
    def run(self, dur=0.2, dt=0.0001, seed=47):
        # set random seed
        np.random.seed(seed)

        # initialize arrays to store values
        t = self._neurons[0].get_t_vec(dur, dt) # time array
        num_t = len(t)
        v = np.zeros((self._n_neurons, num_t)) # voltage array
        i = np.zeros((self._n_neurons, num_t)) # current array
        

        # create synaptic inputs
        in_prespk = []
        for ind in range(self._n_neurons):
            in_prespk.append(spk_train(t, np.random.normal(self._mean_t, self._std_t, 1)))
        
        # create slow noise input
        slow_noise = np.random.normal(0, 1, num_t)
        slow_noise = sig.detrend(np.cumsum(slow_noise))
        slow_noise = (slow_noise/(np.std(slow_noise))) * self._slow_noise

        # create individual noise inputs
        indiv_noise = np.random.normal(0, self._noise, (self._n_neurons, len(t)))

        # run simulation
        for ind, curr_nrn in enumerate(self._neurons):
            inp_sig = slow_noise + indiv_noise[ind, :]
            v[ind, :], i[ind, :] = curr_nrn.run(dur=dur, dt=dt, inp=inp_sig, prespk=in_prespk[ind])

        # save simulation results
        self._t = t
        self._v = v
        self._i = i