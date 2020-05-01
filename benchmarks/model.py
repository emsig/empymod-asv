import numpy as np
from empymod import model
from copy import deepcopy as dc

VariableCatch = (LookupError, AttributeError, ValueError, TypeError, NameError)

try:
    from empymod.transform import hankel_dlf  # noqa
    VERSION2 = True
    HTARG101 = {'dlf': 'key_101_2009'}
    HTARG201 = {'dlf': 'key_201_2009'}
    HTARG401 = {'dlf': 'key_401_2009'}
except ImportError:
    VERSION2 = False
    HTARG101 = 'key_101_2009'
    HTARG201 = 'key_201_2009'
    HTARG401 = 'key_401_2009'


class Bipole:
    """Timing for empymod.model.bipole.

    The heavy computation is all checked in the classes Kernel and Transform.
    Here we just check ``bipole`` with its checks and most importantly its
    looping. No big models, just to assure it doesn't slow down for some
    reason.

    """
    def time_frequency(self):
        model.bipole(
                src=[[-50, 0], [0, 30],
                     [0, 10], [5, 50],
                     [970, 999], [980, 990]],
                rec=[[2000, 3000], [2100, 3200],
                     [0, 200], [100, 400],
                     [960, 970], [950, 1000]],
                depth=[0, 1000, 2000, 2100],
                res=[2e14, 0.3, 1, 100, 1],
                freqtime=[0.1, 1.0, 10.],
                srcpts=5, recpts=5, strength=1000,
                htarg=HTARG201,
                xdirect=False, verb=0)

    def time_time(self):
        model.bipole(
                src=[0, 0, 950, 10, 30],
                rec=[3000, 100, 1000, 5, 7],
                depth=[0, 1000, 2000, 2100],
                res=[2e14, 0.3, 1, 100, 1],
                freqtime=[0.1, 1.0, 10.],
                htarg=HTARG201,
                srcpts=1, recpts=1, strength=0,
                signal=-1, xdirect=False, verb=0)


class Dipole:
    """Timing for empymod.model.dipole.

    The heavy computation is all checked in the classes Kernel and Transform.
    Here we just check ``dipole`` with its checks and looping. No big models,
    just to assure it doesn't slow down for some reason.

    """

    # Parameters to loop over
    params = [[None, 'freq', 'off'], ]
    param_names = ['loop', ]

    def setup(self, loop):
        self.model = {
                'src': [0, 0, 990],
                'depth': [0, 1000, 2000, 2100],
                'res': [2e14, 0.3, 1, 100, 1],
                'xdirect': False,
                'htarg': HTARG201,
                'loop': loop,
                'verb': 0}
        if not VERSION2:
            self.model['opt'] = None

        self.freqtime = np.logspace(-2, 2, 21)
        self.fmodel = dc(self.model)
        self.tmodel = dc(self.model)

        # Till c73d6647 (btw. v1.0.0 and v1.1.0) there were the routines
        # `frequency` and `time`, which were later merged into `dipole`.
        try:
            # Test
            model.dipole([0, 0, 1], [10, 0, 2], [], 1, 1, verb=0)
            # Frequency
            self.freq = model.dipole
            self.fmodel['freqtime'] = self.freqtime
            # Time
            self.time = model.dipole
            self.tmodel['freqtime'] = self.freqtime
        except VariableCatch:
            # Frequency
            self.freq = model.frequency
            self.fmodel['freq'] = self.freqtime
            # Time
            self.time = model.time
            self.tmodel['time'] = self.freqtime

    def time_frequency(self, loop):
        self.freq(rec=[np.arange(1, 21)*300, np.zeros(20), 1000],
                  **self.fmodel)

    def time_time(self, loop):
        self.time(rec=[np.arange(1, 4)*1000, np.zeros(3), 1000],
                  signal=0, **self.tmodel)


class VariousDipole:
    """Timing for empymod.model.dipole.

    Check a few other cases (ab, angle, freq-range, src/rec-layer).

    """

    def setup(self):

        # Till c73d6647 (btw. v1.0.0 and v1.1.0) there were the routines
        # `frequency` and `time`, which were later merged into `dipole`.
        try:
            model.dipole([0, 0, 1], [10, 0, 2], [], 1, 1, verb=0)
            self.func = model.dipole
        except VariableCatch:
            self.func = model.frequency

    def time_marine_angle_12(self):
        # First arguments without name, for backwards comp. with `frequency`
        self.func([0, 0, 990],             # src
                  [np.arange(1, 11)*600, np.arange(1, 11)*400, 1000],  # rec
                  [0, 1000],               # Depths
                  [2e14, 0.3, 1],          # Resistivities
                  np.logspace(-2, 2, 11),  # Frequencies
                  ab=12, xdirect=False, htarg=HTARG101, verb=0)

    def time_land_angle_16(self):
        # First arguments without name, for backwards comp. with `frequency`
        self.func([0, 0, 1e-5],            # src
                  [np.arange(1, 11)*600, np.arange(1, 11)*400, 1e-5],  # rec
                  0,                       # Depth
                  [2e14, 10],              # Resistivities
                  np.logspace(-2, 2, 11),  # Frequencies
                  ab=16, epermH=[0, 1], epermV=[0, 1], xdirect=False,
                  htarg=HTARG101, verb=0)

    def time_difflsrclrec_42(self):
        # First arguments without name, for backwards comp. with `frequency`
        self.func([0, 0, -20],             # src
                  [np.arange(1, 11)*600, np.zeros(10), 100],  # rec
                  [0, 50],                 # Depth
                  [2e14, 10, 1],           # Resistivities
                  np.logspace(-2, 2, 11),  # Frequencies
                  aniso=[1, 2, 0.5], ab=42, xdirect=False,
                  htarg=HTARG101, verb=0)

    def time_highfreq_11(self):
        # First arguments without name, for backwards comp. with `frequency`
        self.func([0, 0, 2],              # src
                  [np.arange(1, 11), np.arange(1, 11)/4, 3],  # rec
                  [0, 10],                # Depth
                  [2e14, 10, 100],        # Resistivities
                  np.logspace(6, 8, 11),  # Frequencies
                  aniso=[1, 2, 0.5], ab=11, epermH=[1, 80, 5],
                  epermV=[1, 40, 10], mpermH=[1, 1, 4], mpermV=[1, 2, 0.5],
                  xdirect=False, htarg=HTARG401, verb=0)


class Analytical:
    """Timing for empymod.model.analytical."""

    # Parameters to loop over
    params = [['fs', 'dhs', 'dfs'], ]
    param_names = ['solution', ]

    def setup(self, solution):
        if solution == 'dfs':
            signal = 0
        else:
            signal = None

        self.hsfs_inp = {
                'src': [0, 0, 0],
                'rec': [np.arange(1, 101)*200, np.zeros(100), 0],
                'res': 3.5,
                'freqtime': np.logspace(-2, 2, 101),
                'signal': signal,
                'solution': solution,
                'verb': 0}

    def time_analytical(self, solution):
        model.analytical(**self.hsfs_inp)
