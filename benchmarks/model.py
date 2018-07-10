import numpy as np
from empymod import model


class Bipole:
    """Timing for empymod.model.bipole.

    The heavy calculation is all checked in the classes Kernel and Transform.
    Here we just check ``bipole`` with its checks and most importantly its
    looping. No big models, just to assure it doesn't slow down for some
    reason.

    """

    def time_bipole_freq(self):
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
                htarg='key_201_2009',
                xdirect=False, verb=3)

    def time_bipole_time(self):
        model.bipole(
                src=[0, 0, 950, 10, 30],
                rec=[3000, 100, 1000, 5, 7],
                depth=[0, 1000, 2000, 2100],
                res=[2e14, 0.3, 1, 100, 1],
                freqtime=[0.1, 1.0, 10.],
                htarg='key_201_2009',
                srcpts=1, recpts=1, strength=0,
                signal=-1, xdirect=False, verb=3)


class Dipole:
    """Timing for empymod.model.dipole.

    The heavy calculation is all checked in the classes Kernel and Transform.
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
                'freqtime': np.logspace(-2, 2, 21),
                'xdirect': False,
                'htarg': 'key_201_2009',
                'opt': None,
                'loop': loop,
                'verb': 0}

    def time_dipole_freq(self, loop):
        model.dipole(rec=[np.arange(1, 21)*300, np.zeros(20), 1000],
                     **self.model)

    def time_dipole_time(self, loop):
        model.dipole(rec=[3000, 0, 1000], signal=0, **self.model)


class DipoleVariousCases:
    """Timing for empymod.model.dipole.

    Check a few other cases (ab, angle, freq-range, src/rec-layer).

    """

    def time_dipole_marine_angle_12(self):
        model.dipole(src=[0, 0, 990],
                     rec=[np.arange(1, 11)*600, np.arange(1, 11)*400, 1000],
                     depth=[0, 1000],
                     res=[2e14, 0.3, 1],
                     freqtime=np.logspace(-2, 2, 11),
                     ab=12,
                     xdirect=False,
                     htarg='key_101_2009',
                     verb=0)

    def time_dipole_land_angle_16(self):
        model.dipole(src=[0, 0, 1e-5],
                     rec=[np.arange(1, 11)*600, np.arange(1, 11)*400, 1e-5],
                     depth=0,
                     res=[2e14, 10],
                     freqtime=np.logspace(-2, 2, 11),
                     ab=16,
                     epermH=[0, 1],
                     epermV=[0, 1],
                     xdirect=False,
                     htarg='key_101_2009',
                     verb=0)

    def time_dipole_difflsrclrec_42(self):
        model.dipole(src=[0, 0, -20],
                     rec=[np.arange(1, 11)*600, np.zeros(10), 100],
                     depth=[0, 50],
                     res=[2e14, 10, 1],
                     aniso=[1, 2, 0.5],
                     freqtime=np.logspace(-2, 2, 11),
                     ab=42,
                     xdirect=False,
                     htarg='key_101_2009',
                     verb=0)

    def time_dipole_highfreq_11(self):
        model.dipole(src=[0, 0, 2],
                     rec=[np.arange(1, 11), np.arange(1, 11)/4, 3],
                     depth=[0, 10],
                     res=[2e14, 10, 100],
                     aniso=[1, 2, 0.5],
                     freqtime=np.logspace(6, 8, 11),
                     ab=11,
                     epermH=[1, 80, 5],
                     epermV=[1, 40, 10],
                     mpermH=[1, 1, 4],
                     mpermV=[1, 2, 0.5],
                     xdirect=False,
                     htarg='key_401_2012',
                     verb=0)


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
