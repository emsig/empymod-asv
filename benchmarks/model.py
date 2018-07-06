import numpy as np
import empymod as epm


class Dipole:
    """
    Timing for empymod.model.dipole.
    """
    params = [[None, 'parallel', 'spline'],
              [1, 10, 100],
              [1, 100]]
    param_names = ['opt', 'noff', 'nfreq']

    def setup(self, opt, noff, nfreq):
        # Parameters over which to loop
        nl = 5

        depth = np.r_[0, 1000+np.linspace(0, 5000, nl-2)]
        res = np.r_[2e14, 0.3, np.linspace(.1, 10, nl-2)]
        rec = [np.linspace(500, min(20000, noff*500+500), noff),
               np.zeros(noff), 1000]
        freq = np.logspace(0.01, 10, nfreq)

        self.model = {'src': [0, 0, 990], 'rec': rec, 'depth': depth, 'res':
                      res, 'freqtime': freq, 'ab': 11, 'htarg': 'key_201_2009',
                      'xdirect': False, 'opt': opt, 'verb': 0}

    def time_dipole(self, opt, noff, nfreq):
        epm.dipole(**self.model)
