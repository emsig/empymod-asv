# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import numpy as np
import empymod as epm


class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        # Parameters over which to loop
        nf = 1
        no = 100
        nl = 5

        depth = np.r_[0, 1000+np.linspace(0, 5000, nl-2)]
        res = np.r_[2e14, 0.3, np.linspace(.1, 10, nl-2)]
        rec = [np.linspace(500, min(20000, no*500+500), no),
               np.zeros(no), 1000]
        freq = np.array([1]) #np.linspace(0.01, 10, nf)

        self.model = {'src': [0, 0, 990], 'rec': rec, 'depth': depth, 'res':
                      res, 'freqtime': freq, 'ab': 11, 'htarg': 'key_201_2009',
                      'xdirect': False, 'verb': 2}

        # Printed in the asv `dev version`
        # epm.versions()

    def time_None(self):
        epm.dipole(opt=None, **self.model)

    def time_parallel(self):
        epm.dipole(opt='parallel', **self.model)

    def time_spline(self):
        epm.dipole(opt='spline', **self.model)
