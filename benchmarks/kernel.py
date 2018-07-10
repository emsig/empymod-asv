import numpy as np
from empymod import kernel, filters
from scipy.constants import mu_0       # Magn. permeability of free space [H/m]
from scipy.constants import epsilon_0  # Elec. permittivity of free space [F/m]


class Core:
    """Timing for empymod.kernel functions.

    Timing checks for:

       - kernel.wavenumber
       - kernel.greenfct
       - kernel.reflections
       - kernel.fields

    Really the core of empymod. We check it for a small and a big example:

       - Small: 5 layers, 1 offset, 1 frequency, 1 wavenumber
       - Big: 5 layers, 100 offsets, 100 frequencies, 201 wavenumbers

    In the small case Gamma has size 5, in the big example 10'050'000.

    We don't check the other functions here. ``angle_factor`` is a rather small
    function, and ``fullspace`` and ``halfspace`` we check with
    ``model.analytical``.

    """

    # Parameters to loop over
    params = [['Small', 'Big'], ]
    param_names = ['size']

    def setup(self, size):

        # One big, one small model
        if size == 'Small':  # Total size: 5*1*1*1 = 5
            freq = np.array([1])
            off = np.array([500.])
            base = np.array([1])
        else:          # Total size: 5*100*100*201 = 10'050'000
            freq = np.logspace(-2, 2, 100)
            off = np.arange(1, 101)*200.
            base = filters.key_201_2009().base

        # Define survey
        lsrc = 1
        zsrc = np.array([250.])
        lrec = 1
        zrec = np.array([300.])
        ab = 11
        msrc = False
        mrec = False

        # Define model
        depth = np.array([-np.infty, 0, 300, 2000, 2100])
        res = np.array([2e14, .3, 1, 50, 1])
        aniso = np.ones(res.shape)
        epermH = np.ones(res.shape)
        epermV = np.ones(res.shape)
        mpermH = np.ones(res.shape)
        mpermV = np.ones(res.shape)

        # Other parameters
        use_ne_eval = False
        xdirect = False
        TM = True

        # Calculate eta, zeta, wavenumber, Gamma
        etaH = 1/res + np.outer(2j*np.pi*freq, epermH*epsilon_0)
        etaV = 1/(res*aniso*aniso) + np.outer(2j*np.pi*freq, epermV*epsilon_0)
        zetaH = np.outer(2j*np.pi*freq, mpermH*mu_0)
        zetaV = np.outer(2j*np.pi*freq, mpermV*mu_0)
        lambd = base/off[:, None]
        Gam = np.sqrt((etaH/etaV)[:, None, :, None] *
                      (lambd*lambd)[None, :, None, :] +
                      (zetaH*etaH)[:, None, :, None])

        # Collect input for kernel.greenfct()
        self.green_wave = {'zsrc': zsrc, 'zrec': zrec, 'lsrc': lsrc,
                           'lrec': lrec, 'depth': depth, 'etaH': etaH,
                           'etaV': etaV, 'zetaH': zetaH, 'zetaV': zetaV,
                           'lambd': lambd, 'ab': ab, 'xdirect': xdirect,
                           'msrc': msrc, 'mrec': mrec,
                           'use_ne_eval': use_ne_eval}

        # Collect input for kernel.reflections()
        self.reflection = {'depth': depth, 'e_zH': etaH, 'Gam': Gam, 'lrec':
                           lrec, 'lsrc': lsrc, 'use_ne_eval': use_ne_eval}

        # Calculate plus/minus reflection coefficients
        Rp, Rm = kernel.reflections(**self.reflection)

        # Collect input for kernel.fields()
        self.fields = {'depth': depth, 'Gam': Gam, 'lrec': lrec, 'lsrc': lsrc,
                       'Rp': Rp, 'Rm': Rm, 'zsrc': zsrc, 'ab': ab, 'TM': TM,
                       'use_ne_eval': use_ne_eval}

    # kernel.wavenumber()
    def time_wavenumber(self, size):
        kernel.wavenumber(**self.green_wave)

    # kernel.greenfct()
    def time_greenfct(self, size):
        kernel.greenfct(**self.green_wave)

    # kernel.reflections()
    def time_reflections(self, size):
        kernel.reflections(**self.reflection)

    # kernel.fields()
    def time_fields(self, size):
        kernel.fields(**self.fields)
