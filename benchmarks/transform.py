import numpy as np
from empymod import model, transform, kernel, utils
from scipy.constants import mu_0       # Magn. permeability of free space [H/m]
from scipy.constants import epsilon_0  # Elec. permittivity of free space [F/m]


class Hankel:
    """Timing for empymod.transform functions related to Hankel transform.

    Timing checks for:

       - transform.fht
       - transform.hqwe
       - transform.hquad

    We check it for a small and a big example:

       - Small: 5 layers, 1 offset, 1 frequency, 1 wavenumber
       - Big: 5 layers, 100 offsets, 1 frequency, 201 wavenumbers

    In the small case Gamma has size 5, in the big example 100'500.
    Not check for many frequencies, as in the splined version this would have
    to be looped over it.

    """

    # Parameters to loop over
    params = [['Small', 'Big']]
    param_names = ['size', ]

    def setup(self, size):

        # One big, one small model
        if size == 'Small':  # Total size: 5*1*1*1 = 5
            off = np.array([500., 1000.])
        else:          # Total size: 5*100*1*201 = 100'500
            off = np.arange(1, 101)*200.

        # Define survey
        freq = np.array([1])
        lsrc = 1
        zsrc = 250.
        lrec = 1
        zrec = 300.
        angle = np.zeros(off.shape)
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
        xdirect = False
        verb = 0

        # Calculate eta, zeta
        etaH = 1/res + np.outer(2j*np.pi*freq, epermH*epsilon_0)
        etaV = 1/(res*aniso*aniso) + np.outer(2j*np.pi*freq, epermV*epsilon_0)
        zetaH = np.outer(2j*np.pi*freq, mpermH*mu_0)
        zetaV = np.outer(2j*np.pi*freq, mpermV*mu_0)

        # Collect input
        self.hankel = {'zsrc': zsrc, 'zrec': zrec, 'lsrc': lsrc, 'lrec': lrec,
                       'off': off, 'depth': depth, 'ab': ab, 'etaH': etaH,
                       'etaV': etaV, 'zetaH': zetaH, 'zetaV': zetaV, 'xdirect':
                       xdirect, 'msrc': msrc, 'mrec': mrec}

        # HT arguments
        _, fhtarg_st = utils.check_hankel('fht', ['key_201_2009', 0], verb)
        self.fhtarg_st = {'htarg': fhtarg_st}
        _, fhtarg_sp = utils.check_hankel('fht', ['key_201_2009', 10], verb)
        self.fhtarg_sp = {'htarg': fhtarg_sp}
        _, fhtarg_la = utils.check_hankel('fht', ['key_201_2009', -1], verb)
        self.fhtarg_la = {'htarg': fhtarg_la}

        # QWE: We lower the requirements here, otherwise it takes too long
        # ['rtol', 'atol', 'nquad', 'maxint', 'pts_per_dec', 'diff_quad', 'a',
        # 'b', 'limit']

        args_sp = [1e-6, 1e-10, 51, 100, 10, np.inf]
        args_st = [1e-6, 1e-10, 51, 100, 0, np.inf]
        _, qwearg_sp = utils.check_hankel('qwe', args_sp, verb)
        _, qwearg_st = utils.check_hankel('qwe', args_st, verb)
        self.qwearg_st = {'htarg': qwearg_st}
        self.qwearg_sp = {'htarg': qwearg_sp}

        # QUAD: We lower the requirements here, otherwise it takes too long
        # ['rtol', 'atol', 'limit', 'a', 'b', 'pts_per_dec']
        args = [1e-6, 1e-10, 100, 1e-6, 0.1, 10]
        _, quadargs = utils.check_hankel('quad', args, verb)
        self.quadargs = {'htarg': quadargs}

        self.hankel['ang_fact'] = kernel.angle_factor(angle, ab, msrc, mrec)

    def time_fht_standard(self, size):
        transform.hankel_dlf(**self.fhtarg_st, **self.hankel)

    def time_fht_lagged(self, size):
        transform.hankel_dlf(**self.fhtarg_la, **self.hankel)

    def time_fht_splined(self, size):
        transform.hankel_dlf(**self.fhtarg_sp, **self.hankel)

    def time_hqwe_standard(self, size):
        transform.hankel_qwe(**self.qwearg_st, **self.hankel)

    def time_hqwe_splined(self, size):
        transform.hankel_qwe(**self.qwearg_sp, **self.hankel)

    def time_hquad(self, size):
        transform.hankel_quad(**self.quadargs, **self.hankel)


class Dlf:
    """Timing for empymod.transform.dlf.

    We check it for a small and a big example:

       - Small: 5 layers, 1 offset, 1 frequency, 1 wavenumber
       - Big: 5 layers, 100 offsets, 1 frequency, 201 wavenumbers

    In the small case Gamma has size 5, in the big example 100'500.
    Not check for many frequencies, as in the splined version this would have
    to be looped over it.

    Also check for standard, lagged convolution and splined types.

    """
    # Parameters to loop over
    params = [['Small', 'Big'],
              ['Standard', 'Lagged', 'Splined']]
    param_names = ['size', 'htype']

    def setup_cache(self):
        """setup_cache is not parametrized, so we do it manually. """

        data = {}
        for size in self.params[0]:  # size

            data[size] = {}

            # One big, one small model
            if size == 'Small':  # Small; Total size: 5*1*1*1 = 5
                x = np.array([500., 1000.])
            else:       # Big; Total size: 5*100*100*201 = 10'050'000
                x = np.arange(1, 101)*200.

            # Define model parameters
            freq = np.array([1])
            src = [0, 0, 250]
            rec = [x, np.zeros(x.shape), 300]
            depth = np.array([-np.infty, 0, 300, 2000, 2100])
            res = np.array([2e14, .3, 1, 50, 1])
            ab = 11
            xdirect = False
            verb = 0

            # Checks (since DLF exists the `utils`-checks haven't changed, so
            # we just use them here.
            model = utils.check_model(depth, res, None, None, None, None, None,
                                      xdirect, verb)
            depth, res, aniso, epermH, epermV, mpermH, mpermV, _ = model
            frequency = utils.check_frequency(freq, res, aniso, epermH, epermV,
                                              mpermH, mpermV, verb)
            freq, etaH, etaV, zetaH, zetaV = frequency
            ab, msrc, mrec = utils.check_ab(ab, verb)
            src, nsrc = utils.check_dipole(src, 'src', verb)
            rec, nrec = utils.check_dipole(rec, 'rec', verb)
            off, angle = utils.get_off_ang(src, rec, nsrc, nrec, verb)
            lsrc, zsrc = utils.get_layer_nr(src, depth)
            lrec, zrec = utils.get_layer_nr(rec, depth)

            for htype in self.params[1]:  # htype

                # pts_per_dec depending on htype
                if htype == 'Standard':
                    pts_per_dec = 0
                elif htype == 'Lagged':
                    pts_per_dec = -1
                else:
                    pts_per_dec = 10

                # HT arguments
                _, fhtarg = utils.check_hankel(
                        'fht', ['key_201_2009', pts_per_dec], 0)

                # Calculate kernels for dlf
                lambd, _ = transform.get_dlf_points(
                        fhtarg['dlf'], off, fhtarg['pts_per_dec'])

                PJ = kernel.wavenumber(
                        zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH,
                        zetaV, lambd, ab, xdirect, msrc, mrec)

                factAng = kernel.angle_factor(angle, ab, msrc, mrec)

                dlf = {'signal': PJ, 'points': lambd, 'out_pts': off,
                       'ang_fact': factAng, 'ab': ab, 'filt': fhtarg['dlf'],
                       'pts_per_dec': fhtarg['pts_per_dec']}
                transform.dlf(**dlf)

                data[size][htype] = dlf

        return data

    def time_dlf(self, data, size, htype):
        transform.dlf(**data[size][htype])


class Fourier:
    """Timing for empymod.transform functions related to Fourier transform.

    Timing checks for:

       - transform.ffht
       - transform.fqwe
       - transform.fftlog
       - transform.fft

    We check it for a small and a big example:

       - Small: 5 layers, 1 offset, 1 time
       - Big: 5 layers, 1 offsets, 11 times

    """

    # Parameters to loop over
    params = [['Small', 'Big']]
    param_names = ['size', ]

    def setup_cache(self):
        """setup_cache is not parametrized, so we do it manually. """

        data = {}
        for size in self.params[0]:  # size
            tdat = {}

            # One big, one small model
            if size == 'Small':
                freqtime = np.array([2.])
            else:
                freqtime = np.logspace(-1, 1, 11)

            # Define survey
            lsrc = 1
            zsrc = 250.
            lrec = 1
            zrec = 300.
            angle = np.array([0])
            off = np.array([5000])
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
            verb = 0
            loop_freq = True
            loop_off = False
            signal = 0

            ht, htarg = utils.check_hankel('dlf', ['', -1], verb)

            def get_args(freqtime, ft, ftarg):
                time, freq, ft, ftarg = utils.check_time(
                        freqtime, signal, ft, ftarg, verb)

                # Calculate eta, zeta
                etaH = 1/res + np.outer(2j*np.pi*freq, epermH*epsilon_0)
                etaV = 1/(res*aniso*aniso) + np.outer(2j*np.pi*freq,
                                                      epermV*epsilon_0)
                zetaH = np.outer(2j*np.pi*freq, mpermH*mu_0)
                zetaV = np.outer(2j*np.pi*freq, mpermV*mu_0)

                inp = (ab, off, angle, zsrc, zrec, lsrc, lrec, depth, freq,
                       etaH, etaV, zetaH, zetaV, False, False, ht, htarg,
                       msrc, mrec, loop_freq, loop_off)
                fEM = np.squeeze(model.fem(*inp)[0])

                return (fEM, time, freq, ftarg)

            tdat['ffht_st'] = get_args(
                    freqtime, 'dlf', ['key_201_CosSin_2012', 0])
            tdat['ffht_la'] = get_args(
                    freqtime, 'dlf', ['key_201_CosSin_2012', -1])
            tdat['ffht_sp'] = get_args(
                    freqtime, 'dlf', ['key_201_CosSin_2012', 10])

            tdat['qwe'] = get_args(freqtime, 'qwe', ['', '', '', '', 10])

            tdat['fftlog'] = get_args(freqtime, 'fftlog', None)

            tdat['fft'] = get_args(freqtime, 'fft', None)

            data[size] = tdat

        return data

    def time_ffht_lagged(self, data, size):
        transform.fourier_dlf(*data[size]['ffht_la'])

    def time_ffht_standard(self, data, size):
        transform.fourier_dlf(*data[size]['ffht_st'])

    def time_ffht_splined(self, data, size):
        transform.fourier_dlf(*data[size]['ffht_sp'])

    def time_fqwe(self, data, size):
        transform.fourier_qwe(*data[size]['qwe'])

    def time_fftlog(self, data, size):
        transform.fourier_fftlog(*data[size]['fftlog'])

    def time_fft(self, data, size):
        transform.fourier_fft(*data[size]['fft'])
