import numpy as np
from empymod import model, transform, kernel, utils
from scipy.constants import mu_0       # Magn. permeability of free space [H/m]
from scipy.constants import epsilon_0  # Elec. permittivity of free space [F/m]

VariableCatch = (LookupError, AttributeError, ValueError, TypeError, NameError)


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
            off = np.array([500.])
        else:          # Total size: 5*100*1*201 = 100'500
            off = np.arange(1, 101)*200.

        # Define survey
        freq = np.array([1])
        lsrc = 1
        zsrc = np.array([250.])
        lrec = 1
        zrec = np.array([300.])
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
        use_ne_eval = False
        xdirect = False
        verb = 0

        # Calculate eta, zeta
        etaH = 1/res + np.outer(2j*np.pi*freq, epermH*epsilon_0)
        etaV = 1/(res*aniso*aniso) + np.outer(2j*np.pi*freq, epermV*epsilon_0)
        zetaH = np.outer(2j*np.pi*freq, mpermH*mu_0)
        zetaV = np.outer(2j*np.pi*freq, mpermV*mu_0)

        # Collect input for kernel.greenfct()
        self.hankel = {'zsrc': zsrc, 'zrec': zrec, 'lsrc': lsrc, 'lrec': lrec,
                       'off': off, 'angle': angle, 'depth': depth, 'ab': ab,
                       'etaH': etaH, 'etaV': etaV, 'zetaH': zetaH, 'zetaV':
                       zetaV, 'xdirect': xdirect, 'msrc': msrc, 'mrec': mrec,
                       'use_ne_eval': use_ne_eval}

        # Before c73d6647; you had to give `ab` to `check_hankel`;
        # check_opt didn't exist then.
        try:
            opt = utils.check_opt(None, None, 'fht', ['', 0], verb)
            charg = (0, )
            if np.size(opt) == 4:
                new_version = False
            else:
                new_version = True
        except VariableCatch:
            new_version = False
            charg = (ab, verb)

        # From 9bed72b0 onwards, there is no `use_spline`; `ftarg` input
        # changed (29/04/2018; before v1.4.1).
        if new_version:
            ftarg = ['key_201_2009', -1]
        else:
            ftarg = ['key_201_2009', None]

        # HT arguments
        _, fhtarg_st = utils.check_hankel('fht', ['key_201_2009', 0], *charg)
        self.fhtarg_st = {'fhtarg': fhtarg_st}
        _, fhtarg_sp = utils.check_hankel('fht', ['key_201_2009', 10], *charg)
        self.fhtarg_sp = {'fhtarg': fhtarg_sp}
        _, fhtarg_la = utils.check_hankel('fht', ftarg, *charg)
        self.fhtarg_la = {'fhtarg': fhtarg_la}

        # QWE: We lower the requirements here, otherwise it takes too long
        # ['rtol', 'atol', 'nquad', 'maxint', 'pts_per_dec', 'diff_quad', 'a',
        # 'b', 'limit']
        args = [1e-6, 1e-10, 51, 100, 0]
        _, qwearg_st = utils.check_hankel('qwe', args, *charg)
        self.qwearg_st = {'qweargs': qwearg_st}
        args = [1e-6, 1e-10, 51, 100, 10]
        _, qwearg_sp = utils.check_hankel('qwe', args, *charg)
        self.qwearg_sp = {'qweargs': qwearg_sp}

        # QUAD: We lower the requirements here, otherwise it takes too long
        # ['rtol', 'atol', 'limit', 'a', 'b', 'pts_per_dec']
        args = [1e-6, 1e-10, 100, '', '', 10]
        try:  # QUAD wasn't included from the beginning on
            _, quadargs = utils.check_hankel('quad', args, *charg)
            self.quadargs = {'quadargs': quadargs}
        except VariableCatch:
            self.quadargs = {}

        if not new_version:
            self.fhtarg_la.update({'use_spline': True})
            self.fhtarg_sp.update({'use_spline': True})
            self.fhtarg_st.update({'use_spline': False})
            self.qwearg_sp.update({'use_spline': True})
            self.qwearg_st.update({'use_spline': False})
            self.quadargs.update({'use_spline': True})

    def time_fht_standard(self, size):
        transform.fht(**self.fhtarg_st, **self.hankel)

    def time_fht_lagged(self, size):
        transform.fht(**self.fhtarg_la, **self.hankel)

    def time_fht_splined(self, size):
        transform.fht(**self.fhtarg_sp, **self.hankel)

    def time_hqwe_standard(self, size):
        transform.hqwe(**self.qwearg_st, **self.hankel)

    def time_hqwe_splined(self, size):
        transform.hqwe(**self.qwearg_sp, **self.hankel)

    def time_hquad(self, size):
        transform.hquad(**self.quadargs, **self.hankel)


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

    def setup(self, size, htype):

        # One big, one small model
        if size == 'Small':  # Total size: 5*1*1*1 = 5
            x = np.array([500.])
        else:          # Total size: 5*100*100*201 = 10'050'000
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
        use_ne_eval = False

        # Checks (since DLF exists the `utils`-checks haven't changed, so we
        # just use them here.
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

        # pts_per_dec depending on htype
        if htype == 'Lagged':
            pts_per_dec = -1
        elif htype == 'Splined':
            pts_per_dec = 10
        else:
            pts_per_dec = 0

        # HT arguments
        _, fhtarg = utils.check_hankel('fht', ['key_201_2009', pts_per_dec], 0)

        # Calculate kernels for dlf
        lambd, _ = transform.get_spline_values(fhtarg[0], off, fhtarg[1])
        PJ = kernel.wavenumber(zsrc, zrec, lsrc, lrec, depth, etaH, etaV,
                               zetaH, zetaV, lambd, ab, xdirect, msrc, mrec,
                               use_ne_eval)
        factAng = kernel.angle_factor(angle, ab, msrc, mrec)

        # Signature changed at commit a15af07 (20/05/2018; before v1.6.2)
        try:
            dlf = {'signal': PJ, 'points': lambd, 'out_pts': off,
                   'filt': fhtarg[0], 'pts_per_dec': fhtarg[1],
                   'factAng': factAng, 'ab': ab}
            transform.dlf(**dlf)
        except VariableCatch:
            dlf = {'signal': PJ, 'points': lambd, 'out_pts': off,
                   'targ': fhtarg, 'factAng': factAng}

        self.dlf = dlf

    def time_dlf(self, size, htype):
        transform.dlf(**self.dlf)


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

    def setup(self, size):

        # One big, one small model
        if size == 'Small':
            freqtime = np.array([2.])
        else:
            freqtime = np.logspace(-1, 1, 11)

        # Define survey
        lsrc = 1
        zsrc = np.array([250.])
        lrec = 1
        zrec = np.array([300.])
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
        use_ne_eval = False
        loop_freq = True
        loop_off = False
        signal = 0

        # `pts_per_dec` changed at 9bed72b0 (29/04/2018; bef. v1.4.1)
        try:
            ht, htarg = utils.check_hankel('fht', ['', -1], verb)
        except VariableCatch:
            # `check_hankel`-signature changed at c73d6647
            try:
                ht, htarg = utils.check_hankel('fht', None, verb)
            except VariableCatch:
                ht, htarg = utils.check_hankel('fht', None, ab, verb)

        def get_args(freqtime, ft, ftarg):
            time, freq, ft, ftarg = utils.check_time(freqtime, signal, ft,
                                                     ftarg, verb)

            # Calculate eta, zeta
            etaH = 1/res + np.outer(2j*np.pi*freq, epermH*epsilon_0)
            etaV = 1/(res*aniso*aniso) + np.outer(2j*np.pi*freq,
                                                  epermV*epsilon_0)
            zetaH = np.outer(2j*np.pi*freq, mpermH*mu_0)
            zetaV = np.outer(2j*np.pi*freq, mpermV*mu_0)

            # `model.fem`-signature changed on 9bed72b0
            # (29/04/2018; bef. v1.4.1)
            inp = (ab, off, angle, zsrc, zrec, lsrc, lrec, depth, freq, etaH,
                   etaV, zetaH, zetaV, False, False, ht, htarg, use_ne_eval,
                   msrc, mrec, loop_freq, loop_off)
            try:
                out = model.fem(*inp)
            except VariableCatch:
                out = model.fem(*inp[:17], True, *inp[17:])

            # `model.fem` returned in the beginning only fEM;
            # then (fEM, kcount) and finally (fEM, kcount, conv).
            if isinstance(out, tuple):
                fEM = np.squeeze(out[0])
            else:
                fEM = np.squeeze(out)

            return (fEM, time, freq, ftarg)

        # ffht used to be fft until the introduction of fft
        try:
            getattr(transform, 'ffht')
            fft_and_ffht = True
            name_ffht = 'ffht'
        except VariableCatch:
            fft_and_ffht = False
            name_ffht = 'fft'
        self.ffht_calc = getattr(transform, name_ffht)

        # Check default pts_per_dec to see if new or old case
        try:
            test = utils.check_time(freqtime, signal, 'sin',
                                    ['key_201_CosSin_2012', 'test'], 0)
            old_case = test[3][1] is None
        except VariableCatch:
            old_case = True

        if old_case:
            self.ffht_st = ()  # Standard was not possible in old case
            self.ffht_la = get_args(freqtime, name_ffht, None)
        else:
            self.ffht_st = get_args(freqtime, name_ffht,
                                    ['key_201_CosSin_2012', 0])
            self.ffht_la = get_args(freqtime, name_ffht,
                                    ['key_201_CosSin_2012', -1])
        self.ffht_sp = get_args(freqtime, name_ffht,
                                ['key_201_CosSin_2012', 10])

        self.fqwe = get_args(freqtime, 'fqwe', ['', '', '', '', 10])

        self.fftlog = get_args(freqtime, 'fftlog', None)

        if fft_and_ffht:
            self.fft = get_args(freqtime, 'fft', None)
        else:
            self.fft = ()

    def time_ffht_lagged(self, size):
        self.ffht_calc(*self.ffht_la)

    def time_ffht_standard(self, size):
        self.ffht_calc(*self.ffht_st)

    def time_ffht_splined(self, size):
        self.ffht_calc(*self.ffht_sp)

    def time_fqwe(self, size):
        transform.fqwe(*self.fqwe)

    def time_fftlog(self, size):
        transform.fftlog(*self.fftlog)

    def time_fft(self, size):
        transform.fft(*self.fft)
