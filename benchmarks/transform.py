import numpy as np
from empymod import model, transform, kernel, utils


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
            x = np.array([500.])
        else:          # Total size: 5*100*1*201 = 100'500
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

        # Checks
        try:  # From f1cfe201 onwards (28/04/2018; before v1.4.1)
            model = utils.check_model(depth, res, None, None, None, None, None,
                                      xdirect, verb)
        except VariableCatch:  # Till f1cfe201
            model = utils.check_model(depth, res, None, None, None, None, None,
                                      verb)
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

        # Other params
        use_ne_eval = False

        # Collect input for kernel.greenfct()
        self.hankel = {'zsrc': zsrc, 'zrec': zrec, 'lsrc': lsrc, 'lrec': lrec,
                       'off': off, 'angle': angle, 'depth': depth, 'ab': ab,
                       'etaH': etaH, 'etaV': etaV, 'zetaH': zetaH, 'zetaV':
                       zetaV, 'xdirect': xdirect, 'msrc': msrc, 'mrec': mrec,
                       'use_ne_eval': use_ne_eval}

        # Check if new or old version
        # (from 9bed72b0 onwards; 29/04/2018; before v1.4.1)
        opt = utils.check_opt(None, None, 'fht', ['', 0], verb)
        if np.size(opt) == 4:
            new_version = False
        else:
            new_version = True

        # HT arguments
        _, fhtarg_st = utils.check_hankel('fht', ['key_201_2009', 0], 0)
        self.fhtarg_st = {'fhtarg': fhtarg_st}
        _, fhtarg_sp = utils.check_hankel('fht', ['key_201_2009', 10], 0)
        self.fhtarg_sp = {'fhtarg': fhtarg_sp}
        if new_version:
            _, fhtarg_la = utils.check_hankel('fht', ['key_201_2009', -1], 0)
            self.fhtarg_la = {'fhtarg': fhtarg_la}
        else:
            _, fhtarg_la = utils.check_hankel('fht', ['key_201_2009', 0], 0)
            self.fhtarg_la = {'use_spline': True, 'fhtarg': fhtarg_la}

        # QWE: We lower the requirements here, otherwise it takes too long
        args = {'pts_per_dec': 0, 'rtol': 1e-6, 'atol': 1e-10}
        _, qwearg_st = utils.check_hankel('qwe', args, 0)
        self.qwearg_st = {'qweargs': qwearg_st}
        args = {'pts_per_dec': 10, 'rtol': 1e-6, 'atol': 1e-10}
        _, qwearg_sp = utils.check_hankel('qwe', args, 0)
        self.qwearg_sp = {'qweargs': qwearg_sp}

        # QUAD: We lower the requirements here, otherwise it takes too long
        args = {'pts_per_dec': 10, 'rtol': 1e-6, 'atol': 1e-10, 'limit': 100}
        try:  # QUAD wasn't included from the beginning on
            _, quadargs = utils.check_hankel('quad', args, 0)
            self.quadargs = {'quadargs': quadargs}
        except VariableCatch:
            self.quadargs = {}

        if not new_version:
            self.fhtarg_st.update({'use_spline': False})
            self.fhtarg_sp.update({'use_spline': True})
            self.qwearg_st.update({'use_spline': False})
            self.qwearg_sp.update({'use_spline': True})
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

        # Checks
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

        # Other params
        use_ne_eval = False

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

        try:  # From a15af07 onwards (20/05/2018; before v1.6.2)
            dlf = {'signal': PJ, 'points': lambd, 'out_pts': off,
                   'filt': fhtarg[0], 'pts_per_dec': fhtarg[1],
                   'factAng': factAng, 'ab': ab}
            transform.dlf(**dlf)
        except VariableCatch:  # Till a15af07
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

        src = [0, 0, 250]
        rec = [5000, 0, 300]
        depth = np.array([-np.infty, 0, 300, 2000, 2100])
        res = np.array([2e14, .3, 1, 50, 1])

        signal = 0
        verb = 1

        try:  # From f1cfe201 onwards (28/04/2018; before v1.4.1)
            cmodel = utils.check_model(depth, res, None, None, None, None,
                                       None, False, verb)
        except VariableCatch:  # Till f1cfe201
            cmodel = utils.check_model(depth, res, None, None, None, None,
                                       None, verb)
        depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace = cmodel

        try:  # From 9bed72b0 onwards (29/04/2018; before v1.4.1)
            ht, htarg = utils.check_hankel('fht', {'pts_per_dec': -1}, verb)
            optimization = utils.check_opt(None, None, ht, htarg, verb)
            use_ne_eval, loop_freq, loop_off = optimization
        except VariableCatch:  # Till 9bed72b0
            ht, htarg = utils.check_hankel('fht', None, verb)
            optimization = utils.check_opt('spline', None, ht, htarg, verb)
            use_spline, use_ne_eval, loop_freq, loop_off = optimization

        ab, msrc, mrec = utils.check_ab(11, verb)
        src, nsrc = utils.check_dipole(src, 'src', verb)
        rec, nrec = utils.check_dipole(rec, 'rec', verb)
        off, angle = utils.get_off_ang(src, rec, nsrc, nrec, verb)
        lsrc, zsrc = utils.get_layer_nr(src, depth)
        lrec, zrec = utils.get_layer_nr(rec, depth)

        def get_args(freqtime, ft, ftarg):
            time, freq, ft, ftarg = utils.check_time(freqtime, signal, ft,
                                                     ftarg, verb)
            frequency = utils.check_frequency(freq, res, aniso, epermH, epermV,
                                              mpermH, mpermV, verb)
            freq, etaH, etaV, zetaH, zetaV = frequency

            try:  # From 9bed72b0 onwards (29/04/2018; before v1.4.1)
                EM, _, _ = model.fem(ab, off, angle, zsrc, zrec, lsrc, lrec,
                                     depth, freq, etaH, etaV, zetaH, zetaV,
                                     False, isfullspace, ht, htarg,
                                     use_ne_eval, msrc, mrec, loop_freq,
                                     loop_off)
            except VariableCatch:  # Till 9bed72b0
                EM, _, _ = model.fem(ab, off, angle, zsrc, zrec, lsrc, lrec,
                                     depth, freq, etaH, etaV, zetaH, zetaV,
                                     False, isfullspace, ht, htarg, True,
                                     use_ne_eval, msrc, mrec, loop_freq,
                                     loop_off)

            return (np.squeeze(EM), time, freq, ftarg)

        # ffht used to be fft until the introduction of fft
        try:
            getattr(transform, 'ffht')
            fft_and_ffht = True
            name_ffht = 'ffht'
        except VariableCatch:
            fft_and_ffht = False
            name_ffht = 'fft'
        self.ffht_calc = getattr(transform, name_ffht)

        self.ffht_st = get_args(freqtime, name_ffht, {'pts_per_dec': 0})
        self.ffht_la = get_args(freqtime, name_ffht, {'pts_per_dec': -1})
        self.ffht_sp = get_args(freqtime, name_ffht, {'pts_per_dec': 10})

        self.fqwe = get_args(freqtime, 'fqwe', {'pts_per_dec': 10})

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
