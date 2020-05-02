import numpy as np
from empymod import model, transform, kernel, utils
from scipy.constants import mu_0       # Magn. permeability of free space [H/m]
from scipy.constants import epsilon_0  # Elec. permittivity of free space [F/m]

VariableCatch = (LookupError, AttributeError, ValueError, TypeError, NameError)

try:
    from empymod.transform import hankel_dlf  # noqa
    VERSION2 = True
except ImportError:
    VERSION2 = False


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
        lrec = 1
        angle = np.zeros(off.shape)
        ab = 11
        msrc = False
        mrec = False

        if VERSION2:
            zsrc = 250.
            zrec = 300.
        else:
            zsrc = np.array([250.])  # Not sure if this distinction
            zrec = np.array([300.])  # is actually needed
            use_ne_eval = False

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

        # Compute eta, zeta
        etaH = 1/res + np.outer(2j*np.pi*freq, epermH*epsilon_0)
        etaV = 1/(res*aniso*aniso) + np.outer(2j*np.pi*freq, epermV*epsilon_0)
        zetaH = np.outer(2j*np.pi*freq, mpermH*mu_0)
        zetaV = np.outer(2j*np.pi*freq, mpermV*mu_0)

        # Collect input
        self.hankel = {'zsrc': zsrc, 'zrec': zrec, 'lsrc': lsrc, 'lrec': lrec,
                       'off': off, 'depth': depth, 'ab': ab, 'etaH': etaH,
                       'etaV': etaV, 'zetaH': zetaH, 'zetaV': zetaV, 'xdirect':
                       xdirect, 'msrc': msrc, 'mrec': mrec}
        if not VERSION2:
            self.hankel['use_ne_eval'] = use_ne_eval

        # Before c73d6647; you had to give `ab` to `check_hankel`;
        # check_opt didn't exist then.
        if VERSION2:
            charg = (verb, )
            new_version = True
        else:
            try:
                opt = utils.check_opt(None, None, 'fht', ['', 0], verb)
                charg = (verb, )
                if np.size(opt) == 4:
                    new_version = False
                else:
                    new_version = True
            except VariableCatch:
                new_version = False
                charg = (ab, verb)

        # From 9bed72b0 onwards, there is no `use_spline`; `htarg` input
        # changed (29/04/2018; before v1.4.1).
        if new_version:
            if VERSION2:
                htarg = {'dlf': 'key_201_2009', 'pts_per_dec': -1}
            else:
                htarg = ['key_201_2009', -1]
        else:
            htarg = ['key_201_2009', None]

        # HT arguments
        if VERSION2:
            dlfargname = 'htarg'
            qweargname = 'htarg'
            quadargname = 'htarg'
            htarg1 = {'dlf': 'key_201_2009', 'pts_per_dec': 0}
            htarg2 = {'dlf': 'key_201_2009', 'pts_per_dec': 10}
            name = 'dlf'
        else:
            dlfargname = 'fhtarg'
            qweargname = 'qweargs'
            quadargname = 'quadargs'
            htarg1 = ['key_201_2009', 0]
            htarg2 = ['key_201_2009', 10]
            name = 'fht'

        _, fhtarg_st = utils.check_hankel(name, htarg1, *charg)
        self.fhtarg_st = {dlfargname: fhtarg_st}
        _, fhtarg_sp = utils.check_hankel(name, htarg2, *charg)
        self.fhtarg_sp = {dlfargname: fhtarg_sp}
        _, fhtarg_la = utils.check_hankel(name, htarg, *charg)
        self.fhtarg_la = {dlfargname: fhtarg_la}

        # QWE: We lower the requirements here, otherwise it takes too long
        # ['rtol', 'atol', 'nquad', 'maxint', 'pts_per_dec', 'diff_quad', 'a',
        # 'b', 'limit']

        # Args depend if QUAD included into QWE or not
        try:
            if VERSION2:
                args_sp = {'atol': 1e-6, 'rtol': 1e-10, 'nquad': 51,
                           'maxint': 100, 'pts_per_dec': 10,
                           'diff_quad': np.inf}
                args_st = {'atol': 1e-6, 'rtol': 1e-10, 'nquad': 51,
                           'maxint': 100, 'pts_per_dec': 0,
                           'diff_quad': np.inf}
            else:
                args_sp = [1e-6, 1e-10, 51, 100, 10, np.inf]
                args_st = [1e-6, 1e-10, 51, 100, 0, np.inf]
            _, qwearg_sp = utils.check_hankel('qwe', args_sp, *charg)
            _, qwearg_st = utils.check_hankel('qwe', args_st, *charg)
        except VariableCatch:
            args_sp = [1e-6, 1e-10, 51, 100, 10]
            args_st = [1e-6, 1e-10, 51, 100, 0]
            _, qwearg_sp = utils.check_hankel('qwe', args_sp, *charg)
            _, qwearg_st = utils.check_hankel('qwe', args_st, *charg)

        self.qwearg_st = {qweargname: qwearg_st}
        self.qwearg_sp = {qweargname: qwearg_sp}

        # QUAD: We lower the requirements here, otherwise it takes too long
        # ['rtol', 'atol', 'limit', 'a', 'b', 'pts_per_dec']
        if VERSION2:
            args = {'atol': 1e-6, 'rtol': 1e-10, 'limit': 100, 'a': 1e-6,
                    'b': 0.1, 'pts_per_dec': 10}
        else:
            args = [1e-6, 1e-10, 100, 1e-6, 0.1, 10]
        try:  # QUAD only included since 6104614e (before v1.3.0)
            _, quadargs = utils.check_hankel('quad', args, *charg)
            self.quadargs = {quadargname: quadargs}
        except VariableCatch:
            self.quadargs = {}

        if not new_version and not VERSION2:
            self.fhtarg_la.update({'use_spline': True})
            self.fhtarg_sp.update({'use_spline': True})
            self.fhtarg_st.update({'use_spline': False})
            self.qwearg_sp.update({'use_spline': True})
            self.qwearg_st.update({'use_spline': False})
            self.quadargs.update({'use_spline': True})

        if VERSION2:
            self.hankel['ang_fact'] = kernel.angle_factor(
                    angle, ab, msrc, mrec)
        else:
            # From bb6447a onwards ht-transforms take `factAng`, not `angle`,
            # to avoid re-calculation in loops.
            try:
                transform.fht(angle=angle, **self.fhtarg_la, **self.hankel)
                self.hankel['angle'] = angle
            except VariableCatch:
                self.hankel['factAng'] = kernel.angle_factor(
                        angle, ab, msrc, mrec)

        if not VERSION2:
            # From b6f6872 onwards fht-transforms calculates lambd/int_pts in
            # model.fem, not in transform.fht, to avoid re-calculation in
            # loops.
            try:
                transform.fht(**self.fhtarg_la, **self.hankel)
            except VariableCatch:
                lambd, int_pts = transform.get_spline_values(
                        fhtarg_st[0], off, fhtarg_st[1])
                self.fhtarg_st.update({'fhtarg': (
                    fhtarg_st[0], fhtarg_st[1], lambd, int_pts)})
                lambd, int_pts = transform.get_spline_values(
                        fhtarg_la[0], off, fhtarg_la[1])
                self.fhtarg_la.update(
                        {'fhtarg':
                         (fhtarg_la[0], fhtarg_la[1], lambd, int_pts)})
                lambd, int_pts = transform.get_spline_values(
                        fhtarg_sp[0], off, fhtarg_sp[1])
                self.fhtarg_sp.update(
                        {'fhtarg':
                         (fhtarg_sp[0], fhtarg_sp[1], lambd, int_pts)})

    def time_fht_standard(self, size):
        if VERSION2:
            transform.hankel_dlf(**self.fhtarg_st, **self.hankel)
        else:
            transform.fht(**self.fhtarg_st, **self.hankel)

    def time_fht_lagged(self, size):
        if VERSION2:
            transform.hankel_dlf(**self.fhtarg_la, **self.hankel)
        else:
            transform.fht(**self.fhtarg_la, **self.hankel)

    def time_fht_splined(self, size):
        if VERSION2:
            transform.hankel_dlf(**self.fhtarg_sp, **self.hankel)
        else:
            transform.fht(**self.fhtarg_sp, **self.hankel)

    def time_hqwe_standard(self, size):
        if VERSION2:
            transform.hankel_qwe(**self.qwearg_st, **self.hankel)
        else:
            transform.hqwe(**self.qwearg_st, **self.hankel)

    def time_hqwe_splined(self, size):
        if VERSION2:
            transform.hankel_qwe(**self.qwearg_sp, **self.hankel)
        else:
            transform.hqwe(**self.qwearg_sp, **self.hankel)

    def time_hquad(self, size):
        if VERSION2:
            transform.hankel_quad(**self.quadargs, **self.hankel)
        else:
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

            if not VERSION2:
                use_ne_eval = False

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

                # Compute kernels for dlf
                if VERSION2:
                    # HT arguments
                    _, fhtarg = utils.check_hankel(
                            'dlf',
                            {'dlf': 'key_201_2009',
                             'pts_per_dec': pts_per_dec},
                            0)

                    inp = (fhtarg['dlf'], off, fhtarg['pts_per_dec'])
                    lambd, _ = transform.get_dlf_points(*inp)
                else:
                    # HT arguments
                    _, fhtarg = utils.check_hankel(
                            'fht', ['key_201_2009', pts_per_dec], 0)

                    inp = (fhtarg[0], off, fhtarg[1])
                    lambd, _ = transform.get_spline_values(*inp)

                if VERSION2:
                    inp = (zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH,
                           zetaV, lambd, ab, xdirect, msrc, mrec)
                else:
                    inp = (zsrc, zrec, lsrc, lrec, depth, etaH,
                           etaV, zetaH, zetaV, lambd, ab, xdirect,
                           msrc, mrec, use_ne_eval)
                PJ = kernel.wavenumber(*inp)

                factAng = kernel.angle_factor(angle, ab, msrc, mrec)

                # Signature changed at commit a15af07 (20/05/2018; before
                # v1.6.2)
                try:
                    dlf = {'signal': PJ, 'points': lambd, 'out_pts': off,
                           'ab': ab}
                    if VERSION2:
                        dlf['ang_fact'] = factAng
                        dlf['filt'] = fhtarg['dlf']
                        dlf['pts_per_dec'] = fhtarg['pts_per_dec']
                    else:
                        dlf['factAng'] = factAng
                        dlf['filt'] = fhtarg[0]
                        dlf['pts_per_dec'] = fhtarg[1]
                    transform.dlf(**dlf)
                except VariableCatch:
                    dlf = {'signal': PJ, 'points': lambd, 'out_pts': off,
                           'targ': fhtarg, 'factAng': factAng}

                data[size][htype] = dlf

        return data

    def time_dlf(self, data, size, htype):
        transform.dlf(**data[size][htype])


class Fourier:
    """Timing for empymod.transform functions related to Fourier transform.

    Timing checks for:

       - transform.fourier_dlf
       - transform.fourier_qwe
       - transform.fourier_fftlog
       - transform.fourier_fft

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
            lrec = 1
            angle = np.array([0])
            off = np.array([5000])
            ab = 11
            msrc = False
            mrec = False

            if VERSION2:
                zsrc = 250.
                zrec = 300.
            else:
                zsrc = np.array([250.])  # Not sure if this distinction
                zrec = np.array([300.])  # is actually needed
                use_ne_eval = False

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

            # Get Hankel arguments
            if VERSION2:
                ht, htarg = utils.check_hankel(
                        'dlf', {'pts_per_dec': -1}, verb)
            else:
                # `pts_per_dec` changed at 9bed72b0 (29/04/2018; bef. v1.4.1)
                try:
                    ht, htarg = utils.check_hankel('fht', ['', -1], verb)
                except VariableCatch:
                    # `check_hankel`-signature changed at c73d6647
                    try:
                        ht, htarg = utils.check_hankel('fht', None, verb)
                    except VariableCatch:
                        ht, htarg = utils.check_hankel('fht', None, ab, verb)

            # Get frequency-domain stuff for time-domain computation
            def get_args(freqtime, ft, ftarg):
                time, freq, ft, ftarg = utils.check_time(
                        freqtime, signal, ft, ftarg, verb)

                # Compute eta, zeta
                etaH = 1/res + np.outer(2j*np.pi*freq, epermH*epsilon_0)
                etaV = 1/(res*aniso*aniso) + np.outer(2j*np.pi*freq,
                                                      epermV*epsilon_0)
                zetaH = np.outer(2j*np.pi*freq, mpermH*mu_0)
                zetaV = np.outer(2j*np.pi*freq, mpermV*mu_0)

                # `model.fem`-signature changed on 9bed72b0
                # (29/04/2018; bef. v1.4.1)
                inp = (ab, off, angle, zsrc, zrec, lsrc, lrec, depth, freq,
                       etaH, etaV, zetaH, zetaV, False, False, ht, htarg,
                       msrc, mrec, loop_freq, loop_off)
                try:
                    if not VERSION2:
                        inp = (*inp[:17], use_ne_eval, *inp[17:])
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

            # Define function name of transform
            fft_and_ffht = True
            if VERSION2:
                name_dlf = 'fourier_dlf'
                name_fqwe = 'fourier_qwe'
                name_fftlog = 'fourier_fftlog'
                name_fft = 'fourier_fft'
            else:
                name_fqwe = 'fqwe'
                name_fftlog = 'fftlog'
                name_fft = 'fft'
                # ffht used to be fft until the introduction of fft
                try:
                    getattr(transform, 'ffht')
                    name_ffht = 'ffht'
                except VariableCatch:
                    fft_and_ffht = False
                    name_ffht = 'fft'
                name_dlf = name_ffht

            # Store functions
            tdat['fourier_dlf'] = getattr(transform, name_dlf)
            tdat['fourier_qwe'] = getattr(transform, name_fqwe)
            tdat['fourier_fftlog'] = getattr(transform, name_fftlog)
            tdat['fourier_fft'] = getattr(transform, name_fft)

            # Check default pts_per_dec to see if new or old case
            if VERSION2:
                old_case = False
            else:
                try:
                    test = utils.check_time(freqtime, signal, 'sin',
                                            ['key_201_CosSin_2012', 'test'], 0)
                    old_case = test[3][1] is None
                except VariableCatch:
                    old_case = True

            # Get fourier_dlf arguments
            if old_case and not VERSION2:
                tdat['dlf_st'] = ()  # Standard was not possible in old case
                tdat['dlf_la'] = get_args(freqtime, name_ffht, None)
            elif VERSION2:
                tdat['dlf_st'] = get_args(
                        freqtime, 'dlf',
                        {'dlf': 'key_201_CosSin_2012', 'pts_per_dec': 0})
                tdat['dlf_la'] = get_args(
                        freqtime, 'dlf',
                        {'dlf': 'key_201_CosSin_2012', 'pts_per_dec': -1})
            else:
                tdat['dlf_st'] = get_args(
                        freqtime, name_ffht, ['key_201_CosSin_2012', 0])
                tdat['dlf_la'] = get_args(
                        freqtime, name_ffht, ['key_201_CosSin_2012', -1])

            if VERSION2:
                tdat['dlf_sp'] = get_args(
                        freqtime, 'dlf',
                        {'dlf': 'key_201_CosSin_2012', 'pts_per_dec': 10})

                # Get fourier_qwe arguments
                tdat['qwe'] = get_args(freqtime, 'qwe', {'pts_per_dec': 10})

                # Get fourier_fftlog arguments
                tdat['fftlog'] = get_args(freqtime, 'fftlog', {})

                # Get fourier_fft arguments
                tdat['fft'] = get_args(freqtime, 'fft', {})

            else:
                tdat['dlf_sp'] = get_args(
                        freqtime, name_ffht, ['key_201_CosSin_2012', 10])

                # Get fourier_qwe arguments
                tdat['qwe'] = get_args(freqtime, 'fqwe', ['', '', '', '', 10])

                # Get fourier_fftlog arguments
                tdat['fftlog'] = get_args(freqtime, 'fftlog', None)

                # Get fourier_fft arguments
                if fft_and_ffht:
                    tdat['fft'] = get_args(freqtime, 'fft', None)
                else:
                    tdat['fft'] = ()  # Will fail

            data[size] = tdat

        return data

    def time_dlf_lagged(self, data, size):
        data[size]['fourier_dlf'](*data[size]['dlf_la'])

    def time_dlf_standard(self, data, size):
        data[size]['fourier_dlf'](*data[size]['dlf_st'])

    def time_dlf_splined(self, data, size):
        data[size]['fourier_dlf'](*data[size]['dlf_sp'])

    def time_qwe(self, data, size):
        data[size]['fourier_qwe'](*data[size]['qwe'])

    def time_fftlog(self, data, size):
        data[size]['fourier_fftlog'](*data[size]['fftlog'])

    def time_fft(self, data, size):
        data[size]['fourier_fft'](*data[size]['fft'])
