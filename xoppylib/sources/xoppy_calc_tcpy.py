#
# xoppy_calc_tcpy: pure-python reimplementation of the TC code
#                  (on-axis brilliance tuning curves of an ideal undulator)
#
# Reproduces the method and the results of TC v1.97 by R.J. Dejus (APS, ANL),
# the binary distributed with xoppylib,
# as driven by xoppylib.xoppy_run_binaries.xoppy_calc_xtc, without calling
# the external binary. Implemented method: "infinite-N with convolution"
# (Dejus' method, METHOD=1 of XTC), for planar (HELICAL=0) and helical
# (HELICAL=1) devices.
#
# Physics summary
# ---------------
# 1) The angular spectral flux of harmonic i of an ideal undulator is computed
#    in the Bessel function approximation: at observation angle (theta, phi),
#    with alpha = gamma*theta, a = 1 + (Kx^2+Ky^2)/2 + alpha^2, the (complex)
#    field amplitudes are expansions in products of Bessel functions
#    Jn(y)*J_{i+2n+m}(x) (m = -1, 0, +1) -- see _s0_harmonic() below.
# 2) In the infinite-N limit, photons of energy E at harmonic i are emitted
#    on the cone alpha_i^2 = i*ER/E - (1 + K^2/2), ER = 2*gamma^2*(hc/lambda0).
#    The convolution of the angular flux density with the Gaussian angular
#    distribution of the electron beam, evaluated on axis, reduces to a 1D
#    integral over the azimuth phi along that cone.
# 3) The finite number of periods is restored by convolving the spectrum
#    with the sinc^2 line shape of width DEW = E1/N.
# 4) The result is divided by the effective source area
#    2*pi*sqrt((sigx^2+sigr^2)*(sigy^2+sigr^2)), where sigr is the
#    detuning-dependent effective source size (Walker's parametrization),
#    giving the on-axis brilliance in ph/s/mrad^2/mm^2/0.1%bw.
# 5) The beam energy spread is applied as a Gaussian convolution (rms width
#    2*sige*E) and, for each K of the scan, the peak of the resulting curve
#    near harmonic i is stored: this builds the tuning curve.
#
import numpy
import scipy

# ----------------------------------------------------------------------
# physical constants (updated from values as in tc.f/usb.f, Physics Today Aug. 1990)
# ----------------------------------------------------------------------
_C_LIGHT = scipy.constants.c  # 2.99792458e8        # speed of light [m/s]
_ME = scipy.constants.m_e  # 9.1093897e-31       # electron rest mass [kg]
_EC = scipy.constants.e  # 1.60217733e-19      # elementary charge [C]
_MEE = _ME * _C_LIGHT ** 2 / _EC * 1e-6  # 0.51099906          # electron rest mass [MeV]
_HBAR = scipy.constants.hbar  # 1.05457266e-34      # hbar [Js]
_H = scipy.constants.h  # 6.6260755e-34                          # Planck constant [Js]
_EPSZ = scipy.constants.epsilon_0  # 8.854187817e-12     # permittivity of vacuum [F/m]
_PI = numpy.pi
_BW = 1.0e-3  # 0.1% bandwidth

_FSC     = _EC * _EC / (4.0 * _PI * _EPSZ * _HBAR * _C_LIGHT)  # fine structure
_C_EVANG = _H * _C_LIGHT / _EC * 1e10                          # 12398.42
_PTOT_FAC = _PI / 3.0 * _EC / _EPSZ / _MEE**2 * 1e6            # 0.07257
_PD_FAC   = 21.0 / (16.0 * _PI * _MEE**2) * _PTOT_FAC          # 0.11611
_CV       = 1.0 / (8.0 * _PI * _PI) * 1e-10 * 1e6              # 1.2665e-6

# numerical parameters (as in usb.f/tc.f)
_NPHI    = 20      # azimuthal points in (0, pi/2)
_NSIGMA  = 4       # angular integration extent of the e-beam Gaussian
_NOMEGA  = 64      # base number of points per 2*EW for the sinc^2 convolution
_COMEGA  = 8.0     # half-extent of the sinc^2 line shape in units of DEW
_EPSH    = 1.0e-2  # relative cutoff for the sum over harmonics
_EPSE    = 1.0e-8  # bracketing distance at harmonic edges [eV]
_SPECLIM = 1.0e10  # minimum brilliance to retain
_IB      = 3       # peak validity margin (points)
_CUTOFF  = 1.0e-3  # harmonic intensity cutoff rel. to previous harmonic
                   # (TC v1.97; TC v1.95 used 5.0e-2)
_NSMALL  = 2       # consecutive small harmonics before truncating the
                   # harmonic sum (TC v1.97; v1.95 stopped after one)


def _bessjn_all(mmax, x):
    """
    J_m(x) for all integer orders m = 0..mmax, vectorized over x (x >= 0).
    Miller's downward recurrence with renormalization (the same algorithm as
    bright_bessjn in brighte.f / Numerical Recipes bessjn), evaluated for the
    whole x array at once. Returns an array of shape (mmax+1,) + x.shape.
    """
    x  = numpy.asarray(x, dtype=float)
    xf = x.ravel()
    out = numpy.zeros((mmax + 1, xf.size))
    tiny = xf < 1e-12
    xs = numpy.where(tiny, 1.0, xf)
    m0 = int(max(mmax, numpy.ceil(xs.max()))) + 2
    m0 = m0 + int(numpy.sqrt(40.0 * m0))
    if m0 % 2:
        m0 += 1                                   # even starting order
    bjp  = numpy.zeros_like(xs)
    bj   = numpy.full_like(xs, 1e-30)
    bsum = numpy.zeros_like(xs)
    for m in range(m0, 0, -1):
        bjm = (2.0 * m / xs) * bj - bjp           # J_{m-1} (unnormalized)
        bjp, bj = bj, bjm
        big = numpy.abs(bj) > 1e10                # rescale to avoid overflow
        if big.any():
            bj[big] *= 1e-10
            bjp[big] *= 1e-10
            bsum[big] *= 1e-10
            out[:, big] *= 1e-10
        if m - 1 <= mmax:
            out[m - 1] = bj
        if (m - 1) % 2 == 0 and (m - 1) > 0:      # accumulate even orders
            bsum += bj
    out /= (2.0 * bsum + out[0])                  # J0 + 2*sum_k J_{2k} = 1
    out[:, tiny] = 0.0
    out[0, tiny] = 1.0
    return out.reshape((mmax + 1,) + x.shape)


def _jn_signed(table, m):
    """J_m from a table over orders 0..mmax (first axis), for integer m of
    any sign, using J_{-m} = (-1)^m J_m. Returns orders on the last axis."""
    m = numpy.asarray(m)
    am = numpy.abs(m)
    sgn = numpy.where((m < 0) & (am % 2 == 1), -1.0, 1.0)
    return sgn * numpy.moveaxis(table[am], 0, -1)


def _s0_harmonic(i, kx, ky, alpha, cosphi, sinphi):
    """
    |A|^2 angular function of harmonic i of an ideal planar/helical/elliptical
    undulator (Bessel function approximation). Equivalent to s0 of BRIGHTE
    (bright1/bright3 in brighte.f), in fully vectorized complex form:

      S_m   = sum_n Jn(y) * J_{i+2n+m}(x) * exp(-i(i+2n+m)*phi'),  m=-1,0,+1
      Ax    = (i/a) * (2*alpha*cos(phi)*S_0 - ky*(S_+1 + S_-1))
      Ay    = (i/a) * (2*alpha*sin(phi)*S_0 - 1j*kx*(S_+1 - S_-1))
      s0    = |Ax|^2 + |Ay|^2

    with a = 1+(kx^2+ky^2)/2+alpha^2, x = (2i/a)*alpha*sqrt((kx sinphi)^2 +
    (ky cosphi)^2), y = (i/4a)*(ky^2-kx^2), phi' = atan2(kx sinphi, ky cosphi).

    alpha: array (...,), cosphi/sinphi: arrays broadcastable to alpha.
    """
    alpha2 = alpha * alpha
    a   = 1.0 + 0.5 * (kx * kx + ky * ky) + alpha2
    x   = (2.0 * i / a) * alpha * numpy.sqrt((kx * sinphi)**2 + (ky * cosphi)**2)
    y   = (0.25 * i / a) * (ky * ky - kx * kx)

    nlim = max(int(6.2 + 1.41 * float(numpy.max(numpy.abs(y)))) + 2, 3)
    n    = numpy.arange(-nlim, nlim + 1)                  # (nn,)
    ordr = i + 2 * n
    Jy   = _jn_signed(_bessjn_all(nlim, numpy.abs(y)), n) # (..., nn)
    if ky * ky - kx * kx < 0.0:                           # J_n(-|y|)
        Jy = Jy * numpy.where(numpy.abs(n) % 2 == 1, -1.0, 1.0)
    xtab = _bessjn_all(i + 2 * nlim + 1, x)
    Jx0  = _jn_signed(xtab, ordr)
    Jxp  = _jn_signed(xtab, ordr + 1)
    Jxm  = _jn_signed(xtab, ordr - 1)

    if kx == 0.0:                                         # planar: all real
        S0 = numpy.sum(Jy * Jx0, axis=-1)
        Ss = numpy.sum(Jy * (Jxp + Jxm), axis=-1)
        Ax = (i / a) * (2.0 * alpha * cosphi * S0 - ky * Ss)
        Ay = (i / a) * (2.0 * alpha * sinphi * S0)
        return Ax * Ax + Ay * Ay

    phi = numpy.arctan2(kx * sinphi, ky * cosphi)
    ph  = numpy.exp(-1j * ordr * phi[..., None])
    S0  = numpy.sum(Jy * Jx0 * ph, axis=-1)
    Sp  = numpy.sum(Jy * Jxp * ph * numpy.exp(-1j * phi[..., None]), axis=-1)
    Sm  = numpy.sum(Jy * Jxm * ph * numpy.exp(+1j * phi[..., None]), axis=-1)
    Ax = (i / a) * (2.0 * alpha * cosphi * S0 - ky * (Sp + Sm))
    Ay = (i / a) * (2.0 * alpha * sinphi * S0 - 1j * kx * (Sp - Sm))
    return (Ax * Ax.conj() + Ay * Ay.conj()).real


def _usb(energy, cur, sigx, sigy, sigx1, sigy1, period, n, kx, ky,
         eminu, emaxu, nek):
    """
    On-axis brilliance spectrum of an ideal undulator including the electron
    beam emittance: equivalent to subroutine USB (usb.f) with METHOD=4
    (Dejus' infinite-N + sinc^2 convolution).

    Returns (e, spec): nek+1 energies [eV] from eminu to emaxu and the
    brilliance [ph/s/mrad^2/mm^2/0.1%bw].
    """
    gamma  = energy / _MEE * 1e3
    lamdar = period * 1e8 / (2.0 * gamma**2)      # reduced wavelength [A]
    er     = _C_EVANG / lamdar                    # reduced energy [eV]
    k3     = 1.0 + 0.5 * (kx * kx + ky * ky)
    e1z    = er / k3                              # first harmonic on axis [eV]
    length = n * period * 1e-2                    # device length [m]

    sigu = sigx1 * 1e-3                           # divergences [rad]
    sigv = sigy1 * 1e-3
    fu   = 0.5 / sigu**2 if sigu > 0 else 0.0
    fv   = 0.5 / sigv**2 if sigv > 0 else 0.0
    ap2max = gamma**2 * ((_NSIGMA * sigu)**2 + (_NSIGMA * sigv)**2)
    argmax = 0.5 * _NSIGMA**2
    sigx2, sigy2 = sigx**2, sigy**2               # sizes [mm^2]

    # prefactor: angular convolution normalization, per mrad^2 (via 1e6)
    c1 = n * n * _FSC * _BW * cur * 1e-3 / _EC / (2.0 * _PI * sigu * sigv * 1e6)

    # azimuthal grid (first quadrant midpoints), 4-fold symmetry
    dphi = (_PI / 2.0) / _NPHI
    phic = (numpy.arange(_NPHI) + 0.5) * dphi
    cphi, sphi = numpy.cos(phic), numpy.sin(phic)

    # sinc^2 line shape parameters
    dew = e1z / n                                 # natural line width [eV]
    ew  = _COMEGA * dew                           # convolution half-extent
    pe  = _PI / dew

    # ------------------------------------------------------------------
    # master energy grid: harmonic-reachable range, refined at the edges
    # ------------------------------------------------------------------
    e1min = e1z * k3 / (k3 + ap2max)              # lowest reachable E1
    e1max = e1z
    emin, emax = eminu - ew, emaxu + ew
    ie1 = max(1, int(numpy.ceil(emin / e1max - 1e-12)))
    emin = max(ie1 * e1min, emin)
    ie2 = int(numpy.ceil(emax / e1min - 1e-12)) - 1
    if ie2 < ie1:
        raise RuntimeError("usb: no harmonics reachable in the energy range")
    emax = min(ie2 * e1max, emax)
    if emax <= emin:
        raise RuntimeError("usb: no harmonics reachable in the energy range")

    dw = 2.0 * ew / _NOMEGA                       # base step = dew/4
    # Dejus' variable-step master grid: walk from emin to emax with the base
    # step dw, halving the step (down to dw/128) when approaching each
    # harmonic upper edge ep = i*e1max (the spectrum is discontinuous there);
    # bracket each edge with points at ep -/+ EPSE and jump to the next
    # harmonic lower edge i*e1min when the harmonic bands do not overlap.
    pts = [emin]
    ih  = ie1
    ep  = ih * e1max
    gap = -1.0
    while pts[-1] < emax:
        idw = 1
        while idw <= 128 and pts[-1] > ep - dew / idw:
            idw *= 2
        while pts[-1] < ep - _EPSE and pts[-1] < emax - _EPSE:
            if idw <= 128 and pts[-1] > ep - dew / idw:
                idw *= 2
            pts.append(pts[-1] + dw / min(idw, 128))
        sl = min(ep, emax)
        pts[-1] = sl - _EPSE                      # place point just below edge
        pts.append(sl + _EPSE)                    # and just above
        ih += 1
        gap = ih * e1min - pts[-1]
        if gap > 0.0:
            pts.append(ih * e1min)                # jump across the gap
        ep = ih * e1max
    if gap > 0.0:
        pts = pts[:-1]                            # drop point beyond emax
    e_m = numpy.asarray(pts)
    nm  = e_m.size

    # ------------------------------------------------------------------
    # infinite-N spectrum on the master grid (sum over contributing harmonics)
    # ------------------------------------------------------------------
    spec_m = numpy.zeros(nm)
    r      = er / e_m                                       # (nm,)
    imin   = numpy.floor(k3 / r).astype(int) + 1
    imax   = numpy.floor((ap2max + k3) / r).astype(int)
    first  = numpy.ones(nm, dtype=bool)                     # first harmonic flag
    active = numpy.ones(nm, dtype=bool)
    strikes = numpy.zeros(nm, dtype=int)                    # consecutive small harmonics
    for ih in range(int(imin.min()), int(imax.max()) + 1):
        sel = active & (ih >= imin) & (ih <= imax)
        if not sel.any():
            continue
        rs      = r[sel]
        alpha2i = rs * ih - k3
        alphai  = numpy.sqrt(numpy.maximum(alpha2i, 0.0))
        thetai  = alphai / gamma
        # effective source size at this detuning (Walker)
        delta = alpha2i * n / rs
        sigr2 = numpy.where(delta < 2.15,
                            (1.29 + 1.229 * (delta - 0.8)**2)**2,
                            5.81 * delta)
        sigr2 = sigr2 * _CV * _C_EVANG / e_m[sel] * length  # [mm^2]
        const = c1 / (2.0 * _PI * numpy.sqrt((sigx2 + sigr2) * (sigy2 + sigr2)))
        # ring integral over phi with the Gaussian angular weight
        u   = thetai[:, None] * cphi[None, :]
        v   = thetai[:, None] * sphi[None, :]
        arg = u * u * fu + v * v * fv
        w   = numpy.where(arg < argmax, numpy.exp(-arg), 0.0)
        s0  = _s0_harmonic(ih, kx, ky, alpha=alphai[:, None],
                           cosphi=cphi[None, :], sinphi=sphi[None, :])
        contr = 4.0 * const * (rs / (2.0 * n)) * dphi * numpy.sum(w * s0, axis=1)
        idx = numpy.where(sel)[0]
        spec_m[idx] += contr
        # stop summing after _NSMALL consecutive (non-first) harmonics each
        # adding less than EPS of the running total (TC v1.97)
        small = (contr <= _EPSH * spec_m[idx]) & (~first[idx])
        strikes[idx] = numpy.where(small, strikes[idx] + 1, 0)
        active[idx[strikes[idx] >= _NSMALL]] = False
        first[idx] = False

    # ------------------------------------------------------------------
    # convolution with the sinc^2 line shape, onto the user energy grid.
    # The window endpoints EU -/+ EW carry sinc^2 = 0 exactly, so only the
    # trapezoid weights of the first/last grid points inside each window
    # need the partial-interval correction.
    # ------------------------------------------------------------------
    de = (emaxu - eminu) / nek
    eu = eminu + numpy.arange(nek + 1) * de
    he = numpy.diff(e_m)                                    # (nm-1,)
    ha = numpy.empty(nm)                                    # trapezoid weights
    ha[1:-1] = 0.5 * (he[:-1] + he[1:])
    ha[0], ha[-1] = 0.5 * he[0], 0.5 * he[-1]
    arg  = pe * (eu[:, None] - e_m[None, :])                # (neu, nm)
    inwin = numpy.abs(arg) < (_COMEGA * _PI) * (1.0 - 1e-12)
    snc  = numpy.ones_like(arg)
    nz   = numpy.abs(arg) > 1e-6
    snc[nz] = (numpy.sin(arg[nz]) / arg[nz]) ** 2
    W = numpy.where(inwin, ha[None, :] * snc, 0.0)
    # partial-interval weights at the window boundaries
    a_w, b_w = eu - ew, eu + ew
    j1 = numpy.searchsorted(e_m, a_w, side='right') - 1     # E[j1] <= a < E[j1+1]
    j2 = numpy.searchsorted(e_m, b_w, side='right') - 1     # E[j2] <= b < E[j2+1]
    rows = numpy.arange(nek + 1)
    s1 = (j1 >= 0) & (j1 < nm - 1)                          # left edge inside grid
    jl = numpy.clip(j1 + 1, 0, nm - 1)
    W[rows[s1], jl[s1]] = snc[rows[s1], jl[s1]] * 0.5 * (
        (e_m[jl[s1]] - a_w[s1]) +
        he[numpy.clip(jl[s1], 0, nm - 2)] * (jl[s1] < nm - 1))
    s2 = (j2 > 0) & (j2 < nm - 1)                           # right edge inside grid
    W[rows[s2], j2[s2]] = snc[rows[s2], j2[s2]] * 0.5 * (
        (b_w[s2] - e_m[j2[s2]]) + he[j2[s2] - 1])
    empty = (j2 <= 0) | (j1 >= nm - 1) | (j2 - j1 < 1)
    W[empty, :] = 0.0
    spec = (W @ spec_m) / dew
    return eu, spec


def _gauss_convolve(e, spec, sige, eiz):
    """Gaussian beam-energy-spread convolution (rms width 2*sige*eiz),
    equivalent to gauss_convolve in tc.f; trims the array edges."""
    de   = e[1] - e[0]
    sigp = 2.0 * sige * eiz / de               # sigma in grid units
    if sigp < 5.0:                             # NPPSIGMA-1
        raise RuntimeError("gauss_convolve: too few points (sige too small)")
    np_g = int(6.0 * sigp + 1.0)               # 2*NSIGMA*sigp
    if np_g % 2 == 0:
        np_g += 1
    xk = numpy.arange(np_g) - (np_g - 1) / 2.0
    gs = numpy.exp(-xk**2 / (2.0 * sigp**2))
    spec2 = numpy.convolve(spec, gs / gs.sum(), mode='valid')
    nh = np_g // 2
    return e[nh:len(e) - nh], spec2


def _peak(e, spec):
    """Abscissa and ordinate of the maximum; invalid (as in tc.f) if the
    maximum falls within IB points of either end of the array."""
    ip = int(numpy.argmax(spec))
    if ip >= len(spec) - 1 - _IB or ip <= _IB - 1:
        return None, None
    return e[ip], spec[ip]


def xoppy_calc_tcpy(
        ENERGY        = 7.0,      # ring energy [GeV]
        CURRENT       = 100.0,    # ring current [mA]
        ENERGY_SPREAD = 0.00096,  # sigma(E)/E
        SIGX          = 0.274,    # rms horizontal beam size [mm]
        SIGY          = 0.011,    # rms vertical beam size [mm]
        SIGX1         = 0.0113,   # rms horizontal divergence [mrad]
        SIGY1         = 0.0036,   # rms vertical divergence [mrad]
        PERIOD        = 3.23,     # undulator period [cm]
        NP            = 70,       # number of periods
        EMIN          = 2950.0,   # min energy of the FIRST harmonic [eV]
        EMAX          = 13500.0,  # max energy of the FIRST harmonic [eV]
        N             = 40,       # number of points of the tuning curve
        HARMONIC_FROM = 1,
        HARMONIC_TO   = 15,
        HARMONIC_STEP = 2,
        HELICAL       = 0,        # 0: planar, 1: helical
        METHOD        = 1,        # kept for interface compatibility (Dejus')
        NEKS          = 100,      # energy points for the peak search
        output_file   = "tcpy.out",
        verbose       = True,
        ):
    """
    Pure python version of xoppylib.xoppy_run_binaries.xoppy_calc_xtc
    (TC: undulator on-axis brilliance tuning curves, infinite-N method
    with convolution). Returns (data, harmonics_data) with the same
    structure: data is the stack of all rows
    [E_no_emittance, E_peak, Brilliance, Ky, Ptot, Pd], and harmonics_data
    is a list of [harmonic_number, rows-of-that-harmonic].
    """
    if METHOD not in (0, 1):
        raise ValueError("only METHOD=0,1 (Dejus, infinite-N + convolution) implemented")
    sige  = ENERGY_SPREAD
    neks  = NEKS if NEKS != 0 else 100
    ne    = N if N != 0 else 20
    ihmin = HARMONIC_FROM if HARMONIC_FROM != 0 else 1
    ihmax = HARMONIC_TO if HARMONIC_TO != 0 else 5
    ihstep= HARMONIC_STEP if HARMONIC_STEP != 0 else 2

    gamma  = ENERGY / _MEE * 1e3
    lamdar = PERIOD * 1e8 / (2.0 * gamma**2)
    er     = _C_EVANG / lamdar
    emin, emax = EMIN, EMAX
    if emax >= er:                       # reset to K=0.2 value (as in tc.f)
        emax = _C_EVANG / (lamdar * (1.0 + 0.20**2 / 2.0))
        if verbose:
            print("Warning: emax reset to %.1f eV" % emax)
    e1zmin, e1zmax = emin, emax
    de = (e1zmax - e1zmin) / (ne - 1) if ne > 1 else 0.0

    def k_of(ek):                        # deflection parameter(s) at E1=ek
        kyv = numpy.sqrt(2.0 * (er / ek - 1.0))
        if HELICAL == 1:
            kyv /= numpy.sqrt(2.0)
            return kyv, kyv              # kx, ky
        return 0.0, kyv

    harmonics = list(range(ihmin, ihmax + 1, ihstep))
    nharm = len(harmonics)
    gk_planar  = lambda k: k * (k**6 + 24.0 * k**4 / 7.0 + 4.0 * k * k + 16.0 / 7.0) / (1.0 + k * k)**3.5
    gk_helical = lambda k: 32.0 / 7.0 * k / (1.0 + k * k)**3

    # ------------------------------------------------------------------
    # peak shifts dep1, dep2 of harmonics 1 and 2 at K = Kmin
    # ------------------------------------------------------------------
    ek = e1zmax
    kx, ky = k_of(ek)
    dep = []
    for i in (1, 2):
        eiz = i * ek
        ekmin, ekmax = (0.95 * eiz, 1.01 * eiz) if i == 1 else (0.820 * eiz, 1.002 * eiz)
        e_, s_ = _usb(ENERGY, CURRENT, SIGX, SIGY, SIGX1, SIGY1, PERIOD, NP,
                      kx, ky, ekmin, ekmax, 500)
        ep, _sp = _peak(e_, s_)
        if ep is None:
            raise RuntimeError("tc: no peak found for the initial shift search (harmonic %d)" % i)
        ep = min(ep, 0.9995 * eiz)
        dep.append(eiz - ep)
    dep1, dep2 = dep

    # ------------------------------------------------------------------
    # main loop over harmonics and K values
    # ------------------------------------------------------------------
    ei = numpy.zeros((ne, nharm))
    eb = numpy.zeros((ne, nharm))
    sb = numpy.zeros((ne, nharm))
    kyb  = numpy.zeros(ne)
    kxb  = numpy.zeros(ne)
    ptot = numpy.zeros(ne)
    pd   = numpy.zeros(ne)

    shmax = 0.0
    aborted = False
    for ih, i in enumerate(harmonics):
        if aborted:
            break
        lodd = (i % 2 == 1)
        specmin = _SPECLIM if ih == 0 else _CUTOFF * shmax
        ein = 1.10 * (i + ihstep) * e1zmin
        eiz = 1.01 * ein
        je  = ne + 1
        smax = 0.0
        e_, s_ = None, None

        def store_k(j, kxv, kyv):        # K, Ptot, Pd columns (first harmonic)
            if ih == 0:
                kk2 = kxv * kxv + kyv * kyv
                kyb[j]  = kyv
                kxb[j]  = kxv
                ptot[j] = _PTOT_FAC * NP * kk2 * ENERGY**2 * CURRENT * 1e-3 / (PERIOD * 1e-2)
                gk = gk_helical(kyv) if HELICAL == 1 else gk_planar(kyv)
                pd[j] = _PD_FAC * NP * kyv * gk * ENERGY**4 * CURRENT * 1e-3 / (PERIOD * 1e-2)

        # scan down from Kmin to find the highest K-point with intensity
        while smax < specmin and je > 1 and eiz > ein:
            je -= 1
            ek = e1zmin + (je - 1) * de
            kx, ky = k_of(ek)
            store_k(je - 1, kx, ky)
            e1z = ek
            eiz = i * e1z
            if lodd:
                if HELICAL == 1:
                    ekmin = eiz - i**2 * 1.5 * dep1
                else:
                    ekmin = eiz - i * dep1
                    if i == 1:
                        ekmin -= dep1
                ekmax = eiz + i * dep1 / 2.0
                if i > int(e1z / dep1):
                    if verbose:
                        print("Warning: overlapping range for initial peak search, harmonic %d" % i)
                    aborted = True
                    break
            else:
                ekmin = (eiz - i**2 * 0.5 * dep2) if HELICAL == 1 else (eiz - 4.0 * dep2)
                ekmax = eiz
            e_, s_ = _usb(ENERGY, CURRENT, SIGX, SIGY, SIGX1, SIGY1, PERIOD, NP,
                          kx, ky, ekmin, ekmax, neks)
            smax = s_.max()
            ei[je - 1, ih] = eiz
            eb[je - 1, ih] = eiz
            sb[je - 1, ih] = 0.0
        if aborted:
            break
        if smax < _SPECLIM:
            if verbose:
                print("Warning: harmonic intensity too small for harmonic %d" % i)
            break
        ep, _sp = _peak(e_, s_)
        if ep is None:
            break
        fc = 0.990 * ep / eiz
        if i > int(1.0 / (1.0 - fc)):
            if verbose:
                print("Warning: overlapping range for peak search, harmonic %d" % i)
            break

        shmax = 0.0
        for j in range(1, je + 1):
            ek = e1zmin + (j - 1) * de
            kx, ky = k_of(ek)
            store_k(j - 1, kx, ky)
            fc2 = 1.002 if lodd else 1.000
            eiz = i * ek
            ekmin, ekmax = fc * eiz, fc2 * eiz
            nek = neks
            if sige > 0.0:
                dek = (ekmax - ekmin) / nek
                sigee = 2.0 * sige * eiz
                ekmin -= sigee * 3.0     # NSIGMA=3 for the espread convolution
                ekmax += sigee * 3.0
                if sigee / 6.0 < dek:    # NPPSIGMA=6
                    dek = sigee / 6.0
                nek = int((ekmax - ekmin) / dek + 1.0)
            e_, s_ = _usb(ENERGY, CURRENT, SIGX, SIGY, SIGX1, SIGY1, PERIOD, NP,
                          kx, ky, ekmin, ekmax, nek)
            if sige > 0.0:
                e_, s_ = _gauss_convolve(e_, s_, sige, eiz)
            ep, sp = _peak(e_, s_)
            if ep is None:
                aborted = True
                break
            shmax = max(shmax, sp)
            ei[j - 1, ih] = eiz
            eb[j - 1, ih] = ep
            sb[j - 1, ih] = sp
        if aborted:
            break
        if verbose:
            print("Harmonic %d completed" % i)

    # ------------------------------------------------------------------
    # assemble output (same structure as xoppy_calc_xtc) and write file
    # ------------------------------------------------------------------
    harmonics_data = []
    rows_all = []
    f = open(output_file, "w") if output_file is not None else None
    if f is not None:
        f.write("# TCPY: python tuning curves (infinite-N + convolution)\n")
        f.write("# On-axis Brilliance (ph/s/mrad^2/mm^2/0.1%bw)\n")
        f.write("# Energy(eV no emittance)  Energy(eV)  Brilliance  Ky  Ptot(W)  Pd(W/mr^2)\n")
    for ih, i in enumerate(harmonics):
        if HELICAL == 1:
            rows = numpy.column_stack((ei[:, ih], eb[:, ih], sb[:, ih],
                                       kxb, kyb, ptot, pd))
            fmt = "%12.3f  %12.3f  %14.5e  %8.3f  %8.3f  %9.1f  %12.1f\n"
        else:
            rows = numpy.column_stack((ei[:, ih], eb[:, ih], sb[:, ih],
                                       kyb, ptot, pd))
            fmt = "%12.3f  %12.3f  %14.5e  %8.3f  %9.1f  %12.1f\n"
        harmonics_data.append([i, rows])
        rows_all.append(rows)
        if f is not None:
            f.write("# Harmonic %d\n" % i)
            for r in rows:
                f.write(fmt % tuple(r))
    if f is not None:
        f.close()
        if verbose:
            print("File written to disk: %s" % output_file)
    data = numpy.vstack(rows_all)
    return data, harmonics_data


if __name__ == "__main__":
    if 0: # simple comparison
        args = {
            'ENERGY'        : 6.0,
            'CURRENT'       : 200.0,
            'ENERGY_SPREAD' : 0.001,
            'SIGX'          : 0.0334281,
            'SIGY'          : 0.0072813899999999996,
            'SIGX1'         : 0.00451097,
            'SIGY1'         : 0.00194034,
            'PERIOD'        : 1.7999999999999998,
            'NP'            : 111.111,
            'EMIN'          : 8008.164712466435,
            'EMAX'          : 40040.823562332174,
            'N'             : 40,
            'HARMONIC_FROM' : 1,
            'HARMONIC_TO'   : 15,
            'HARMONIC_STEP' : 2,
            'HELICAL'       : 0,
            'METHOD'        : 1,
            'NEKS'          : 100,
        }

        # NOTE on comparing with the Fortran via xoppylib.xoppy_calc_xtc:
        # the wrapper writes tc.inp with rounded values ("%d" for NP, "%10.4f"
        # for the sigmas, "%10.1f" for EMIN/EMAX), so the binary never sees the
        # full-precision args above (e.g. NP=111.111 -> 111: +0.2% in B since
        # B ~ N^2; SIGY1=0.00194034 -> 0.0019: -2.1%). For an apples-to-apples
        # comparison, feed the python code the same rounded values:
        args_as_fortran_sees_them = dict(args)
        args_as_fortran_sees_them.update(
            NP    = int(args['NP']),
            SIGX  = float("%10.4f" % args['SIGX']),
            SIGY  = float("%10.4f" % args['SIGY']),
            SIGX1 = float("%10.4f" % args['SIGX1']),
            SIGY1 = float("%10.4f" % args['SIGY1']),
            EMIN  = float("%10.1f" % args['EMIN']),
            EMAX  = float("%10.1f" % args['EMAX']),
        )

        # python (use plain **args instead for full-precision results)
        data, harmonics_data = xoppy_calc_tcpy( **args_as_fortran_sees_them )
        print("PY: Number of harmonics calculated: ", len(harmonics_data))
        print(harmonics_data[0][1][:5])

        # fortran
        from xoppylib.xoppy_run_binaries import xoppy_calc_xtc
        data_f, harmonics_data_f = xoppy_calc_xtc( **args )
        print("F: Number of harmonics calculated: ", len(harmonics_data_f))
        print(harmonics_data_f[0][1][:5])


        #
        # example plot
        #
        if True:
            from srxraylib.plot.gol import plot

            #

            #
            print("Number of harmonics calculated: ", len(harmonics_data))

            plot((harmonics_data[0][1])[:, 0],
                 (harmonics_data[0][1])[:, 2],
                 (harmonics_data_f[0][1])[:, 0],
                 (harmonics_data_f[0][1])[:, 2],
                 title="harmonic number = %d" % (harmonics_data[0][0]), xtitle="Energy[eV]", ytitle="Brilliance",
                 ylog=0, legend=['python', 'fortran'])

        #
        # end script
        #`

    if 1: #claude run
        # =============================================================================
        # make_tc_comparison.py
        #
        # Generates tc_comparison.png: tuning curves of the Fortran TC v1.97 binary
        # (solid) overlaid with xoppy_calc_tcpy (dashed red), plus relative-difference
        # panels, for two machine configurations.
        #
        # Requirements:
        #   - xoppy_calc_tcpy.py importable (same directory is fine)
        #   - one Fortran tc.out per case. Either take it from a previous
        #     xoppy_calc_xtc run, or write a tc.inp and run the xoppylib binary:
        #         <python-site-packages>/xoppylib/bin/linux/tc tc.inp
        #  Now: Run the previous example (case 1)
        #       Run the XOPPY/tc default and rename it (now /home/srio/Oasys2/tc_default.out
        # IMPORTANT: the python call below must receive the *same values the binary
        # read from tc.inp*. xoppylib writes tc.inp with "%d" for NP, "%10.4f" for
        # the four sigmas and "%10.1f" for EMIN/EMAX, so pass the rounded values
        # (e.g. NP=111, SIGY1=0.0019), not the full-precision ones.
        # =============================================================================

        import numpy as np
        import matplotlib

        # matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # from xoppy_calc_tcpy import xoppy_calc_tcpy


        def parse_tc_out(path, ncol=6):
            """Parse tc.out into a list of (ne, ncol) arrays, one per 'Harmonic' block.

            Columns: E_zero_emittance, E_peak, Brilliance, Ky, Ptot, Pdensity
            (7 columns for helical runs, which insert Kx). Non-numeric 6-field
            header lines are skipped by the float() try/except.
            """
            blocks, cur = [], None
            for line in open(path):
                if line.strip().startswith("Harmonic"):
                    if cur:
                        blocks.append(np.array(cur))
                    cur = []
                else:
                    t = line.split()
                    if cur is not None and len(t) == ncol:
                        try:
                            cur.append([float(v) for v in t])
                        except ValueError:
                            pass
            if cur:
                blocks.append(np.array(cur))
            return blocks


        # ----------------------------------------------------------------------------
        # The two cases: (title, fortran tc.out path, args exactly as in tc.inp)
        # ----------------------------------------------------------------------------
        cases = [
            ("APS-like: 7 GeV, $\\lambda_0$=3.23 cm, N=70",
             "/home/srio/Oasys2/tc_default.out",  # <- adapt path
             dict(ENERGY=7.0, CURRENT=100.0, ENERGY_SPREAD=0.00096,
                  SIGX=0.274, SIGY=0.011, SIGX1=0.0113, SIGY1=0.0036,
                  PERIOD=3.23, NP=70, EMIN=2950.0, EMAX=13500.0)),

            ("MBA-like: 6 GeV, $\\lambda_0$=1.8 cm, N=111",
             "tc.out",  # <- adapt path (your run)
             dict(ENERGY=6.0, CURRENT=200.0, ENERGY_SPREAD=0.001,
                  SIGX=0.0334, SIGY=0.0073, SIGX1=0.0045, SIGY1=0.0019,
                  PERIOD=1.8, NP=111, EMIN=8008.2, EMAX=40040.8)),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))

        for col, (title, ftout, kw) in enumerate(cases):
            b_fortran = parse_tc_out(ftout)

            _, hd = xoppy_calc_tcpy(N=40, HARMONIC_FROM=1, HARMONIC_TO=15,
                                    HARMONIC_STEP=2, HELICAL=0, METHOD=1, NEKS=100,
                                    verbose=False, output_file=None, **kw)

            ax0, ax1 = axes[0, col], axes[1, col]
            for ihx, blk in enumerate(b_fortran):
                py = hd[ihx][1]
                m = blk[:, 2] > 0  # skip cutoff (zeroed) rows
                # top: tuning curves, peak energy vs peak brilliance
                ax0.plot(blk[m, 1], blk[m, 2], "-", lw=2.2, alpha=0.85, color="C0",
                         label="Fortran" if ihx == 0 else None)
                ax0.plot(py[m, 1], py[m, 2], "--", lw=1.1, color="red",
                         label="python" if ihx == 0 else None)
                # bottom: relative difference in units of 1e-5
                ax1.plot(blk[m, 1], 1e5 * (py[m, 2] / blk[m, 2] - 1), ".-",
                         ms=3, lw=0.7, label="h%d" % hd[ihx][0])

            ax0.set_xscale("log");
            ax0.set_yscale("log")
            ax0.set_title(title, fontsize=10)
            ax0.set_ylabel("Brilliance (ph/s/mrad$^2$/mm$^2$/0.1%bw)", fontsize=8)
            ax0.legend(fontsize=8);
            ax0.tick_params(labelsize=8)

            ax1.set_xscale("log")
            ax1.set_xlabel("Photon energy (eV)", fontsize=9)
            ax1.set_ylabel("rel. difference ($\\times 10^{-5}$)", fontsize=8)
            ax1.legend(fontsize=6, ncol=4);
            ax1.tick_params(labelsize=8)

        fig.suptitle("xoppy_calc_tcpy vs TC v1.97 binary (xoppylib): odd harmonics 1-15",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        # fig.savefig("tc_comparison.png", dpi=140)
        # print("tc_comparison.png written")
        plt.show()