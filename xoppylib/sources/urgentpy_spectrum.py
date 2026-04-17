"""
urgent_spectrum.py  —  URGENT Fortran faithful Python translation
Walker & Diviacco, Rev. Sci. Instrum. 63, 392 (1992).

Two modes, matching the two Fortran paths:
  zero_emittance=True  → SUB4/ICALC=3  : RAD0 = F3 · |An|² · SINC at slit grid
  zero_emittance=False → SUB2/ICALC=1  : infinite-N envelope + sinc² convolution

Grid convention (both modes):
  np.outer(x_mm, np.ones(NYP)) → shape (NXP, NYP), axis-0 = X = Fortran IB-loop
  np.outer(np.ones(NXP), y_mm) → shape (NXP, NYP), axis-1 = Y = Fortran IC-loop
  W2d = np.outer(Wx, Wy) with the same (NXP, NYP) layout.

Convolution (emittance mode):
  Fortran: F(I) = sum_{J=1}^{NW} G(I + NE1 - J) * H(J) * DW   (1-indexed)
  Python:  F[i] = sum_{j=0}^{NW-1} G[i + NE1_py - j] * H[j] * DW  (0-indexed)
  where NE1_py = NOMEGA//2 (= Fortran NE1 - 1).
"""
import time
import numpy as np
import scipy.constants as codata
from scipy.special import jv

ECHARGE = codata.e; EPS0 = codata.epsilon_0; HBAR = codata.hbar
C_LIGHT = codata.c; M_E = codata.m_e
_F455e7 = 4.55e7   # Fortran practical constant

# ---------------------------------------------------------------------------
def calc1d_urgentpy(bl,
                  photonEnergyMin=1000.0,
                  photonEnergyMax=100000.0,
                  photonEnergyPoints=500,
                  n_harmonics=None,
                  hSlitPoints=50,
                  vSlitPoints=50,
                  zero_emittance=False,
                  verbose=True):
    """
    Undulator spectral flux [ph/s/0.1%bw] through a rectangular slit.
    Faithful Python translation of URGENT Fortran (Walker & Diviacco 1992).

    Parameters
    ----------
    bl : dict
        ElectronEnergy          [GeV]
        ElectronCurrent         [A]
        ElectronBeamSizeH/V     [m]   rms
        ElectronBeamDivergenceH/V [rad] rms
        PeriodID                [m]
        NPeriods                int
        Kv                      deflection parameter Ky
        distance                [m]
        gapH, gapV              [m]   full aperture
    photonEnergyMin/Max : float [eV]
    photonEnergyPoints  : int
    n_harmonics         : int or None  (None → auto)
    hSlitPoints, vSlitPoints : int
        Number of slit integration intervals (Fortran NXP/NYP input).
        NXP_points = hSlitPoints+1, DXP = gapH/2 / hSlitPoints.
        Default 50 matches the Fortran example (NXP=50).
    zero_emittance : bool
    verbose         : bool

    Returns
    -------
    eArray    : ndarray [eV]
    fluxArray : ndarray [ph/s/0.1%bw]
    """
    t0 = time.time()
    gamma = _gamma(bl['ElectronEnergy'])
    E1    = _fortran_E1(bl['Kv'], bl['PeriodID'], gamma)
    if verbose:
        print("URGENT Python  (Walker & Diviacco 1992)")
        print(f"  γ={gamma:.1f}  E₁={E1/1e3:.3f} keV  K={bl['Kv']:.4f}  N={bl['NPeriods']}")
        mode = "zero emittance (SUB4/ICALC=3)" if zero_emittance else "with emittance (SUB2/ICALC=1)"
        print(f"  {mode},  slit grid {hSlitPoints}×{vSlitPoints}")

    eArray = np.linspace(photonEnergyMin, photonEnergyMax, photonEnergyPoints)

    if zero_emittance:
        f = _calc_zero_emittance(bl, eArray, n_harmonics,
                                  hSlitPoints, vSlitPoints, verbose)
    else:
        f = _calc_emittance(bl, eArray, hSlitPoints, vSlitPoints, verbose)

    if verbose:
        ip = f.argmax()
        print(f"  Done in {time.time()-t0:.1f} s  "
              f"Peak={f[ip]:.4e} ph/s/0.1%bw  at E={eArray[ip]/1e3:.3f} keV")
    return eArray, f


# ---------------------------------------------------------------------------
def _gamma(E_GeV):
    return 1e9 * E_GeV / (M_E * C_LIGHT**2 / ECHARGE)

def _fortran_E1(K, period_m, gamma):
    """E1 = 12398.5 / (LAMDAR_Å × K3)  [eV]  — Fortran convention."""
    LAMDAR_A = period_m * 1e10 / (2.0 * gamma**2)
    return 12398.5 / (LAMDAR_A * (1.0 + K**2 / 2.0))

def harmonic_energy(n, K, period_m, gamma):
    return n * _fortran_E1(K, period_m, gamma)

# ---------------------------------------------------------------------------
def _p_max(n, K, extra=3):
    return max(3, int(np.ceil((n * K / np.sqrt(1 + K**2 / 2) + n) / 2)) + extra)

def _An2_at_alpha_phi(n, K, gamma, alpha, cos_phi, sin_phi):
    """
    |An(alpha, phi)|^2 for scalar alpha and phi-arrays.
    Matches Fortran BRIGH1 (plane undulator, Kx=0).
    """
    ax = alpha * cos_phi; ay = alpha * sin_phi
    denom = 1 + K**2 / 2 + ax**2 + ay**2; xi = n / denom
    X = 2 * xi * ax * K; Y = xi * K**2 / 4
    pm = _p_max(n, K); p = np.arange(-pm, pm + 1)
    Jp_Y  = jv(p[:, None], Y[None, :])
    ords  = 2 * p[:, None] + np.array([-1, 0, 1])[None, :] + n
    Jm_X  = jv(ords[:, :, None], X[None, None, :])
    S     = (Jp_Y[:, None, :] * Jm_X).sum(0)
    Ax = xi * (2 * ax * S[1] - K * (S[2] + S[0])); Ay = xi * (2 * ay * S[1])
    return Ax**2 + Ay**2

def _SINC(alpha2, ALP2I, R, N):
    """Fortran SINC = (sin(X)/X)^2,  X = N*pi*(alpha^2-ALP2I)/R.  Peak=1."""
    X = N * np.pi * (alpha2 - ALP2I) / R
    return np.where(np.abs(X) < 1e-9, 1.0, (np.sin(X) / X)**2)

def _An2_2d(n, K, gamma, theta_x, theta_y):
    """|An|^2 on 2D angle grids (shape NXP x NYP)."""
    ax = gamma * theta_x; ay = gamma * theta_y
    denom = 1 + K**2 / 2 + ax**2 + ay**2; xi = n / denom
    X = 2 * xi * ax * K; Y = xi * K**2 / 4
    pm = _p_max(n, K); p = np.arange(-pm, pm + 1)
    Jp_Y = jv(p[:, None, None], Y[None])
    ords  = 2 * p[:, None] + np.array([-1, 0, 1])[None, :] + n
    Jm_X  = jv(ords[:, :, None, None], X[None, None])
    S     = (Jp_Y[:, None] * Jm_X).sum(0)
    Ax = xi * (2 * ax * S[1] - K * (S[2] + S[0])); Ay = xi * (2 * ay * S[1])
    return Ax**2 + Ay**2

# ---------------------------------------------------------------------------
def _slit_grid(gapH, gapV, NXP_inp, NYP_inp):
    """
    Build slit grid matching Fortran (centred slit, 1-quadrant, IB outer, IC inner).

    Fortran input NXP_inp → NXP = NXP_inp+1 points, DXP = (gapH/2)/NXP_inp [mm].
    np.outer layout: axis-0 = X (IB), axis-1 = Y (IC).
    """
    NXP = NXP_inp + 1; NYP = NYP_inp + 1
    DXP = gapH / 2 * 1e3 / NXP_inp   # mm
    DYP = gapV / 2 * 1e3 / NYP_inp
    x_mm = np.linspace(0, gapH / 2 * 1e3, NXP)
    y_mm = np.linspace(0, gapV / 2 * 1e3, NYP)
    # axis-0 = X (IB), axis-1 = Y (IC)
    XP_mm = np.outer(x_mm, np.ones(NYP))
    YP_mm = np.outer(np.ones(NXP), y_mm)
    XP_m  = XP_mm / 1000.0
    YP_m  = YP_mm / 1000.0
    Wx = np.ones(NXP); Wx[0] = Wx[-1] = 0.5
    Wy = np.ones(NYP); Wy[0] = Wy[-1] = 0.5
    W2d = np.outer(Wx, Wy)   # (NXP, NYP)
    return XP_m, YP_m, W2d, DXP, DYP, NXP, NYP

# ---------------------------------------------------------------------------
def _calc_emittance(bl, E_user, NXP_inp, NYP_inp, verbose):
    """
    SUB2/ICALC=1 exact translation:
      Step A — infinite-N angular spectrum with emittance Gaussian convolution.
      Step B — sinc² convolution restoring the natural line shape.
    """
    E_GeV = bl['ElectronEnergy']; I_A = bl['ElectronCurrent']
    K = bl['Kv']; N = int(bl['NPeriods']); period_m = bl['PeriodID']
    D = bl['distance']; gapH = bl['gapH']; gapV = bl['gapV']
    gamma = _gamma(E_GeV)
    LAMDAR_A = period_m * 1e10 / (2.0 * gamma**2)
    K3 = 1 + K**2 / 2; E1 = 12398.5 / (LAMDAR_A * K3)

    sig_xp = bl.get('ElectronBeamDivergenceH', 0.0)
    sig_yp = bl.get('ElectronBeamDivergenceV', 0.0)
    sig_x  = bl.get('ElectronBeamSizeH',       0.0)
    sig_y  = bl.get('ElectronBeamSizeV',        0.0)
    sigu2  = sig_xp**2 + (sig_x / D)**2
    sigv2  = sig_yp**2 + (sig_y / D)**2
    sigu   = np.sqrt(sigu2); sigv = np.sqrt(sigv2)
    FU     = 0.5 / sigu2;    FV   = 0.5 / sigv2
    NSIG   = 4;               ARGMAX = NSIG**2 / 2.0

    # Acceptance (Fortran lines 200-227, centred slit)
    XEMAX  = gapH / (2 * D) + NSIG * sigu
    YEMAX  = gapV / (2 * D) + NSIG * sigv
    APMAX  = gamma**2 * (XEMAX**2 + YEMAX**2)
    APMIN  = 0.0

    F1  = _F455e7 * N**2 * I_A / (2.0 * np.pi * sigu * sigv * D**2)
    FAC = 4.0   # centred slit: 4-fold symmetry

    # Phi grid (Fortran NPHI=20, NPHI1=80)
    NPHI1   = 80; DPHI = 2.0 * np.pi / NPHI1
    phi     = np.arange(NPHI1) * DPHI
    cos_phi = np.cos(phi); sin_phi = np.sin(phi)

    # Slit grid
    XP_m, YP_m, W2d, DXP, DYP, NXP, NYP = _slit_grid(gapH, gapV, NXP_inp, NYP_inp)

    # Fine energy grid (Fortran lines 115-130)
    DE_u = E_user[1] - E_user[0]; DOMEGA = 2.0
    NOMEGA = int(2 * DOMEGA * E1 / (N * DE_u) + 1); NOMEGA = 2 * (NOMEGA // 2)
    if NOMEGA < int(4 * DOMEGA + 0.5):
        NOMEGA = int(4 * DOMEGA + 0.5); NOMEGA = 2 * (NOMEGA // 2)
    DE     = 2.0 * DOMEGA * E1 / (NOMEGA * N)
    EMIN_f = E_user[0]  - NOMEGA * DE / 2.0
    EMAX_f = E_user[-1] + NOMEGA * DE / 2.0
    NE_f   = int((EMAX_f - EMIN_f) / DE) + 2
    NE1    = NOMEGA // 2           # 0-indexed (= Fortran NE1 - 1)
    NE2    = NE_f - NE1
    E_fine = EMIN_f + np.arange(NE_f) * DE

    n_hmax = max(1, int(E_user[-1] / E1 * 1.2))
    if verbose:
        print(f"  NOMEGA={NOMEGA}, DE_fine={DE:.1f} eV, NE1={NE1}, n_harm={n_hmax}")

    # --- Step A: infinite-N spectrum (Fortran SUB2 lines 544-705) -----------
    SPEC0 = np.zeros(NE_f)
    for iE, E in enumerate(E_fine):
        R    = E1 * K3 / E
        IMIN = int((APMIN + K3) / R) + 1    # Fortran line 566
        IMAX = int((APMAX + K3) / R)        # Fortran line 567
        if IMAX < IMIN: continue
        for n in range(max(1, IMIN), min(n_hmax, IMAX) + 1):
            ALP2I = R * n - K3              # Fortran line 582
            if ALP2I < 0: continue
            ALPI  = np.sqrt(ALP2I); THETA = ALPI / gamma
            # BRI0 = |An(ALPI, phi)|^2  — infinite-N, no SINC (Fortran BRIF ICALC=2)
            BRI0  = _An2_at_alpha_phi(n, K, gamma, ALPI, cos_phi, sin_phi)
            XE1v  = THETA * cos_phi; YE1v = THETA * sin_phi   # emission angles
            # Emittance Gaussian: U = XP/D - XE1  (Fortran lines 659-661)
            # XP_m: (NXP,NYP), XE1v: (NPHI1,) → broadcast to (NXP,NYP,NPHI1)
            U   = XP_m[:, :, None] / D - XE1v[None, None, :]
            V   = YP_m[:, :, None] / D - YE1v[None, None, :]
            ARG = U**2 * FU + V**2 * FV
            P   = np.where(ARG < ARGMAX, np.exp(-np.where(ARG < ARGMAX, ARG, 0.0)), 0.0)
            SUM0   = (BRI0[None, None, :] * P).sum(axis=2)   # (NXP,NYP)
            DELTA0 = F1 * SUM0 * DPHI * R / (2.0 * N)        # Fortran line 672
            SPEC0[iE] += FAC * (W2d * DELTA0).sum() * DXP * DYP

    # --- Step B: convolve with H(W)=sinc²(πW), W=N·ΔE/E1 (Fortran lines 709-714) ---
    DW    = N * DE / E1; NW = NOMEGA + 1
    W_arr = -DOMEGA + np.arange(NW) * (2.0 * DOMEGA / NOMEGA)
    H_arr = np.where(np.abs(W_arr) < 1e-9, 1.0,
                     (np.sin(np.pi * W_arr) / (np.pi * W_arr))**2)

    # Fortran CONV: F(I) = sum_{J=1}^{NW} G(I+NE1_fort-J)*H(J)*DW  (1-indexed)
    # 0-indexed:    F[i] = sum_{j=0}^{NW-1} G[i+NE1-j]*H[j]*DW
    SPEC_conv = np.zeros(NE_f)
    for i in range(NE1, NE2):
        s = 0.0
        for j in range(NW):
            idx = i + NE1 - j          # correct Fortran CONV index
            if 0 <= idx < NE_f:
                s += SPEC0[idx] * H_arr[j]
        SPEC_conv[i] = s * DW

    # Interpolate fine-grid output onto user grid
    E_out = E_fine[NE1:NE2]; F_out = SPEC_conv[NE1:NE2]
    return np.interp(E_user, E_out, F_out)

# ---------------------------------------------------------------------------
def _calc_zero_emittance(bl, E_user, n_harmonics, NXP_inp, NYP_inp, verbose):
    """
    SUB4/ICALC=3 exact translation:
      RAD0(XP,YP) = F3 · |An(alpha,phi)|^2 · SINC(alpha^2, ALP2I, R, N)
    """
    E_GeV = bl['ElectronEnergy']; I_A = bl['ElectronCurrent']
    K = bl['Kv']; N = int(bl['NPeriods']); period_m = bl['PeriodID']
    D = bl['distance']; gapH = bl['gapH']; gapV = bl['gapV']
    gamma = _gamma(E_GeV)
    LAMDAR_A = period_m * 1e10 / (2.0 * gamma**2)
    K3 = 1 + K**2 / 2; E1 = 12398.5 / (LAMDAR_A * K3)
    F3  = _F455e7 * N**2 * gamma**2 * I_A / D**2
    FAC = 4.0

    XP_m, YP_m, W2d, DXP, DYP, NXP, NYP = _slit_grid(gapH, gapV, NXP_inp, NYP_inp)
    alpha2 = (gamma * XP_m / D)**2 + (gamma * YP_m / D)**2

    if n_harmonics is None:
        n_harmonics = max(1, int(E_user[-1] / E1 * 1.2))
    # Precompute |An|^2 on grid
    An2g = [_An2_2d(n, K, gamma, XP_m / D, YP_m / D)
            for n in range(1, n_harmonics + 1)]

    flux = np.zeros(len(E_user))
    for iE, E in enumerate(E_user):
        R = E1 * K3 / E; val = 0.0
        for n in range(1, n_harmonics + 1):
            ALP2I = R * n - K3
            ls    = _SINC(alpha2, ALP2I, R, N)
            if ls.max() == 0.0: continue
            val += FAC * (W2d * An2g[n - 1] * ls).sum() * DXP * DYP
        flux[iE] = F3 * val
    return flux




# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt


    # ---------------------------------------------------------------------------
    def run_urgent_fortran(bl,
                           photonEnergyMin=3000.0,
                           photonEnergyMax=110000.0,
                           photonEnergyPoints=1500,
                           zero_emittance=False):
        """
        Call xoppylib URGENT (METHOD=1 = URGENT Fortran) and return
        (energy [eV], flux [ph/s/0.1%bw]).
        """
        from xoppylib.sources.xoppy_undulators import xoppy_calc_undulator_spectrum
        energy, flux, _, _ = xoppy_calc_undulator_spectrum(
            ELECTRONENERGY=bl['ElectronEnergy'],
            ELECTRONENERGYSPREAD=bl.get('ElectronEnergySpread', 0.001),
            ELECTRONCURRENT=bl['ElectronCurrent'],
            ELECTRONBEAMSIZEH=bl.get('ElectronBeamSizeH', 3.34e-5),
            ELECTRONBEAMSIZEV=bl.get('ElectronBeamSizeV', 7.28e-6),
            ELECTRONBEAMDIVERGENCEH=bl.get('ElectronBeamDivergenceH', 4.51e-6),
            ELECTRONBEAMDIVERGENCEV=bl.get('ElectronBeamDivergenceV', 1.94e-6),
            PERIODID=bl['PeriodID'],
            NPERIODS=bl['NPeriods'],
            KV=bl['Kv'],
            KH=bl.get('Kh', 0.0),
            KPHASE=bl.get('Kphase', 0.0),
            DISTANCE=bl['distance'],
            GAPH=bl['gapH'],
            GAPV=bl['gapV'],
            GAPH_CENTER=0.0,
            GAPV_CENTER=0.0,
            PHOTONENERGYMIN=photonEnergyMin,
            PHOTONENERGYMAX=photonEnergyMax,
            PHOTONENERGYPOINTS=photonEnergyPoints,
            METHOD=1,  # 1=URGENT Fortran, 2=SRW
            USEEMITTANCES=0 if zero_emittance else 1)
        return energy, flux


    bl = dict(
        ElectronEnergy          = 6.0,
        ElectronCurrent         = 0.2,
        ElectronBeamSizeH       = 3.34281e-05,
        ElectronBeamSizeV       = 7.28139e-06,
        ElectronBeamDivergenceH = 4.51097e-06,
        ElectronBeamDivergenceV = 1.94034e-06,
        ElectronEnergySpread    = 0.001,
        PeriodID                = 0.018,
        NPeriods                = 111,
        Kv                      = 1.68,
        distance                = 30.0,
        gapH                    = 0.001,
        gapV                    = 0.001,
    )
    Emin, Emax, Npts = 3000.0, 110000.0, 1500

    print("=" * 60)
    print("  URGENT Python — Walker & Diviacco (1992)")
    print("=" * 60)

    results = {}
    for label, ze in [("zero emittance", True), ("with emittance", False)]:
        print(f"\n--- Python ({label}) ---")
        e, f = calc1d_urgentpy(bl, Emin, Emax, Npts,
                             hSlitPoints=50, vSlitPoints=50,
                             zero_emittance=ze, verbose=True)
        results[label] = (e, f)

    try:
        for label, ze in [("zero emittance", True), ("with emittance", False)]:
            ef, ff = run_urgent_fortran(bl, Emin, Emax, Npts, ze)
            results[f"Fortran {label}"] = (ef, ff)
            print(f"Fortran ({label}): peak={ff.max():.4e}")
    except Exception as ex:
        print(f"Fortran not available: {ex}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    style = {
        "zero emittance":        ("b-",   1.0),
        "with emittance":        ("g-",   1.0),
        "Fortran zero emittance":("r--",  1.0),
        "Fortran with emittance":("m--",  1.0),
    }
    for label, (e, f) in results.items():
        ls, lw = style.get(label, ("k-", 1.0))
        axes[0].plot(e / 1e3, f, ls, lw=lw, label=label)
        axes[1].semilogy(e / 1e3, np.maximum(f, 1e10), ls, lw=lw, label=label)

    title = (f"URGENT Python  K={bl['Kv']}, N={bl['NPeriods']}, "
             f"E={bl['ElectronEnergy']} GeV, I={bl['ElectronCurrent']*1e3:.0f} mA, "
             f"slit {bl['gapH']*1e3:.0f}×{bl['gapV']*1e3:.0f} mm @ {bl['distance']:.0f} m")
    fig.suptitle(title, fontsize=9)
    for ax, sc in zip(axes, ["Linear", "Log"]):
        ax.set_xlabel("Photon energy [keV]")
        ax.set_ylabel("Flux [ph/s/0.1%bw]")
        ax.set_title(sc)
        ax.set_xlim(Emin / 1e3, Emax / 1e3)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    # plt.savefig("urgent_spectrum.png", dpi=150, bbox_inches="tight")
    # print("\nSaved urgent_spectrum.png")
    plt.show()