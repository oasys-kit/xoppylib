"""

Calculation of the power density for the individual harmonics. 

Exact implementation like in the URGENT code: Walker & Diviacco, Rev. Sci. Instrum. 63, 392 (1992)

PLANE UNDULATOR (Kx=0), Section VI Power Density.

Amplitude (from paper, Kx=0):
    A = xi * ( 2*alpha_x*S0 - Ky*(S1 + S_minus1),
               2*alpha_y*S0,
               0 )

    S_q = sum_{p} J_p(Y) * J_{2p+q+n}(X)    q = 0, +1, -1
    xi  = n / (1 + Ky^2/2 + alpha^2)
    X   = 2 * xi * alpha_x * Ky      (alpha_x = gamma * theta_x)
    Y   = xi * Ky^2 / 4

Power density (Walker 1992, Section VI):
    Integrate d²I/dw dΩ over frequency:
        integral L(Δω/ω₁) dω = N · ω₁           [N in NUMERATOR]
        ω₁ = 4πcγ² / (λ₀ · denom)
    =>
        dI/dΩ = (e · γ⁴ · Ib · N) / (ε₀ · λ₀)  ·  |An|² / denom   [W/rad²]

p_max RULE (adaptive):
    The Bessel sums converge once |J_p(Y)| and |J_{2p+n}(X)| are negligible.
    X is maximised at alpha = sqrt(1 + Ky²/2), giving X_max = n·Ky/sqrt(1+Ky²/2).
    Safe rule (error < 1e-12):
        p_max(n, K) = ceil( n · (K/sqrt(1+K²/2) + 1) / 2 ) + 3
    For K=1.358: p_max = 4, 5, 6, 7, 8 … for n = 1, 2, 3, 4, 5 …
    Using a fixed p_max=20 wastes ~40% CPU for typical K and low harmonics.

"""

import numpy as np
from scipy.special import jv
from scipy.ndimage import gaussian_filter
import scipy.constants as codata

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def harmonic_energy(n, K, period_m, gamma):
    """Resonant photon energy [eV] for harmonic n on-axis."""
    hc_eVm = codata.h*codata.c / codata.e # 1239.84193e-9
    return n * 2.0 * gamma**2 * hc_eVm / (period_m * (1.0 + K**2 / 2.0))


def _p_max(n, Ky, extra=3):
    """
    Minimum truncation order for the Bessel sum to converge to <1e-12 relative error.

    The maximum argument of J_{2p+q+n}(X) occurs at alpha = sqrt(1+Ky²/2):
        X_max = n · Ky / sqrt(1 + Ky²/2)
    The sum needs 2·p_max + n > X_max. Adding `extra` as safety margin.

    Results for K=1.358:  n=1→4, n=2→5, n=3→6, n=5→8, n=10→13, n=20→23
    cf. fixed p_max=20: always correct but 40-80% over-conservative for n<17.
    """
    X_max = n * Ky / np.sqrt(1.0 + Ky**2 / 2.0)
    return max(3, int(np.ceil((X_max + n) / 2.0)) + extra)


def _Sq_all(X, Y, n, p_max):
    """
    Compute S_{-1}, S_0, S_{+1} in one vectorised call.

    S_q = sum_{p=-pmax}^{pmax} J_p(Y) · J_{2p+q+n}(X)

    Speedup over three separate loops: J_p(Y) is computed once and all three
    Bessel index offsets (q = -1, 0, +1) are stacked into a single jv() call.

    Returns
    -------
    Sm1, S0, S1 : 2-D arrays (same shape as X)
    """
    p_arr = np.arange(-p_max, p_max + 1)               # (2p+1,)

    # J_p(Y) for all p: shape (2p+1, ny, nx)
    Jp_Y = jv(p_arr[:, None, None], Y[None, :, :])

    # Bessel indices for q = -1, 0, +1: shape (2p+1, 3)
    orders = 2 * p_arr[:, None] + np.array([-1, 0, 1])[None, :] + n

    # J_{orders}(X): shape (2p+1, 3, ny, nx)
    Jm_X = jv(orders[:, :, None, None], X[None, None, :, :])

    # Sum over p -> (3, ny, nx)
    S = (Jp_Y[:, None, :, :] * Jm_X).sum(axis=0)
    return S[0], S[1], S[2]    # Sm1, S0, S1


# --------------------------------------------------------------------------- #
# Core calculation                                                             #
# --------------------------------------------------------------------------- #

def power_density_harmonic(n, Ky, N_periods, period_m, gamma, current_A,
                            theta_x_rad, theta_y_rad, p_max=None):
    """
    Power density [W/rad²] for harmonic n of a plane undulator.

    Walker & Diviacco, Rev. Sci. Instrum. 63, 392 (1992), Section VI.

    Parameters
    ----------
    p_max : int or None
        Bessel sum truncation. None (default) uses the adaptive rule and is
        recommended. A fixed p_max=20 is always valid but up to 4× slower.
    """
    if p_max is None:
        p_max = _p_max(n, Ky)

    Ky2     = Ky**2
    alpha_x = gamma * theta_x_rad
    alpha_y = gamma * theta_y_rad
    alpha2  = alpha_x**2 + alpha_y**2

    denom = 1.0 + Ky2 / 2.0 + alpha2
    xi    = n / denom
    X     = 2.0 * xi * alpha_x * Ky   # signed; |A|² is always symmetric in theta_x
    Y     = xi * Ky2 / 4.0

    Sm1, S0, S1 = _Sq_all(X, Y, n, p_max)

    Ax = xi * (2.0 * alpha_x * S0 - Ky * (S1 + Sm1))
    Ay = xi * (2.0 * alpha_y * S0)

    # Walker 1992 Section VI prefactor — N is in the NUMERATOR:
    #   integral L(Δω/ω₁) dω = N·ω₁   =>   dI/dΩ = (e·γ⁴·Ib·N)/(ε₀·λ₀) · |An|²/denom
    prefactor = codata.e * gamma**4 * current_A * N_periods / (codata.epsilon_0 * period_m)

    return prefactor * (Ax**2 + Ay**2) / denom   # [W/rad²]


def power_density_all_harmonics(Ky, N_periods, period_m, gamma, current_A,
                                 theta_x_rad, theta_y_rad,
                                 n_harmonics=200, p_max=None,
                                 sigma_x=0.0, sigma_y=0.0,
                                 sigma_xp=0.0, sigma_yp=0.0,
                                 distance_m=1.0,
                                 zero_emittance=True):
    """
    Sum power density [W/rad²] over harmonics 1 … n_harmonics.

    Optional Gaussian emittance convolution (angular broadening only):
        use_emittance      : flag, to do convolution
        sigma_x, sigma_y   : rms beam sizes [m]
        sigma_xp, sigma_yp : rms beam divergences [rad]
        distance_m         : source-to-screen distance [m]
    """

    indiv = np.zeros((n_harmonics, theta_x_rad.shape[0], theta_x_rad.shape[1]), dtype=float)


    # use_emittance = any(v > 0 for v in (sigma_x, sigma_y, sigma_xp, sigma_yp))
    if not zero_emittance:
        dtheta_x = abs(theta_x_rad[1, 0] - theta_x_rad[0, 0])
        dtheta_y = abs(theta_y_rad[0, 1] - theta_y_rad[0, 0])
        print(">>>>> sigmas: ", sigma_xp, sigma_yp, )
        sig_u = sigma_xp # np.sqrt((sigma_x / distance_m)**2 + sigma_xp**2)
        sig_v = sigma_yp # np.sqrt((sigma_y / distance_m)**2 + sigma_yp**2)
        sx_pix = sig_u / dtheta_x
        sy_pix = sig_v / dtheta_y
        print(">>>>> sigmas: ", sigma_xp, sigma_yp, sig_u, sig_vgit add , sx_pix, sy_pix )

    for n in range(1, n_harmonics + 1):
        pd_n = power_density_harmonic(n, Ky, N_periods, period_m,
                                       gamma, current_A,
                                       theta_x_rad, theta_y_rad, p_max)

        if not zero_emittance:
            pd_n = gaussian_filter(pd_n, sigma=[sx_pix, sy_pix])

        indiv[n-1] = pd_n

    return indiv




# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    # --------------------------------------------------------------------------- #
    # Fortran URGENT wrapper                                                       #
    # --------------------------------------------------------------------------- #
    from xoppylib.sources.srundplug import calc2d_from_harmonics_urgent
    import matplotlib.pyplot as plt
    import time

    def run_urgent_fortran(h5_parameters, n_harmonics=10, zero_emittance=True, do_plot=False, show=True, verbose=False):
        """
        Call the Fortran URGENT code via xoppylib.

        Returns
        -------
        X, Y              : 1-D arrays [mm] at the screen
        POWER_DENSITY     : 2-D array [W/mm²], total, with emittance
        POWER_DENSITY_HARMONICS : 3-D [n_harm, ny, nx]  [W/mm²]
        ENERGY_HARMONICS  : 1-D [eV]
        FLUX              : 3-D [n_harm, ny, nx]  [ph/s/0.1%bw/mm²]
        """

        bl = {
            'ElectronBeamDivergenceH': h5_parameters["ELECTRONBEAMDIVERGENCEH"],
            'ElectronBeamDivergenceV': h5_parameters["ELECTRONBEAMDIVERGENCEV"],
            'ElectronBeamSizeH': h5_parameters["ELECTRONBEAMSIZEH"],
            'ElectronBeamSizeV': h5_parameters["ELECTRONBEAMSIZEV"],
            'ElectronCurrent': h5_parameters["ELECTRONCURRENT"],
            'ElectronEnergy': h5_parameters["ELECTRONENERGY"],
            'ElectronEnergySpread': h5_parameters["ELECTRONENERGYSPREAD"],
            'Kv': h5_parameters["KV"],
            'Kh': h5_parameters["KH"],
            'Kphase': h5_parameters["KPHASE"],
            'NPeriods': h5_parameters["NPERIODS"],
            'PeriodID': h5_parameters["PERIODID"],
            'distance': h5_parameters["DISTANCE"],
            'gapH': h5_parameters["GAPH"],
            'gapV': h5_parameters["GAPV"],
        }

        X, Y, POWER_DENSITY, POWER_DENSITY_HARMONICS, ENERGY_HARMONICS, FLUX = \
            calc2d_from_harmonics_urgent(
                bl, zero_emittance=zero_emittance, fileName=None, fileAppend=False,
                hSlitPoints=h5_parameters["HSLITPOINTS"],
                vSlitPoints=h5_parameters["VSLITPOINTS"],
                harmonic_max=n_harmonics,
                return_flux=1)

        dx = X[1] - X[0];
        dy = Y[1] - Y[0]

        if verbose:
            print("\nharm  power[W]  power-from-flux[W]  pd-peak[W/mm²]  flux[ph/s/0.1%bw]")
            for i in range(POWER_DENSITY_HARMONICS.shape[0]):
                print("  %2d  %10.3f  %10.3f  %12.4f  %g" % (
                    i + 1,
                    POWER_DENSITY_HARMONICS[i].sum() * dx * dy,
                    (FLUX * ENERGY_HARMONICS * codata.e)[i].sum() * dx * dy,
                    POWER_DENSITY_HARMONICS[i].max(),
                    FLUX[i].sum() * dx * dy))

        return X, Y, POWER_DENSITY, POWER_DENSITY_HARMONICS, ENERGY_HARMONICS, FLUX

    #
    # Main inputs
    #
    n_harmonics       = 20
    calculate_python  = 1   # 1=power density, 2=Flux
    calculate_fortran = 1   # requires xoppylib 1=power density, 2=Flux
    zero_emittance = 0

    h5_parameters = dict(
        ELECTRONENERGY          = 6.0,
        ELECTRONENERGYSPREAD    = 0.001,
        ELECTRONCURRENT         = 0.2,
        ELECTRONBEAMSIZEH       = 3.34281e-05,
        ELECTRONBEAMSIZEV       = 7.28139e-06,
        ELECTRONBEAMDIVERGENCEH = 4.51097e-06,
        ELECTRONBEAMDIVERGENCEV = 1.94034e-06,
        PERIODID    = 0.018,
        NPERIODS    = 111,
        KV          = 1.358,
        KH          = 0.0,
        KPHASE      = 0.0,
        DISTANCE    = 30.0,
        GAPH        = 0.01,
        GAPV        = 0.01,
        HSLITPOINTS = 48,
        VSLITPOINTS = 30,
        METHOD      = 1,
        USEEMITTANCES = 1,
    )

    period_m  = h5_parameters["PERIODID"]
    Ky        = h5_parameters["KV"]
    N_periods = h5_parameters["NPERIODS"]
    current_A = h5_parameters["ELECTRONCURRENT"]
    E_GeV     = h5_parameters["ELECTRONENERGY"]
    Z_m       = h5_parameters["DISTANCE"]

    gamma = 1e9 * E_GeV / (codata.m_e * codata.c**2 / codata.e)
    B0    = Ky / (0.9336 * period_m * 1e2)
    P_kim = (N_periods / 6) * codata.value('characteristic impedance of vacuum') * \
        current_A * codata.e * 2 * np.pi * codata.c * gamma**2 * Ky**2 / period_m

    print(f"K={Ky:.4f}  B0={B0:.3f} T  γ={gamma:.1f}")
    print(f"E1={harmonic_energy(1,Ky,period_m,gamma)/1e3:.3f} keV  P_kim={P_kim:.1f} W")

    # ------------------------------------------------------------------ #
    # Python calculation                                                   #
    # ------------------------------------------------------------------ #
    if calculate_python:
        x_m = np.linspace(-h5_parameters["GAPH"]/2, h5_parameters["GAPH"]/2,
                           2 * h5_parameters["HSLITPOINTS"])
        y_m = np.linspace(-h5_parameters["GAPV"]/2, h5_parameters["GAPV"]/2,
                           2 * h5_parameters["VSLITPOINTS"])
        XX, YY  = np.meshgrid(x_m, y_m)
        theta_x = XX / Z_m
        theta_y = YY / Z_m
        dOmega  = (theta_x[0,1]-theta_x[0,0]) * (theta_y[1,0]-theta_y[0,0])

        t0 = time.time()
        total_Wrad2, indiv_Wrad2 = power_density_all_harmonics(
            Ky, N_periods, period_m, gamma, current_A,
            theta_x, theta_y,
            n_harmonics=n_harmonics,
            p_max=None,              # adaptive — recommended
            zero_emittance=zero_emittance,
            sigma_x=h5_parameters["ELECTRONBEAMSIZEH"],
            sigma_y=h5_parameters["ELECTRONBEAMSIZEV"],
            sigma_xp=h5_parameters["ELECTRONBEAMDIVERGENCEH"],
            sigma_yp=h5_parameters["ELECTRONBEAMDIVERGENCEV"],
            distance_m=Z_m)
        print(f"Python: {time.time()-t0:.2f} s")

    # ------------------------------------------------------------------ #
    # Fortran URGENT                                                       #
    # ------------------------------------------------------------------ #
    if calculate_fortran > 0:
        U_X, U_Y, U_PD, U_PDH, U_En, U_FLUX = run_urgent_fortran(
            h5_parameters, n_harmonics=n_harmonics, zero_emittance=zero_emittance)

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #
    extent_py = [x_m[0]*1e3, x_m[-1]*1e3, y_m[0]*1e3, y_m[-1]*1e3] \
        if calculate_python else None
    extent_ft = [U_X[0], U_X[-1], U_Y[0], U_Y[-1]] \
        if calculate_fortran else None

    def make_panel(fig, axes, panels, extent, cbar_label):
        for ax, (title, data) in zip(axes.flat, panels):
            im = ax.imshow(data, extent=extent, origin='lower',
                           aspect='equal', cmap='inferno')
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
            plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046)
        fig.tight_layout()

    # 1. Python — Power density [W/mm²]
    if calculate_python == 1:
        factor_Wrad2_to_Wmm2 = 1e-6 / h5_parameters["DISTANCE"] ** 2

        fig1, ax1 = plt.subplots(2, 4, figsize=(16, 8))
        fig1.suptitle(f"PYTHON — Power density [W/rad²]  (zero-emittance)\n"
                      f"K={Ky:.3f}, N={N_periods}, E={E_GeV:.1f} GeV, I={current_A*1e3:.0f} mA")
        make_panel(fig1, ax1,
                   [('Total H1-%d'%n_harmonics, total_Wrad2 * factor_Wrad2_to_Wmm2)] +
                   [(f'H{n}', indiv_Wrad2[n] * factor_Wrad2_to_Wmm2) for n in range(7)],
                   extent_py, 'W/rad²')

    # 2. Python — Spectral flux [ph/s/0.1%bw/mrad²]
    if calculate_python == 2:

        # Spectral flux [ph/s/0.1%bw/rad²]
        indiv_flux_rad2 = np.zeros_like(indiv_Wrad2)
        for n in range(1, n_harmonics + 1):
            Ei = harmonic_energy(n, Ky, period_m, gamma)
            indiv_flux_rad2[n-1] = indiv_Wrad2[n-1] / (Ei * codata.e)

        # Spectral flux [ph/s/0.1%bw/mm²]
        indiv_flux_mm2 = indiv_flux_rad2 / h5_parameters["DISTANCE"] ** 2 * 1e-6
        total_flux_mm2 = indiv_flux_mm2.sum(axis=0)

        fig2, ax2 = plt.subplots(2, 4, figsize=(16, 8))
        fig2.suptitle(f"PYTHON — Spectral flux [ph/s/0.1%bw/mm²]\n"
                      f"K={Ky:.3f}, N={N_periods}, E={E_GeV:.1f} GeV, I={current_A*1e3:.0f} mA")
        make_panel(fig2, ax2,
                   [('Total H1-%d'%n_harmonics, total_flux_mm2)] +
                   [(f'H{n}', indiv_flux_mm2[n]) for n in range(7)],
                   extent_py, 'ph/s/0.1%bw/mrad²')

    # 3. Fortran — Power density [W/mm²]
    if calculate_fortran == 1:
        fig3, ax3 = plt.subplots(2, 4, figsize=(16, 8))
        fig3.suptitle(f"FORTRAN — Power density [W/mm²]\n"
                      f"K={Ky:.3f}, N={N_periods}, E={E_GeV:.1f} GeV, I={current_A*1e3:.0f} mA")
        make_panel(fig3, ax3,
                   [('Total H1-%d'%n_harmonics, U_PDH.sum(axis=0).T)] +
                   [(f'H{n+1}', U_PDH[n].T) for n in range(7)],
                   extent_ft, 'W/mm²')

    # 3. Fortran — Spectral flux [ph/s/0.1%bw/mm²]  (native Fortran output)
    if calculate_fortran == 2:
        fig4, ax4 = plt.subplots(2, 4, figsize=(16, 8))
        fig4.suptitle(f"FORTRAN — Spectral flux [ph/s/0.1%bw/mm²]\n"
                      f"K={Ky:.3f}, N={N_periods}, E={E_GeV:.1f} GeV, I={current_A*1e3:.0f} mA")
        make_panel(fig4, ax4,
                   [('Total H1-%d'%n_harmonics, U_FLUX.sum(axis=0).T)] +
                   [(f'H{n+1}', U_FLUX[n].T) for n in range(7)],
                   extent_ft, 'ph/s/0.1%bw/mm²')

    plt.show()