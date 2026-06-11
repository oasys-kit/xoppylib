#
# xoppy_calc_wspy: pure-python reimplementation of the WS code
#                  (wiggler spectrum using the Bessel function approximation)
#
# Reproduces the method and the results of WS v1.62 by R.J. Dejus (APS, ANL)
# as driven by xoppylib.xoppy_run_binaries.xoppy_calc_ws (i.e. WS MODE=4:
# flux spectrum through a pinhole), without calling the external binary.
#
# Method (K.J. Kim, AIP Conf. Proc. 184 (1989) p.583, Eq. 3.12):
#   The wiggler is treated as a sequence of 2N bending-magnet-like sources.
#   At a horizontal observation angle theta, radiation is emitted from the
#   two points per period where the electron trajectory points towards the
#   observer; there the local magnetic field is B(theta) = B0*sqrt(1-(g*theta/Ky)^2),
#   giving a local critical energy ecp = ec0*sqrt(1-(g*theta/Ky)^2).
#   The angular flux density is then the bending magnet formula:
#
#   d2F/(dtheta dpsi) = C * gamma^2 * (I/e) * (dE/E) * 2N * y^2 * (1+X^2)^2 *
#                       [ K_{2/3}(eta)^2 + X^2/(1+X^2) * K_{1/3}(eta)^2 ]
#
#   with X = gamma*psi, y = E/ecp(theta), eta = y*(1+X^2)^(3/2)/2,
#   and C = 3*alpha/(4*pi^2).
#   The flux through the pinhole is obtained by 2D trapezoidal integration
#   over the (theta, psi) grid (exploiting 4-fold symmetry when centered).
#
import numpy
import scipy

def xoppy_calc_wspy(
        ENERGY = 7.0,      # ring energy [GeV]
        CUR    = 100.0,    # ring current [mA]
        PERIOD = 8.5,      # wiggler period [cm]
        N      = 28.0,     # number of periods (0.5 for bending magnet)
        KX     = 0.0,      # horizontal deflection parameter (must be 0)
        KY     = 8.74,     # vertical-field deflection parameter
        EMIN   = 1000.0,   # minimum photon energy [eV]
        EMAX   = 100000.0, # maximum photon energy [eV]
        NEE    = 2000,     # number of energy points
        D      = 30.0,     # distance from source [m] (0 -> angular units)
        XPC    = 0.0,      # pinhole center X [mm] (or mrad if D=0)
        YPC    = 0.0,      # pinhole center Y [mm] (or mrad if D=0)
        XPS    = 2.0,      # pinhole full width X [mm] (or mrad if D=0)
        YPS    = 2.0,      # pinhole full height Y [mm] (or mrad if D=0)
        NXP    = 10,       # number of subdivisions of the pinhole in X
        NYP    = 10,       # number of subdivisions of the pinhole in Y
        output_file = "wspy.spec",
        verbose = True,
        return_file = True, # by default returns spec file name, if False returns arrays energy, flux, spectral_power, cumulated_power
        ):
    """
    Pure python version of xoppylib.xoppy_run_binaries.xoppy_calc_ws
    (WS in MODE=4: flux spectrum through a pinhole).

    Returns the name of the written spec file with 4 columns:
    Energy(eV), Flux(photons/s/0.1%bw), Spectral power(W/eV), Cumulated power(W)
    """

    if KX != 0.0:
        raise ValueError("KX must be 0.0 (elliptical wiggler not implemented in WS)")

    # ------------------------------------------------------------------
    # physical constants (updated from used in ws.f, Physics Today Aug. 1990)
    # ------------------------------------------------------------------
    C_LIGHT = scipy.constants.c                                   # 2.99792458e8        # speed of light [m/s]
    ME      = scipy.constants.m_e                                 # 9.1093897e-31       # electron rest mass [kg]
    EC      = scipy.constants.e                                   # 1.60217733e-19      # elementary charge [C]
    MEE     = ME * C_LIGHT**2 / EC * 1e-6 # 0.51099906          # electron rest mass [MeV]
    HBAR    = scipy.constants.hbar                                # 1.05457266e-34      # hbar [Js]
    EPSZ    = scipy.constants.epsilon_0                           # 8.854187817e-12     # permittivity of vacuum [F/m]
    PI      = numpy.pi
    BW      = 1.0e-3                                     # 0.1% bandwidth
    FINE_STRUCTURE_CONST = scipy.constants.fine_structure         # EC * EC / (4.0 * PI * EPSZ * HBAR * C_LIGHT)

    CL1 = EC / (2.0 * PI * ME * C_LIGHT) * 1e-2          # B0 = Ky/(CL1*period[cm]) -> 0.0934..
    CL2 = 1.5 / MEE**2 * HBAR / ME * 1e6                 # ec0 = CL2*E^2*B0 -> 665.02..
    CL3 = 3.0 * FINE_STRUCTURE_CONST / (4.0 * PI**2)     # 5.545e-4
    PTOT_FAC = PI / 3.0 * EC / EPSZ / MEE**2 * 1e6       # 0.07257
    PD_FAC   = 21.0 / (16.0 * PI * MEE**2) * PTOT_FAC    # 0.11611

    # ------------------------------------------------------------------
    # derived machine/device quantities
    # ------------------------------------------------------------------
    H = scipy.constants.h # 6.6260755e-34                          # Planck constant [Js]
    gamma  = ENERGY / MEE * 1e3
    k2     = KX * KX + KY * KY
    k3     = 1.0 + k2 / 2.0
    lamdar = PERIOD * 1e8 / (2.0 * gamma**2)   # reduced wavelength [A]
    er     = (H * C_LIGHT / EC * 1e10) / lamdar  # reduced energy [eV]
    e1z    = er / k3                           # first harmonic on axis [eV]
    b0    = KY / (CL1 * PERIOD)                # peak field [T]
    ec0   = CL2 * ENERGY**2 * b0               # critical energy on axis [eV]

    # total power and on-axis power density (informative, as in WS header)
    kk = KX + KY  # one of them is zero
    gk = kk * (kk**6 + 24.0 * kk**4 / 7.0 + 4.0 * kk * kk + 16.0 / 7.0) / (1.0 + kk * kk)**3.5
    ptot = PTOT_FAC * N * k2 * ENERGY**2 * CUR * 1e-3 / (PERIOD * 1e-2)       # [W]
    pd   = PD_FAC * N * kk * gk * ENERGY**4 * CUR * 1e-3 / (PERIOD * 1e-2)    # [W/mrad^2]

    # ------------------------------------------------------------------
    # geometry: pinhole grid (positions [mm] and angles [rad])
    # ------------------------------------------------------------------
    if D == 0.0:   # angular units: pinhole values given in mrad
        d = 1.0
    else:
        d = D

    # spectral distribution prefactor: ph/s/mrad^2/0.1%bw (or /mm^2 at distance d)
    facs = CL3 * gamma**2 * BW * CUR * 1e-3 / EC * 1e-6 * 2.0 * N / d**2

    if XPC == 0.0 and YPC == 0.0:   # pinhole centered on axis: use 4-fold symmetry
        fac = 4.0
        xpmin, ypmin = 0.0, 0.0
        dxp = XPS / 2.0 / NXP if NXP > 0 else 0.0
        dyp = YPS / 2.0 / NYP if NYP > 0 else 0.0
    else:
        fac = 1.0
        xpmin, ypmin = XPC - XPS / 2.0, YPC - YPS / 2.0
        dxp = XPS / NXP if NXP > 0 else 0.0
        dyp = YPS / NYP if NYP > 0 else 0.0

    xp = xpmin + numpy.arange(NXP + 1) * dxp    # [mm] (or mrad)
    yp = ypmin + numpy.arange(NYP + 1) * dyp
    cx = xp * 1e-3 / d                          # horizontal angles theta [rad]
    cy = yp * 1e-3 / d                          # vertical angles psi [rad]

    # ------------------------------------------------------------------
    # energy grid
    # ------------------------------------------------------------------
    e = numpy.linspace(EMIN, EMAX, NEE)
    estep = (EMAX - EMIN) / (NEE - 1)

    # ------------------------------------------------------------------
    # angular flux density on the grid, for all energies (vectorized)
    # ------------------------------------------------------------------
    xg = gamma * cx                       # gamma*theta, shape (nx,)
    yg = gamma * cy                       # gamma*psi,   shape (ny,)

    vp = numpy.abs(xg) / KY               # relative position within the pole
    valid = vp <= numpy.sqrt(1.0 - 1e-6)  # no emission towards angles > Ky/gamma
    ecp = numpy.where(valid, ec0 * numpy.sqrt(numpy.maximum(1.0 - vp**2, 0.0)), 1.0)  # local Ec [eV]

    yg2  = yg * yg
    yg1  = 1.0 + yg2                      # 1 + (gamma*psi)^2
    cpi  = yg2 / yg1                      # weight of the pi component

    # broadcast to shape (ne, nx, ny)
    y   = e[:, None, None] / ecp[None, :, None]                # E/Ec(theta)
    eta = 0.5 * y * (yg1[None, None, :])**1.5

    with numpy.errstate(over='ignore', under='ignore'):
        k23v = scipy.special.kv(2.0 / 3.0, eta)
        k13v = scipy.special.kv(1.0 / 3.0, eta)
    k23v = numpy.nan_to_num(k23v, nan=0.0, posinf=0.0)
    k13v = numpy.nan_to_num(k13v, nan=0.0, posinf=0.0)

    fc  = facs * yg1**2                                        # shape (ny,)
    ra0 = y**2 * fc[None, None, :] * (k23v**2 + cpi[None, None, :] * k13v**2)
    ra0 *= valid[None, :, None]                                # zero beyond gamma*theta=Ky

    # ------------------------------------------------------------------
    # 2D trapezoidal integration over the pinhole -> flux [ph/s/0.1%bw]
    # ------------------------------------------------------------------
    wx = numpy.ones(NXP + 1); wx[0] = wx[-1] = 0.5
    wy = numpy.ones(NYP + 1); wy[0] = wy[-1] = 0.5
    flux = fac * dxp * dyp * numpy.einsum('eij,i,j->e', ra0, wx, wy)

    # ------------------------------------------------------------------
    # derived columns and integrated quantities (as in xoppylib/WS print_out)
    # ------------------------------------------------------------------
    spectral_power = flux * EC * 1e3                  # [W/eV]

    cumulated_power = scipy.integrate.cumulative_trapezoid(spectral_power, e, initial=0) # [W]
    print(type(cumulated_power), cumulated_power.shape)
    # cumulated_power = numpy.cumsum(spectral_power) * estep   # [W]
    # print(type(cumulated_power), cumulated_power.shape)

    we = numpy.ones(NEE); we[0] = we[-1] = 0.5
    # tot_flux  = numpy.sum(we * flux / e) / BW * estep        # [ph/s]
    # tot_power = numpy.sum(we * spectral_power) * estep       # [W]
    tot_flux  = numpy.trapezoid(we * flux / BW / e, e)   # [ph/s]
    tot_power = numpy.trapezoid(we * spectral_power, e)  # [W]

    if verbose:
        print("Inside xoppy_calc_wspy. ")
        print("Wiggler Flux: B0 = %.3f T, Ec0 = %.3f keV" % (b0, ec0 * 1e-3))
        print("              E1 = %.2f eV, Ptot = %.1f W, Pd(on-axis) = %.1f W/mrad^2" % (e1z, ptot, pd))
        print("Flux through %g x %g pinhole at (%g, %g) @ %g m:" % (XPS, YPS, XPC, YPC, D))
        print("Integrated flux:  %.3e ph/s" % tot_flux)
        print("Integrated power: %.2f W" % tot_power)

    # ------------------------------------------------------------------
    # write spec file (same structure xoppy_calc_ws produces)
    # ------------------------------------------------------------------
    with open(output_file, "w") as f:
        f.write("#F %s\n" % output_file)
        f.write("\n")
        f.write("#S 1 ws (python) results\n")
        f.write("#UD B0 = %f T, Ec0 = %f eV, E1 = %f eV\n" % (b0, ec0, e1z))
        f.write("#UD Ptot = %f W, Pd = %f W/mrad^2\n" % (ptot, pd))
        f.write("#UD Integrated flux = %g ph/s, Integrated power = %f W\n" % (tot_flux, tot_power))
        f.write("#N 4\n")
        f.write("#L  Energy(eV)  Flux(photons/s/0.1%bw)  Spectral power(W/eV)  Cumulated power(W)\n")
        for i in range(NEE):
            f.write("%f  %g  %g  %g \n" % (e[i], flux[i], spectral_power[i], cumulated_power[i]))
    if verbose:
        print("File written to disk: %s" % output_file)

    if return_file:
        return output_file
    else:
        return e, flux, spectral_power, cumulated_power

if __name__ == "__main__":
    args = {
        'ENERGY' : 6.0,
        'CUR'    : 200.0,
        'PERIOD' : 15.0,
        'N'      : 10.0,
        'KX'     : 0.0,
        'KY'     : 22.591,
        'EMIN'   : 100.0,
        'EMAX'   : 200000.0,
        'NEE'    : 500,
        'D'      : 23.0,
        'XPC'    : 0.0,
        'YPC'    : 0.0,
        'XPS'    : 30.0,
        'YPS'    : 10.0,
        'NXP'    : 30,
        'NYP'    : 30,
        }

    # python version
    energy, flux, spectral_power, cumulated_power = xoppy_calc_wspy( **args, return_file=False  )


    # fortran version
    out_file_f = xoppy_calc_wspy( **args )

    # data to pass to power
    data_f = numpy.loadtxt(out_file_f)
    energy_f = data_f[:, 0]
    flux_f = data_f[:, 1]
    spectral_power_f = data_f[:, 2]
    cumulated_power_f = data_f[:, 3]


    #
    # example plot
    #
    if True:
        from srxraylib.plot.gol import plot

        plot(energy, flux, energy_f, flux_f, legend=['WS fortran', 'WS python'],
             xtitle="Photon energy [eV]", ytitle="Flux [photons/s/0.1%bw]", title="WS Flux",
             xlog=True, ylog=True, grid=1, show=False)
        plot(energy, spectral_power, energy_f, spectral_power_f, legend=['WS fortran', 'WS python'],
             xtitle="Photon energy [eV]", ytitle="Power [W/eV]", title="WS Spectral Power",
             xlog=True, ylog=True, grid=1, show=False)
        plot(energy, cumulated_power, energy_f, cumulated_power_f, legend=['WS fortran', 'WS python'],
             xtitle="Photon energy [eV]", ytitle="Cumulated Power [W]", title="WS Cumulated Power",
             xlog=False, ylog=False, grid=1, show=True)


