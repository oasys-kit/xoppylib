import numpy
from srxraylib.sources import srfunc



from srxraylib.util.h5_simple_writer import H5SimpleWriter

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import scipy.constants as codata

from xoppylib.fit_gaussian2d import fit_gaussian2d, info_params, twoD_Gaussian

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

# copied from OASYS1
def get_fwhm(histogram, bins, ret0=None):
    fwhm = ret0
    quote = ret0
    coordinates = None

    if histogram.size > 1:
        quote = numpy.max(histogram)*0.5
        cursor = numpy.where(histogram >= quote)

        if histogram[cursor].size > 1:
            bin_size    = bins[1]-bins[0]
            fwhm        = bin_size*(cursor[0][-1]-cursor[0][0])
            coordinates = (bins[cursor[0][0]], bins[cursor[0][-1]])

    return fwhm, quote, coordinates

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


def xoppy_calc_bm(MACHINE_NAME="ESRF bending magnet",RB_CHOICE=0,MACHINE_R_M=25.0,BFIELD_T=0.8,\
                  BEAM_ENERGY_GEV=6.04,CURRENT_A=0.1,HOR_DIV_MRAD=1.0,VER_DIV=0,\
                  PHOT_ENERGY_MIN=100.0,PHOT_ENERGY_MAX=100000.0,NPOINTS=500,LOG_CHOICE=1,\
                  PSI_MRAD_PLOT=1.0,PSI_MIN=-1.0,PSI_MAX=1.0,PSI_NPOINTS=500,TYPE_CALC=0,FILE_DUMP=0):

    # electron energy in GeV
    codata_mee = 1e-6 * codata.m_e * codata.c**2 / codata.e  # electron mass in meV
    m2ev = codata.c * codata.h / codata.e
    gamma = BEAM_ENERGY_GEV * 1e3 / codata_mee

    r_m = MACHINE_R_M      # magnetic radius in m
    if RB_CHOICE == 1:
        r_m = codata.m_e * codata.c / codata.e / BFIELD_T * numpy.sqrt(gamma * gamma - 1)

    # calculate critical energy in eV
    ec_m = 4.0*numpy.pi*r_m/3.0/numpy.power(gamma,3) # wavelength in m
    ec_ev = m2ev / ec_m

    print("\nLorent factor gamma: %g" % (gamma))
    print("\nMagnetic radius: %g m" % (r_m))
    print("\nCritical energy: %g eV; wavelength: %g A" % (ec_ev, ec_m * 1e10))


    fm = None
    a = None
    energy_ev = None

    if TYPE_CALC == 0:
        if LOG_CHOICE == 0:
            energy_ev = numpy.linspace(PHOT_ENERGY_MIN,PHOT_ENERGY_MAX,NPOINTS) # photon energy grid
        else:
            energy_ev = numpy.logspace(numpy.log10(PHOT_ENERGY_MIN),numpy.log10(PHOT_ENERGY_MAX),NPOINTS) # photon energy grid

        a5 = srfunc.sync_ene(VER_DIV, energy_ev, ec_ev=ec_ev, polarization=0, \
                             e_gev=BEAM_ENERGY_GEV, i_a=CURRENT_A, hdiv_mrad=HOR_DIV_MRAD, \
                             psi_min=PSI_MIN, psi_max=PSI_MAX, psi_npoints=PSI_NPOINTS)

        a5par = srfunc.sync_ene(VER_DIV, energy_ev, ec_ev=ec_ev, polarization=1, \
                                e_gev=BEAM_ENERGY_GEV, i_a=CURRENT_A, hdiv_mrad=HOR_DIV_MRAD, \
                                psi_min=PSI_MIN, psi_max=PSI_MAX, psi_npoints=PSI_NPOINTS)

        a5per = srfunc.sync_ene(VER_DIV, energy_ev, ec_ev=ec_ev, polarization=2, \
                                e_gev=BEAM_ENERGY_GEV, i_a=CURRENT_A, hdiv_mrad=HOR_DIV_MRAD, \
                                psi_min=PSI_MIN, psi_max=PSI_MAX, psi_npoints=PSI_NPOINTS)

        if VER_DIV == 0:
            coltitles=['Photon Energy [eV]','Photon Wavelength [A]','E/Ec','Flux_spol/Flux_total','Flux_ppol/Flux_total','Flux[Phot/s/0.1%bw]','Power[W/eV]','CumulatedPower[W]']
            title='integrated in Psi,'
        if VER_DIV == 1:
            coltitles=['Photon Energy [eV]','Photon Wavelength [A]','E/Ec','Flux_spol/Flux_total','Flux_ppol/Flux_total','Flux[Phot/s/0.1%bw/mrad(Psi)]','Power[W/eV/mrad(Psi)]','CumulatedPower[W/mrad(Psi)]']
            title='at Psi=0,'
        if VER_DIV == 2:
            coltitles=['Photon Energy [eV]','Photon Wavelength [A]','E/Ec','Flux_spol/Flux_total','Flux_ppol/Flux_total','Flux[Phot/s/0.1%bw]','Power[W/eV]','CumulatedPower[W]']
            title='in Psi=[%e,%e]'%(PSI_MIN,PSI_MAX)
        if VER_DIV == 3:
            coltitles=['Photon Energy [eV]','Photon Wavelength [A]','E/Ec','Flux_spol/Flux_total','Flux_ppol/Flux_total','Flux[Phot/s/0.1%bw/mrad(Psi)]','Power[W/eV/mrad(Psi)]','CumulatedPower[W/mrad(Psi)]']
            title='at Psi=%e mrad'%(PSI_MIN)

        a6=numpy.zeros((8,len(energy_ev)))
        a1 = energy_ev
        spectral_power = numpy.array(a5) * 1e3 * codata.e

        # calculate cumulated power
        if VER_DIV in [0,2]:
            cumulated_power_unit = 'W'
        else:
            cumulated_power_unit = 'W/mrad(Psi)'
        if LOG_CHOICE == 0:
            cumulated_power = spectral_power.cumsum() * numpy.abs(energy_ev[0] - energy_ev[1])
            # print("\nPower from integral of spectrum (sum rule): %8.3f %s" % (cumulated_power[-1], cumulated_power_unit))
            print("\nPower from integral of spectrum (trapz rule): %8.3f %s" % (numpy.trapz(spectral_power, energy_ev), cumulated_power_unit))
        else:
            cumulated_power = numpy.zeros_like(energy_ev)
            for i in range(1, energy_ev.size):
                cumulated_power[i] = cumulated_power[i-1] + numpy.abs( spectral_power[i] * (energy_ev[i] - energy_ev[i-1]))
            print("\nPower from integral of spectrum (sum rule, WARNING: energy in LOG): %8.3f %s" % (cumulated_power[-1], cumulated_power_unit))

        a6[0, :] = (a1)
        a6[1, :] = m2ev * 1e10 / (a1)
        a6[2, :] = (a1)/ec_ev # E/Ec
        a6[3, :] = numpy.array(a5par)/numpy.array(a5)
        a6[4, :] = numpy.array(a5per)/numpy.array(a5)
        a6[5, :] = numpy.array(a5)
        a6[6, :] = spectral_power
        a6[7, :] = cumulated_power

    if TYPE_CALC == 1:  # angular distributions over over all energies
        angle_mrad = numpy.linspace(-PSI_MRAD_PLOT, +PSI_MRAD_PLOT,NPOINTS) # angle grid

        a6 = numpy.zeros((6,NPOINTS))
        a6[0,:] = angle_mrad # angle in mrad
        a6[1,:] = angle_mrad*gamma/1e3 # Psi[rad]*Gamma
        a6[2,:] = srfunc.sync_f(angle_mrad * gamma / 1e3)
        a6[3,:] = srfunc.sync_f(angle_mrad * gamma / 1e3, polarization=1)
        a6[4,:] = srfunc.sync_f(angle_mrad * gamma / 1e3, polarization=2)
        a6[5,:] = srfunc.sync_ang(0, angle_mrad, i_a=CURRENT_A, hdiv_mrad=HOR_DIV_MRAD, e_gev=BEAM_ENERGY_GEV, r_m=r_m)

        coltitles=['Psi[mrad]','Psi[rad]*Gamma','F','F s-pol','F p-pol','Power[Watts/mrad(Psi)]']

    if TYPE_CALC == 2:  # angular distributions at a single energy
        angle_mrad = numpy.linspace(-PSI_MRAD_PLOT, +PSI_MRAD_PLOT,NPOINTS) # angle grid

        a6 = numpy.zeros((7,NPOINTS))
        a6[0,:] = angle_mrad # angle in mrad
        a6[1,:] = angle_mrad*gamma/1e3 # Psi[rad]*Gamma
        a6[2,:] = srfunc.sync_f(angle_mrad * gamma / 1e3)
        a6[3,:] = srfunc.sync_f(angle_mrad * gamma / 1e3, polarization=1)
        a6[4,:] = srfunc.sync_f(angle_mrad * gamma / 1e3, polarization=2)
        tmp = srfunc.sync_ang(1, angle_mrad, energy=PHOT_ENERGY_MIN, i_a=CURRENT_A, hdiv_mrad=HOR_DIV_MRAD, e_gev=BEAM_ENERGY_GEV, ec_ev=ec_ev)
        tmp.shape = -1
        a6[5,:] = tmp
        a6[6,:] = a6[5,:] * codata.e * 1e3

        coltitles=['Psi[mrad]','Psi[rad]*Gamma','F','F s-pol','F p-pol','Flux[Ph/sec/0.1%bw/mrad(Psi)]','Power[Watts/eV/mrad(Psi)]']

    if TYPE_CALC == 3:  # angular,energy distributions flux
        angle_mrad = numpy.linspace(-PSI_MRAD_PLOT, +PSI_MRAD_PLOT,NPOINTS) # angle grid

        if LOG_CHOICE == 0:
            energy_ev = numpy.linspace(PHOT_ENERGY_MIN,PHOT_ENERGY_MAX,NPOINTS) # photon energy grid
        else:
            energy_ev = numpy.logspace(numpy.log10(PHOT_ENERGY_MIN),numpy.log10(PHOT_ENERGY_MAX),NPOINTS) # photon energy grid

        # fm[angle,energy]
        fm = srfunc.sync_ene(4, energy_ev, ec_ev=ec_ev, e_gev=BEAM_ENERGY_GEV, i_a=CURRENT_A, \
                                      hdiv_mrad=HOR_DIV_MRAD, psi_min=PSI_MIN, psi_max=PSI_MAX, psi_npoints=PSI_NPOINTS)

        a = numpy.linspace(PSI_MIN,PSI_MAX,PSI_NPOINTS)

        a6 = numpy.zeros((4,len(a)*len(energy_ev)))
        ij = -1
        for i in range(len(a)):
            for j in range(len(energy_ev)):
                ij += 1
                a6[0,ij] = a[i]
                a6[1,ij] = energy_ev[j]
                a6[2,ij] = fm[i,j] * codata.e * 1e3
                a6[3,ij] = fm[i,j]

        coltitles=['Psi [mrad]','Photon Energy [eV]','Power [Watts/eV/mrad(Psi)]','Flux [Ph/sec/0.1%bw/mrad(Psi)]']

    # write spec file
    ncol = len(coltitles)
    npoints = len(a6[0,:])

    if FILE_DUMP:
        outFile = "bm.spec"
        f = open(outFile,"w")
        f.write("#F "+outFile+"\n")
        f.write("\n")
        f.write("#S 1 bm results\n")
        f.write("#N %d\n"%(ncol))
        f.write("#L")
        for i in range(ncol):
            f.write("  "+coltitles[i])
        f.write("\n")

        for i in range(npoints):
                f.write((" %e "*ncol+"\n")%(tuple(a6[:,i].tolist())))
        f.close()
        print("File written to disk: " + outFile)

    # if TYPE_CALC == 0:
    #     if LOG_CHOICE == 0:
    #         print("\nPower from integral of spectrum: %15.3f W"%(a5.sum() * 1e3*codata.e * (energy_ev[1]-energy_ev[0])))

    return a6.T, fm, a, energy_ev


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

def xoppy_calc_wigg(FIELD=0,NPERIODS=12,ULAMBDA=0.125,K=14.0,ENERGY=6.04,PHOT_ENERGY_MIN=100.0,\
                        PHOT_ENERGY_MAX=100100.0,NPOINTS=100,NTRAJPOINTS=101,CURRENT=200.0,FILE="?"):

    print("Inside xoppy_calc_wigg. ")

    outFileTraj = "xwiggler_traj.spec"
    outFile = "xwiggler.spec"

    if FIELD == 0:
        t0,p = srfunc.wiggler_trajectory(b_from=0, nPer=NPERIODS, nTrajPoints=NTRAJPOINTS, \
                                         ener_gev=ENERGY, per=ULAMBDA, kValue=K, \
                                         trajFile=outFileTraj)
    if FIELD == 1:
        # magnetic field from B(s) map
        t0,p = srfunc.wiggler_trajectory(b_from=1, nPer=NPERIODS, nTrajPoints=NTRAJPOINTS, \
                                         ener_gev=ENERGY, inData=FILE, trajFile=outFileTraj)
    if FIELD == 2:
        # magnetic field from harmonics
        # hh = srfunc.wiggler_harmonics(b_t,Nh=41,fileOutH="tmp.h")
        t0, p = srfunc.wiggler_trajectory(b_from=2, nPer=NPERIODS, nTrajPoints=NTRAJPOINTS, \
                                         ener_gev=ENERGY, per=ULAMBDA, inData="", trajFile=outFileTraj)

    #
    # now spectra
    #
    e, f0, p0 = srfunc.wiggler_spectrum(t0, enerMin=PHOT_ENERGY_MIN, enerMax=PHOT_ENERGY_MAX, nPoints=NPOINTS, \
                                    electronCurrent=CURRENT*1e-3, outFile=outFile, elliptical=False)

    try:
        cumulated_power = p0.cumsum() * numpy.abs(e[0] - e[1])
    except:
        cumulated_power = 0.0

    print("\nPower from integral of spectrum (sum rule): %8.3f W" % (cumulated_power[-1]))
    return e, f0, p0 , cumulated_power, t0, p

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

def xoppy_calc_wiggler_on_aperture(FIELD=0,
                                   NPERIODS=12,
                                   ULAMBDA=0.125,
                                   K=14.0,
                                   ENERGY=6.04,
                                   PHOT_ENERGY_MIN=100.0,
                                   PHOT_ENERGY_MAX=100100.0,
                                   NPOINTS=100,
                                   NTRAJPOINTS=101,
                                   CURRENT=200.0,
                                   FILE="?",
                                   SLIT_FLAG=0,
                                   SLIT_D=30.0,
                                   SLIT_NY=100,
                                   SLIT_WIDTH_H_MM=1.0,
                                   SLIT_HEIGHT_V_MM=1.0,
                                   SLIT_CENTER_H_MM=0.0,
                                   SLIT_CENTER_V_MM=0.0,
                                   SHIFT_X_FLAG=0,
                                   SHIFT_X_VALUE=0.0,
                                   SHIFT_BETAX_FLAG=0,
                                   SHIFT_BETAX_VALUE=0.0,
                                   TRAJ_RESAMPLING_FACTOR=1e4,
                                   SLIT_POINTS_FACTOR=1,
                                   ):

    print("Inside xoppy_calc_wiggler_on_aperture. ")

    outFileTraj = "xwiggler_traj.spec"
    outFile = "xwiggler.spec"

    if FIELD == 0:
        t0,p = srfunc.wiggler_trajectory(b_from=0, nPer=NPERIODS, nTrajPoints=NTRAJPOINTS, \
                                         ener_gev=ENERGY, per=ULAMBDA, kValue=K, \
                                         trajFile=outFileTraj)
    if FIELD == 1:
        # magnetic field from B(s) map
        t0,p = srfunc.wiggler_trajectory(b_from=1, nPer=NPERIODS, nTrajPoints=NTRAJPOINTS, \
                                         ener_gev=ENERGY, inData=FILE, trajFile=outFileTraj,\
                                         shift_x_flag = SHIFT_X_FLAG, \
                                         shift_x_value = SHIFT_X_VALUE, \
                                         shift_betax_flag = SHIFT_BETAX_FLAG, \
                                         shift_betax_value = SHIFT_BETAX_VALUE)
    if FIELD == 2:
        # magnetic field from harmonics
        # hh = srfunc.wiggler_harmonics(b_t,Nh=41,fileOutH="tmp.h")
        t0, p = srfunc.wiggler_trajectory(b_from=2, nPer=NPERIODS, nTrajPoints=NTRAJPOINTS, \
                                         ener_gev=ENERGY, per=ULAMBDA, inData="", trajFile=outFileTraj)

    #
    # now spectra
    #
    if SLIT_FLAG == 0:
        e, f0, p0 = srfunc.wiggler_spectrum(t0, enerMin=PHOT_ENERGY_MIN, enerMax=PHOT_ENERGY_MAX, nPoints=NPOINTS, \
                                        electronCurrent=CURRENT*1e-3, outFile=outFile, elliptical=False)
    else:
        e, f0, p0 = srfunc.wiggler_spectrum_on_aperture(t0, enerMin=PHOT_ENERGY_MIN, enerMax=PHOT_ENERGY_MAX, nPoints=NPOINTS, \
                                            electronCurrent=CURRENT * 1e-3, outFile=outFile, elliptical=False,
                                            psi_min=(-SLIT_HEIGHT_V_MM / 2 + SLIT_CENTER_V_MM) / SLIT_D, # mrad
                                            psi_max=(SLIT_HEIGHT_V_MM / 2 + SLIT_CENTER_V_MM) / SLIT_D,  # mrad
                                            psi_npoints=SLIT_NY,
                                            theta_min=(-SLIT_WIDTH_H_MM / 2 + SLIT_CENTER_H_MM) / SLIT_D,  # mrad
                                            theta_max=(SLIT_WIDTH_H_MM / 2 + SLIT_CENTER_H_MM) / SLIT_D,  # mrad
                                            traj_res_fac=TRAJ_RESAMPLING_FACTOR,
                                            slit_points_factor=SLIT_POINTS_FACTOR
                                            )

    try:
        cumulated_power = p0.cumsum() * numpy.abs(e[0] - e[1])
    except:
        cumulated_power = 0.0

    print("\n Flux peak: %.2e Ph/s/0.1%%BW at the photon energy: %.2f eV" % (max(f0), e[numpy.argmax(f0)]))    
    print("\nPower from integral of spectrum (sum rule): %8.3f W" % (cumulated_power[-1]))
    return e, f0, p0 , cumulated_power, t0, p

def trapezoidal_rule_2d_1darrays(data2D,h=None,v=None):
    if h is None:
        h = numpy.arange(data2D.shape[0])
    if v is None:
        v = numpy.arange(data2D.shape[1])
    totPower2 = numpy.trapz(data2D, v, axis=1)
    totPower2 = numpy.trapz(totPower2, h, axis=0)
    return totPower2


#
#
#
def xoppy_calc_wiggler_radiation(
        ELECTRONENERGY           = 3.0,
        ELECTRONCURRENT          = 0.1,
        PERIODID                 = 0.120,
        NPERIODS                 = 37.0,
        KV                       = 22.416,
        DISTANCE                 = 30.0,
        HSLITPOINTS              = 500,
        VSLITPOINTS              = 500,
        PHOTONENERGYMIN          = 100.0,
        PHOTONENERGYMAX          = 100100.0,
        PHOTONENERGYPOINTS       = 101,
        NTRAJPOINTS              = 1001,
        FIELD                    = 0,
        FILE                     = "/Users/srio/Oasys/Bsin.txt",
        POLARIZATION             = 0, # 0=total, 1=parallel (s), 2=perpendicular (p)
        SHIFT_X_FLAG             = 0,
        SHIFT_X_VALUE            = 0.0,
        SHIFT_BETAX_FLAG         = 0,
        SHIFT_BETAX_VALUE        = 0.0,
        CONVOLUTION              = 1,
        PASSEPARTOUT             = 3.0,
        h5_file                  = "wiggler_radiation.h5",
        h5_entry_name            = "XOPPY_RADIATION",
        h5_initialize            = True,
        h5_parameters            = None,
        do_plot                  = False,
        ):


    # calculate wiggler trajectory
    if FIELD == 0:
        (traj, pars) = srfunc.wiggler_trajectory(
        b_from = 0,
        inData = "",
        nPer = int(NPERIODS), #37,
        nTrajPoints = NTRAJPOINTS,
        ener_gev = ELECTRONENERGY,
        per = PERIODID,
        kValue = KV,
        trajFile = "",
        shift_x_flag = SHIFT_X_FLAG,
        shift_x_value = SHIFT_X_VALUE,
        shift_betax_flag = SHIFT_BETAX_FLAG,
        shift_betax_value = SHIFT_BETAX_VALUE)
    if FIELD == 1:
        # magnetic field from B(s) map
        (traj, pars) = srfunc.wiggler_trajectory(
            b_from=1,
            nPer=1,
            nTrajPoints=NTRAJPOINTS,
            ener_gev=ELECTRONENERGY,
            inData=FILE,
            trajFile="",
            shift_x_flag = SHIFT_X_FLAG,
            shift_x_value = SHIFT_X_VALUE,
            shift_betax_flag = SHIFT_BETAX_FLAG,
            shift_betax_value = SHIFT_BETAX_VALUE)
    if FIELD == 2:
        raise("Not implemented")


    energy, flux, power = srfunc.wiggler_spectrum(traj,
        enerMin = PHOTONENERGYMIN,
        enerMax = PHOTONENERGYMAX,
        nPoints = PHOTONENERGYPOINTS,
        electronCurrent = ELECTRONCURRENT,
        outFile = "",
        elliptical = False,
        polarization = POLARIZATION)

    try:
        cumulated_power = power.cumsum() * numpy.abs(energy[0] - energy[1])
    except:
        cumulated_power = 0.0
    print("\nPower from integral of spectrum (sum rule): %8.3f W" % (cumulated_power[-1]))


    try:
        cumulated_power = cumtrapz(power, energy, initial=0)
    except:
        cumulated_power = 0.0
    print("Power from integral of spectrum (trapezoid rule): %8.3f W" % (cumulated_power[-1]))


    codata_mee = 1e-6 * codata.m_e * codata.c ** 2 / codata.e # electron mass in meV
    gamma = ELECTRONENERGY * 1e3 / codata_mee

    Y = traj[1, :].copy()
    divX = traj[3,:].copy()
    By = traj[7, :].copy()

    # rho = (1e9 / codata.c) * ELECTRONENERGY / By
    # Ec0 = 3 * codata.h * codata.c * gamma**3 / (4 * numpy.pi * rho) / codata.e
    # Ec = 665.0 * ELECTRONENERGY**2 * numpy.abs(By)
    # Ecmax = 665.0 * ELECTRONENERGY** 2 * (numpy.abs(By)).max()
    coeff = 3 / (4 * numpy.pi) * codata.h * codata.c**2 / codata_mee ** 3 / codata.e # ~665.0
    Ec = coeff * ELECTRONENERGY ** 2 * numpy.abs(By)
    Ecmax = coeff * ELECTRONENERGY ** 2 * (numpy.abs(By)).max()

    # approx formula for divergence (first formula in pag 43 of Tanaka's paper)
    sigmaBp = 0.597 / gamma * numpy.sqrt(Ecmax / PHOTONENERGYMIN)


    # we use vertical interval 6*sigmaBp and horizontal interval = vertical + trajectory interval

    divXX = numpy.linspace(divX.min() - PASSEPARTOUT * sigmaBp, divX.max() + PASSEPARTOUT * sigmaBp, HSLITPOINTS)

    divZZ = numpy.linspace(-PASSEPARTOUT * sigmaBp, PASSEPARTOUT * sigmaBp, VSLITPOINTS)

    e = numpy.linspace(PHOTONENERGYMIN, PHOTONENERGYMAX, PHOTONENERGYPOINTS)

    p = numpy.zeros( (PHOTONENERGYPOINTS, HSLITPOINTS, VSLITPOINTS) )


    for i in range(e.size):
        Ephoton = e[i]


        # vertical divergence
        intensity = srfunc.sync_g1(Ephoton / Ec, polarization=POLARIZATION)

        Ecmean = (Ec * intensity).sum() / intensity.sum()

        fluxDivZZ = srfunc.sync_ang(1, divZZ * 1e3, polarization=POLARIZATION,
               e_gev=ELECTRONENERGY, i_a=ELECTRONCURRENT, hdiv_mrad=1.0, energy=Ephoton, ec_ev=Ecmean)

        if do_plot:
            from srxraylib.plot.gol import plot
            plot(divZZ, fluxDivZZ, title="min intensity %f" % fluxDivZZ.min(), xtitle="divZ", ytitle="fluxDivZZ", show=1)


        # horizontal divergence after Tanaka
        if False:
            e_over_ec = Ephoton / Ecmax
            uudlim = 1.0 / gamma
            uud = numpy.linspace(-uudlim*0.99, uudlim*0.99, divX.size)
            uu  = e_over_ec / numpy.sqrt(1 - gamma**2 * uud**2)
            plot(uud, 2 * numpy.pi / numpy.sqrt(3) * srfunc.sync_g1(uu))

        # horizontal divergence
        # intensity = srfunc.sync_g1(Ephoton / Ec, polarization=POLARIZATION)
        intensity_interpolated = interpolate_multivalued_function(divX, intensity, divXX, Y, )

        if CONVOLUTION: # do always convolution!
            intensity_interpolated.shape = -1
            divXX_window = divXX[-1] - divXX[0]
            divXXCC = numpy.linspace( -0.5 * divXX_window, 0.5 * divXX_window, divXX.size)
            fluxDivZZCC = srfunc.sync_ang(1, divXXCC * 1e3, polarization=POLARIZATION,
                                        e_gev=ELECTRONENERGY, i_a=ELECTRONCURRENT, hdiv_mrad=1.0,
                                        energy=Ephoton, ec_ev=Ecmax)
            fluxDivZZCC.shape = -1

            intensity_convolved = numpy.convolve(intensity_interpolated/intensity_interpolated.max(),
                                                 fluxDivZZCC/fluxDivZZCC.max(),
                                                 mode='same')
        else:
            intensity_convolved = intensity_interpolated

        if i == 0:
            print("\n\n============ sizes vs photon energy =======================")
            print("Photon energy/eV  FWHM X'/urad  FWHM Y'/urad  FWHM X/mm  FWHM Z/mm ")

        print("%16.3f  %12.3f  %12.3f  %9.2f  %9.2f" %
              (Ephoton,
              1e6 * get_fwhm(intensity_convolved, divXX)[0],
              1e6 * get_fwhm(fluxDivZZ, divZZ)[0],
              1e3 * get_fwhm(intensity_convolved, divXX)[0] * DISTANCE,
              1e3 * get_fwhm(fluxDivZZ, divZZ)[0] * DISTANCE ))

        if do_plot:
            plot(divX, intensity/intensity.max(),
                 divXX, intensity_interpolated/intensity_interpolated.max(),
                 divXX, intensity_convolved/intensity_convolved.max(),
                 divXX, fluxDivZZCC/fluxDivZZCC.max(),
                 title="min intensity %f, Ephoton=%6.2f" % (intensity.min(), Ephoton), xtitle="divX", ytitle="intensity",
                 legend=["orig","interpolated","convolved","kernel"],show=1)


        # combine H * V
        INTENSITY = numpy.outer(intensity_convolved/intensity_convolved.max(), fluxDivZZ/fluxDivZZ.max())
        p[i,:,:] = INTENSITY

        if do_plot:
            from srxraylib.plot.gol import plot_image, plot_surface, plot_show
            plot_image(INTENSITY, divXX, divZZ, aspect='auto', title="E=%6.2f" % Ephoton, show=1)
            # to create oasys icon...
            # plot_surface(INTENSITY, divXX, divZZ, title="", show=0)
            # import matplotlib.pylab as plt
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
            #                 labeltop='off', labelright='off', labelbottom='off')
            #
            # plot_show()
    #

    h = divXX * DISTANCE * 1e3 # in mm for the h5 file
    v = divZZ * DISTANCE * 1e3 # in mm for the h5 file

    print("\nWindow size: %f mm [H] x %f mm [V]" % (h[-1] - h[0], v[-1] - v[0]))
    print("Window size: %g rad [H] x %g rad [V]" % (divXX[-1] - divXX[0], divZZ[-1] - divZZ[0]))

    # normalization and total flux
    for i in range(e.size):
        INTENSITY = p[i, :, :]
        # norm = INTENSITY.sum() * (h[1] - h[0]) * (v[1] - v[0])
        norm = trapezoidal_rule_2d_1darrays(INTENSITY, h, v)
        p[i, :, :] = INTENSITY / norm * flux[i]


    # fit
    fit_ok = False
    try:
        power = p.sum(axis=0) * (e[1] - e[0]) * codata.e * 1e3
        print("\n\n============= Fitting power density to a 2D Gaussian. ==============\n")
        print("Please use these results with care: check if the original data looks like a Gaussian.")
        fit_parameters = fit_gaussian2d(power,h,v)
        print(info_params(fit_parameters))
        H,V = numpy.meshgrid(h,v)
        data_fitted = twoD_Gaussian( (H,V), * fit_parameters)
        print("  Total power (sum rule) in the fitted data [W]: ",data_fitted.sum()*(h[1]-h[0])*(v[1]-v[0]))
        # plot_image(data_fitted.reshape((h.size,v.size)),h, v,title="FIT")
        print("====================================================\n")
        fit_ok = True
    except:
        pass



    # output file
    if h5_file != "":
        try:
            if h5_initialize:
                h5w = H5SimpleWriter.initialize_file(h5_file,creator="xoppy_wigglers.py")
            else:
                h5w = H5SimpleWriter(h5_file,None)
            h5w.create_entry(h5_entry_name,nx_default=None)
            h5w.add_stack(e,h,v,p,stack_name="Radiation",entry_name=h5_entry_name,
                title_0="Photon energy [eV]",
                title_1="X gap [mm]",
                title_2="Y gap [mm]")
            h5w.create_entry("parameters",root_entry=h5_entry_name,nx_default=None)
            if h5_parameters is not None:
                for key in h5_parameters.keys():
                    h5w.add_key(key,h5_parameters[key], entry_name=h5_entry_name+"/parameters")
            h5w.create_entry("trajectory", root_entry=h5_entry_name, nx_default="transversal trajectory")
            h5w.add_key("traj", traj, entry_name=h5_entry_name + "/trajectory")
            h5w.add_dataset(traj[1,:], traj[0,:], dataset_name="transversal trajectory",entry_name=h5_entry_name + "/trajectory", title_x="s [m]",title_y="X [m]")
            h5w.add_dataset(traj[1,:], traj[3,:], dataset_name="transversal velocity",entry_name=h5_entry_name + "/trajectory", title_x="s [m]",title_y="Vx/c")
            h5w.add_dataset(traj[1, :], traj[7, :], dataset_name="Magnetic field",
                            entry_name=h5_entry_name + "/trajectory", title_x="s [m]", title_y="Bz [T]")
            if fit_ok:
                h5w.add_image(power,h,v,image_name="PowerDensity",entry_name=h5_entry_name,title_x="X [mm]",title_y="Y [mm]")

                h5w.add_image(data_fitted.reshape(h.size,v.size),h,v,image_name="PowerDensityFit",entry_name=h5_entry_name,title_x="X [mm]",title_y="Y [mm]")
                h5w.add_key("fit_info",info_params(fit_parameters), entry_name=h5_entry_name+"/PowerDensityFit")
            print("File written to disk: %s"%h5_file)
        except:
            print("ERROR initializing h5 file")

    return e, h, v, p, traj

#
# auxiliar functions
#

def interpolate_multivalued_function(divX, intensity, divX_i, s):

    divXprime = numpy.gradient(divX, s) # derivative
    knots = crossings_nonzero_all(divXprime)
    knots.insert(0,0)
    knots.append(len(divXprime))

    divX_split = numpy.split(divX, knots)
    intensity_split = numpy.split(intensity, knots)
    s_split = numpy.split(intensity, knots)

    # plot(s, divX/divX.max(),
    #      s,divXprime/divXprime.max(),
    #      s[(knots[0]):(knots[1])], (divX/divX.max())[(knots[0]):(knots[1])],
    #      s[(knots[-2]):(knots[-1])], (divX / divX.max())[(knots[-2]):(knots[-1])],
    #      title='derivative',legend=["divX","divXprime","branch 1","branch N"])

    intensity_interpolated = numpy.zeros_like(divX_i)
    for i in range(len(s_split)):
        if divX_split[i].size > 2:
            fintensity = interp1d(divX_split[i], intensity_split[i], kind='linear', axis=-1, copy=True,
                                  bounds_error=False, fill_value=0.0, assume_sorted=False)
            intensity_interpolated += fintensity(divX_i)
    return intensity_interpolated

def crossings_nonzero_all(data):
    # we suppose the array does not contain 0.0000000000000
    # https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    pos = data > 0
    npos = ~pos
    out =  ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]
    return out.tolist()

def create_magnetic_field_for_bending_magnet(do_plot=False,filename="",B0=-1.0,divergence=1e-3,radius=10.0,npoints=500):

    L = radius * divergence
    Lmax = numpy.abs(L * 1.0)
    y = numpy.linspace(-Lmax / 2, Lmax / 2, npoints)

    B = y * 0.0 + B0

    ybad = numpy.where(numpy.abs(y) > numpy.abs(L / 2) )

    B[ybad] = 0

    if do_plot:
        from srxraylib.plot.gol import plot
        plot(y, B, xtitle="y [m]", ytitle="B [T]",title=filename)

    if filename != "":
        f = open(filename, "w")
        for i in range(y.size):
            f.write("%f  %f\n" % (y[i], B[i]))
        f.close()
        print("File written to disk: %s"%filename)

    return y,B


def trapezoidal_rule_2d_1darrays(data2D,h=None,v=None):
    if h is None:
        h = numpy.arange(data2D.shape[0])
    if v is None:
        v = numpy.arange(data2D.shape[1])
    totPower2 = numpy.trapz(data2D, v, axis=1)
    totPower2 = numpy.trapz(totPower2, h, axis=0)
    return totPower2


if __name__ == "__main__":

    from srxraylib.plot.gol import plot, plot_image, plot_scatter, plot_show, set_qt
    set_qt()


    if True:

        # script to make the calculations (created by XOPPY:wiggler)

        from xoppylib.sources.xoppy_bm_wiggler import xoppy_calc_wigg

        energy, flux, spectral_power, cumulated_power, traj, traj_info = xoppy_calc_wigg(
            FIELD=0,
            NPERIODS=10,
            ULAMBDA=0.15,
            K=22.591,
            ENERGY=6.0,
            PHOT_ENERGY_MIN=100.0,
            PHOT_ENERGY_MAX=100100.0,
            NPOINTS=100,
            NTRAJPOINTS=101,
            CURRENT=200.0,
            FILE="?")

        #
        # script to make the calculations (created by XOPPY:WS)
        #
        import numpy
        from xoppylib.xoppy_run_binaries import xoppy_calc_ws

        out_file = xoppy_calc_ws(
            ENERGY=6.0,
            CUR=200.0,
            PERIOD=15.0,
            N=10.0,
            KX=0.0,
            KY=22.591,
            EMIN=100.0,
            EMAX=100100.0,
            NEE=1000,
            D=30.0,
            XPC=0.0,
            YPC=0.0,
            XPS=115.0,
            YPS=60.0,
            NXP=70,
            NYP=70,
        )

        # data to pass to power
        data = numpy.loadtxt(out_file)
        energy2 = data[:, 0]
        flux2 = data[:, 1]
        # spectral_power = data[:, 2]
        # cumulated_power = data[:, 3]

        #
        # example plot
        #
        from srxraylib.plot.gol import plot

        plot(energy, flux,
             energy2, flux2,
             xtitle="Photon energy [eV]", ytitle="Flux [photons/s/0.1%bw]", title="WS Flux",
             legend=['WIGGLER', 'WS'],
             xlog=0, ylog=0, show=True)


    ##########################################################################
    # wiggler comparison (integrated over a slit)
    #

    #
    # script to make the calculations (created by XOPPY:wiggler)
    #

    if True:

        from xoppylib.sources.xoppy_bm_wiggler import xoppy_calc_wiggler_on_aperture

        energy, flux, spectral_power, cumulated_power, traj, traj_info = xoppy_calc_wiggler_on_aperture(
            FIELD=0,
            NPERIODS=10,
            ULAMBDA=0.15,
            K=22.591,
            ENERGY=6.0,
            PHOT_ENERGY_MIN=100.0,
            PHOT_ENERGY_MAX=100100.0,
            NPOINTS=100,
            NTRAJPOINTS=101,
            CURRENT=200.0,
            FILE="?",
            SLIT_FLAG=1,
            SLIT_D=30.0,
            SLIT_NY=50,
            SLIT_WIDTH_H_MM=115 * 0.4,
            SLIT_CENTER_H_MM=0.0,
            SLIT_HEIGHT_V_MM=60  * 0.3,
            SLIT_CENTER_V_MM=0.0,
        )

        #
        # script to make the calculations (created by XOPPY:WS)
        #
        import numpy
        from xoppylib.xoppy_run_binaries import xoppy_calc_ws

        out_file = xoppy_calc_ws(
            ENERGY=6.0,
            CUR=200.0,
            PERIOD=15.0,
            N=10.0,
            KX=0.0,
            KY=22.591,
            EMIN=100.0,
            EMAX=100100.0,
            NEE=1000,
            D=30.0,
            XPC=0.0,
            YPC=0.0,
            XPS=115.0 * 0.4,
            YPS=60.0  * 0.3,
            NXP=70,
            NYP=70,
        )

        # data to pass to power
        data = numpy.loadtxt(out_file)
        energy2 = data[:, 0]
        flux2 = data[:, 1]
        # spectral_power = data[:, 2]
        # cumulated_power = data[:, 3]

        #
        # example plot
        #
        from srxraylib.plot.gol import plot

        plot(energy, flux,
             energy2, flux2,
             xtitle="Photon energy [eV]", ytitle="Flux [photons/s/0.1%bw]", title="WS Flux",
             legend=['WIGGLER', 'WS'],
             xlog=0, ylog=0, show=True)
