# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


import numpy
import os
from collections import OrderedDict

from xoppylib.sources.srundplug import calc1d_us, calc1d_srw, calc1d_urgent, calc1d_pysru
from xoppylib.sources.srundplug import calc2d_us, calc2d_srw, calc2d_urgent, calc2d_pysru
from xoppylib.sources.srundplug import calc2d_from_harmonics_urgent, calc2d_from_harmonics_urgentpy
from xoppylib.sources.srundplug import calc3d_us, calc3d_srw, calc3d_urgent, calc3d_pysru, calc3d_srw_step_by_step

from xoppylib.fit_gaussian2d import fit_gaussian2d, info_params, twoD_Gaussian

from srxraylib.util.h5_simple_writer import H5SimpleWriter

import scipy.constants as codata
codata_mee = codata.physical_constants["electron mass energy equivalent in MeV"][0]

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

def xoppy_calc_undulator_spectrum(ELECTRONENERGY=6.04,ELECTRONENERGYSPREAD=0.001,ELECTRONCURRENT=0.2,\
                              ELECTRONBEAMSIZEH=0.000395,ELECTRONBEAMSIZEV=9.9e-06,\
                              ELECTRONBEAMDIVERGENCEH=1.05e-05,ELECTRONBEAMDIVERGENCEV=3.9e-06,\
                              PERIODID=0.018,NPERIODS=222,KV=1.68,KH=0.0,KPHASE=0.0,DISTANCE=30.0,
                              GAPH=0.001,GAPV=0.001,GAPH_CENTER=0.0,GAPV_CENTER=0.0,\
                              PHOTONENERGYMIN=3000.0,PHOTONENERGYMAX=55000.0,PHOTONENERGYPOINTS=500,METHOD=2,
                              USEEMITTANCES=1):
    print("Inside xoppy_calc_undulator_spectrum. ")

    bl = OrderedDict()
    bl['ElectronBeamDivergenceH'] = ELECTRONBEAMDIVERGENCEH
    bl['ElectronBeamDivergenceV'] = ELECTRONBEAMDIVERGENCEV
    bl['ElectronBeamSizeH'] = ELECTRONBEAMSIZEH
    bl['ElectronBeamSizeV'] = ELECTRONBEAMSIZEV
    bl['ElectronCurrent'] = ELECTRONCURRENT
    bl['ElectronEnergy'] = ELECTRONENERGY
    bl['ElectronEnergySpread'] = ELECTRONENERGYSPREAD
    bl['Kv'] = KV
    bl['Kh'] = KH
    bl['Kphase'] = KPHASE
    bl['NPeriods'] = NPERIODS
    bl['PeriodID'] = PERIODID
    bl['distance'] = DISTANCE
    bl['gapH'] = GAPH
    bl['gapV'] = GAPV
    bl['gapHcenter'] = GAPH_CENTER
    bl['gapVcenter'] = GAPV_CENTER

    if USEEMITTANCES:
        zero_emittance = False
    else:
        zero_emittance = True

    #TODO remove file and export e,f arrays
    outFile = "undulator_spectrum.spec"

    codata_mee = codata.m_e * codata.c**2 / codata.e # electron mass in eV
    gamma = bl['ElectronEnergy'] * 1e9 / codata_mee

    m2ev = codata.c * codata.h / codata.e      # lambda(m)  = m2eV / energy(eV)
    resonance_wavelength = (1 + (bl['Kv']**2 + bl['Kh']**2) / 2.0) / 2 / gamma**2 * bl["PeriodID"]
    resonance_energy = m2ev / resonance_wavelength
    print ("Gamma: %f \n"%(gamma))
    print ("Resonance wavelength [A]: %g \n"%(1e10*resonance_wavelength))
    print ("Resonance energy [eV]: %g \n"%(resonance_energy))


    ptot = (NPERIODS/6) * codata.value('characteristic impedance of vacuum') * \
           ELECTRONCURRENT * codata.e * 2 * numpy.pi * codata.c * gamma**2 * (KV**2+KH**2) / PERIODID
    print ("\nTotal power radiated by the undulator with fully opened slits [W]: %g \n"%(ptot))


    if METHOD == 0:
        print("Undulator flux calculation using US. Please wait...")
        e, f = calc1d_us(bl,photonEnergyMin=PHOTONENERGYMIN,photonEnergyMax=PHOTONENERGYMAX,
              photonEnergyPoints=PHOTONENERGYPOINTS,fileName=outFile,fileAppend=False,zero_emittance=zero_emittance)
        print("Done")
        print("\nCheck calculation output at: %s"%(os.path.join(os.getcwd(),"us.out")))
    if METHOD == 1:
        print("Undulator flux calculation using URGENT. Please wait...")
        e, f = calc1d_urgent(bl,photonEnergyMin=PHOTONENERGYMIN,photonEnergyMax=PHOTONENERGYMAX,
              photonEnergyPoints=PHOTONENERGYPOINTS,fileName=outFile,fileAppend=False,zero_emittance=zero_emittance)
        print("Done")
        print("\nCheck calculation output at: %s"%(os.path.join(os.getcwd(),"urgent.out")))
    if METHOD == 2:
        # get the maximum harmonic number
        h_max = int(2.5*PHOTONENERGYMAX/resonance_energy)

        print ("Number of harmonics considered: %d \n"%(h_max))
        print("Undulator flux calculation using SRW. Please wait...")
        e, f = calc1d_srw(bl,photonEnergyMin=PHOTONENERGYMIN,photonEnergyMax=PHOTONENERGYMAX,
              photonEnergyPoints=PHOTONENERGYPOINTS,fileName=outFile,fileAppend=False,zero_emittance=zero_emittance,
              srw_max_harmonic_number=h_max)
        print("Done")

    if zero_emittance:
        print("\nNo emittance calculation")

    if METHOD == 1 and len(e) == 0: raise Exception("Invalid Input Parameters")

    power_in_spectrum = f.sum()*1e3*codata.e*(e[1]-e[0])
    print("\nPower from integral of spectrum: %8.3f W"%(power_in_spectrum))
    print("\nRatio Power from integral of spectrum over Total emitted power: %5.4f"%(power_in_spectrum / ptot))

    spectral_power = f * codata.e * 1e3
    try:
        cumulated_power = spectral_power.cumsum() * numpy.abs(e[0] - e[1])
    except:
        cumulated_power = 0.0

    return e, f, spectral_power, cumulated_power


def xoppy_calc_undulator_power_density(ELECTRONENERGY=6.04,
                                       ELECTRONENERGYSPREAD=0.001,
                                       ELECTRONCURRENT=0.2,
                                       ELECTRONBEAMSIZEH=0.000395,
                                       ELECTRONBEAMSIZEV=9.9e-06,
                                       ELECTRONBEAMDIVERGENCEH=1.05e-05,
                                       ELECTRONBEAMDIVERGENCEV=3.9e-06,
                                       PERIODID=0.018,
                                       NPERIODS=222,
                                       KV=1.68,
                                       KH=0.0,
                                       KPHASE=0.0,
                                       DISTANCE=30.0,
                                       GAPH=0.001,
                                       GAPV=0.001,
                                       HSLITPOINTS=101,
                                       VSLITPOINTS=51,
                                       METHOD=2,
                                       USEEMITTANCES=1,
                                       MASK_FLAG=0,
                                       MASK_ROT_H_DEG=0.0,
                                       MASK_ROT_V_DEG=0.0,
                                       MASK_H_MIN=None,
                                       MASK_H_MAX=None,
                                       MASK_V_MIN=None,
                                       MASK_V_MAX=None,
                                       h5_file="",
                                       h5_entry_name="XOPPY_POWERDENSITY",
                                       h5_initialize=True,
                                       h5_parameters={},
                                       ):
    print("Inside xoppy_calc_undulator_power_density. ")

    bl = OrderedDict()
    bl['ElectronBeamDivergenceH'] = ELECTRONBEAMDIVERGENCEH
    bl['ElectronBeamDivergenceV'] = ELECTRONBEAMDIVERGENCEV
    bl['ElectronBeamSizeH'] = ELECTRONBEAMSIZEH
    bl['ElectronBeamSizeV'] = ELECTRONBEAMSIZEV
    bl['ElectronCurrent'] = ELECTRONCURRENT
    bl['ElectronEnergy'] = ELECTRONENERGY
    bl['ElectronEnergySpread'] = ELECTRONENERGYSPREAD
    bl['Kv'] = KV
    bl['Kh'] = KH
    bl['Kphase'] = KPHASE
    bl['NPeriods'] = NPERIODS
    bl['PeriodID'] = PERIODID
    bl['distance'] = DISTANCE
    bl['gapH'] = GAPH
    bl['gapV'] = GAPV

    if USEEMITTANCES:
        zero_emittance = False
    else:
        zero_emittance = True

    #TODO remove SPEC file
    outFile = "undulator_power_density.spec"

    if METHOD == 0:
        code = "US"
        print("Undulator power_density calculation using US. Please wait...")
        h,v,p = calc2d_us(bl,fileName=outFile,fileAppend=False,hSlitPoints=HSLITPOINTS,vSlitPoints=VSLITPOINTS,
                                    zero_emittance=zero_emittance)
        print("Done")
    if METHOD == 1:
        code = "URGENT"
        print("Undulator power_density calculation using URGENT. Please wait...")
        h,v,p = calc2d_urgent(bl,fileName=outFile,fileAppend=False,hSlitPoints=HSLITPOINTS,vSlitPoints=VSLITPOINTS,
                                        zero_emittance=zero_emittance)
        print("Done")
    if METHOD == 2:
        code = "SRW"
        print("Undulator power_density calculation using SRW. Please wait...")
        h,v,p = calc2d_srw(bl,fileName=outFile,fileAppend=False,hSlitPoints=HSLITPOINTS,vSlitPoints=VSLITPOINTS,
                                     zero_emittance=zero_emittance)
        print("Done")


    if zero_emittance:
        print("No emittance calculation")
    codata_mee = codata.m_e * codata.c**2 / codata.e # electron mass in eV
    gamma = ELECTRONENERGY * 1e9 / codata_mee
    ptot = (NPERIODS/6) * codata.value('characteristic impedance of vacuum') * \
           ELECTRONCURRENT * codata.e * 2 * numpy.pi * codata.c * gamma**2 * (KV**2 + KH**2)/ PERIODID
    print ("\nTotal power radiated by the undulator with fully opened slits [W]: %g \n"%(ptot))


    if MASK_FLAG:
        #
        # rotation
        #
        v /= numpy.cos(MASK_ROT_H_DEG * numpy.pi / 180)
        h /= numpy.cos(MASK_ROT_V_DEG * numpy.pi / 180)
        # also reduce the power density!!
        p *= numpy.cos(MASK_ROT_H_DEG * numpy.pi / 180)
        p *= numpy.cos(MASK_ROT_V_DEG * numpy.pi / 180)

        #
        # mask
        #
        if MASK_H_MIN is not None:
            lower_window_h = numpy.where(h < MASK_H_MIN)
            if len(lower_window_h) > 0: p[lower_window_h,:] = 0

        if MASK_H_MAX is not None:
            upper_window_h = numpy.where(h > MASK_H_MAX)
            if len(upper_window_h) > 0: p[upper_window_h,:] = 0

        if MASK_V_MIN is not None:
            lower_window_v = numpy.where(v < MASK_V_MIN)
            if len(lower_window_v) > 0: p[:,lower_window_v] = 0

        if MASK_V_MIN is not None:
            upper_window_v = numpy.where(v > MASK_V_MAX)
            if len(upper_window_v) > 0: p[:,upper_window_v] = 0

        txt0 = "============= power density in the modified (masked) screen ==========\n"
    else:
        txt0 = "=================== power density  ======================\n"

    text_info = txt0
    text_info += "  Power density peak: %f W/mm2\n"%p.max()
    text_info += "  Total power: %f W\n"%(p.sum()*(h[1]-h[0])*(v[1]-v[0]))
    text_info += "====================================================\n"
    print(text_info)

    # fit
    fit_ok = False
    try:
        print("============= Fitting power density to a 2D Gaussian. ==============\n")
        print("Please use these results with care: check if the original data looks like a Gaussian.")
        fit_parameters = fit_gaussian2d(p,h,v)
        print(info_params(fit_parameters))
        H,V = numpy.meshgrid(h,v)
        data_fitted = twoD_Gaussian( (H,V), *fit_parameters)
        print("  Total power in the fitted data [W]: ",data_fitted.sum()*(h[1]-h[0])*(v[1]-v[0]))
        # plot_image(data_fitted.reshape((h.size,v.size)),h, v,title="FIT")
        print("====================================================\n")
        fit_ok = True
    except:
        pass

    if h5_file != "":
        try:
            if h5_initialize:
                h5w = H5SimpleWriter.initialize_file(h5_file,creator="xoppy_undulators.py")
            else:
                h5w = H5SimpleWriter(h5_file,None)
            h5w.create_entry(h5_entry_name,nx_default="PowerDensity")
            h5w.add_image(p,h,v,image_name="PowerDensity",entry_name=h5_entry_name,title_x="X [mm]",title_y="Y [mm]")
            h5w.add_key("info",text_info, entry_name=h5_entry_name)
            h5w.create_entry("parameters",root_entry=h5_entry_name,nx_default=None)
            for key in h5_parameters.keys():
                h5w.add_key(key,h5_parameters[key], entry_name=h5_entry_name+"/parameters")
            if fit_ok:
                h5w.add_image(data_fitted.reshape(h.size,v.size),h,v,image_name="PowerDensityFit",entry_name=h5_entry_name,title_x="X [mm]",title_y="Y [mm]")
                h5w.add_key("fit_info",info_params(fit_parameters), entry_name=h5_entry_name+"/PowerDensityFit")

            print("File written to disk: %s"%h5_file)
        except:
            print("ERROR initializing h5 file")

    return h, v, p, code

def xoppy_calc_undulator_power_density_from_harmonics(
                                       ELECTRONENERGY=6.04,
                                       ELECTRONENERGYSPREAD=0.001,
                                       ELECTRONCURRENT=0.2,
                                       ELECTRONBEAMSIZEH=0.000395,
                                       ELECTRONBEAMSIZEV=9.9e-06,
                                       ELECTRONBEAMDIVERGENCEH=1.05e-05,
                                       ELECTRONBEAMDIVERGENCEV=3.9e-06,
                                       PERIODID=0.018,
                                       NPERIODS=222,
                                       KV=1.68,
                                       KH=0.0,
                                       KPHASE=0.0,
                                       DISTANCE=30.0,
                                       GAPH=0.001,
                                       GAPV=0.001,
                                       HSLITPOINTS=101,
                                       VSLITPOINTS=51,
                                       METHOD=0,
                                       USEEMITTANCES=1,
                                       MASK_FLAG=0,
                                       MASK_ROT_H_DEG=0.0,
                                       MASK_ROT_V_DEG=0.0,
                                       MASK_H_MIN=None,
                                       MASK_H_MAX=None,
                                       MASK_V_MIN=None,
                                       MASK_V_MAX=None,
                                       h5_file="",
                                       h5_entry_name="XOPPY_POWERDENSITY",
                                       h5_initialize=True,
                                       h5_parameters={},
                                       harmonic_max=20,
                                       photon_energy_bin=100.0, # in eV, the bin for the Spectral Power calculation
                                       ):
    print("Inside xoppy_calc_undulator_power_density_from_harmonics. ")

    bl = OrderedDict()
    bl['ElectronBeamDivergenceH'] = ELECTRONBEAMDIVERGENCEH
    bl['ElectronBeamDivergenceV'] = ELECTRONBEAMDIVERGENCEV
    bl['ElectronBeamSizeH'] = ELECTRONBEAMSIZEH
    bl['ElectronBeamSizeV'] = ELECTRONBEAMSIZEV
    bl['ElectronCurrent'] = ELECTRONCURRENT
    bl['ElectronEnergy'] = ELECTRONENERGY
    bl['ElectronEnergySpread'] = ELECTRONENERGYSPREAD
    bl['Kv'] = KV
    bl['Kh'] = KH
    bl['Kphase'] = KPHASE
    bl['NPeriods'] = NPERIODS
    bl['PeriodID'] = PERIODID
    bl['distance'] = DISTANCE
    bl['gapH'] = GAPH
    bl['gapV'] = GAPV

    if USEEMITTANCES:
        zero_emittance = False
    else:
        zero_emittance = True

    #TODO remove SPEC file
    outFile = "undulator_power_density.spec"

    if METHOD == 0:
        code = "URGENT"
        print("Undulator power_density_from_harmonics calculation using URGENT. Please wait...")
        h, v, p, power_density_harmonics, energy_harmonics = calc2d_from_harmonics_urgent(bl,
                                             fileName=outFile,
                                             fileAppend=False,
                                             hSlitPoints=HSLITPOINTS,
                                             vSlitPoints=VSLITPOINTS,
                                             zero_emittance=zero_emittance,
                                             harmonic_max=harmonic_max,
                                             )
        print("Done")
    if METHOD == 1:
        code = "URGENTPY"
        print("Undulator power_density_from_harmonics calculation using URGENTPY. Please wait...")
        h, v, p, power_density_harmonics, energy_harmonics = calc2d_from_harmonics_urgentpy(
                                            bl,
                                            fileName=outFile,
                                            fileAppend=False,
                                            hSlitPoints=HSLITPOINTS,
                                            vSlitPoints=VSLITPOINTS,
                                            zero_emittance=zero_emittance,
                                            harmonic_max=harmonic_max,
                                            )
        print("Done")


    if zero_emittance:
        print("No emittance calculation")
    codata_mee = codata.m_e * codata.c**2 / codata.e # electron mass in eV
    gamma = ELECTRONENERGY * 1e9 / codata_mee
    ptot = (NPERIODS/6) * codata.value('characteristic impedance of vacuum') * \
           ELECTRONCURRENT * codata.e * 2 * numpy.pi * codata.c * gamma**2 * (KV**2 + KH**2)/ PERIODID
    print ("\nTotal power radiated by the undulator with fully opened slits [W]: %g \n"%(ptot))


    if MASK_FLAG:
        #
        # rotation
        #
        v /= numpy.cos(MASK_ROT_H_DEG * numpy.pi / 180)
        h /= numpy.cos(MASK_ROT_V_DEG * numpy.pi / 180)
        # also reduce the power density!!
        p *= numpy.cos(MASK_ROT_H_DEG * numpy.pi / 180)
        p *= numpy.cos(MASK_ROT_V_DEG * numpy.pi / 180)

        #
        # mask
        #
        if MASK_H_MIN is not None:
            lower_window_h = numpy.where(h < MASK_H_MIN)
            if len(lower_window_h) > 0:
                p[lower_window_h,:] = 0
                power_density_harmonics[:, lower_window_h, :] = 0

        if MASK_H_MAX is not None:
            upper_window_h = numpy.where(h > MASK_H_MAX)
            if len(upper_window_h) > 0:
                p[upper_window_h, :] = 0
                power_density_harmonics[:, upper_window_h, :] = 0

        if MASK_V_MIN is not None:
            lower_window_v = numpy.where(v < MASK_V_MIN)
            if len(lower_window_v) > 0:
                p[:, lower_window_v] = 0
                power_density_harmonics[:, :, lower_window_v] = 0

        if MASK_V_MIN is not None:
            upper_window_v = numpy.where(v > MASK_V_MAX)
            if len(upper_window_v) > 0:
                p[:, upper_window_v] = 0
                power_density_harmonics[:, :, upper_window_v] = 0

        txt0 = "============= power density in the modified (masked) screen ==========\n"
    else:
        txt0 = "=================== power density  ======================\n"

    text_info = txt0
    text_info += "  Power density peak: %f W/mm2\n"%p.max()
    text_info += "  Total power: %f W\n"%(p.sum()*(h[1]-h[0])*(v[1]-v[0]))
    text_info += "====================================================\n"
    print(text_info)

    # fit
    fit_ok = False
    try:
        print("============= Fitting power density to a 2D Gaussian. ==============\n")
        print("Please use these results with care: check if the original data looks like a Gaussian.")
        fit_parameters = fit_gaussian2d(p,h,v)
        print(info_params(fit_parameters))
        H,V = numpy.meshgrid(h,v)
        data_fitted = twoD_Gaussian( (H,V), *fit_parameters)
        print("  Total power in the fitted data [W]: ",data_fitted.sum()*(h[1]-h[0])*(v[1]-v[0]))
        # plot_image(data_fitted.reshape((h.size,v.size)),h, v,title="FIT")
        print("====================================================\n")
        fit_ok = True
    except:
        pass

    if h5_file != "":
        try:
            if h5_initialize:
                h5w = H5SimpleWriter.initialize_file(h5_file,creator="xoppy_undulators.py")
            else:
                h5w = H5SimpleWriter(h5_file,None)
            h5w.create_entry(h5_entry_name,nx_default="PowerDensity")
            h5w.add_image(p,h,v,image_name="PowerDensity",entry_name=h5_entry_name,title_x="X [mm]",title_y="Y [mm]")
            h5w.add_key("info",text_info, entry_name=h5_entry_name)
            h5w.create_entry("parameters",root_entry=h5_entry_name,nx_default=None)
            for key in h5_parameters.keys():
                h5w.add_key(key,h5_parameters[key], entry_name=h5_entry_name+"/parameters")
            if fit_ok:
                h5w.add_image(data_fitted.reshape(h.size,v.size),h,v,image_name="PowerDensityFit",entry_name=h5_entry_name,title_x="X [mm]",title_y="Y [mm]")
                h5w.add_key("fit_info",info_params(fit_parameters), entry_name=h5_entry_name+"/PowerDensityFit")

            n = numpy.arange(power_density_harmonics.shape[0])
            h5w.add_stack(n, h, v, power_density_harmonics,
                          stack_name="HarmonicsPowerDensity", entry_name=h5_entry_name,
                          title_0="Harmonic index", title_1="X gap [mm]", title_2="Y gap [mm]")
            #
            h5w.add_stack(n, h, v, energy_harmonics,
                          stack_name="HarmonicsPhotonEnergy", entry_name=h5_entry_name,
                          title_0="Harmonic index", title_1="X gap [mm]", title_2="Y gap [mm]")


            print("File written to disk: %s"%h5_file)
        except:
            print("ERROR initializing h5 file")

    # Spectral Power
    energies = energy_harmonics.flatten()  # Flatten all energy values
    powers = power_density_harmonics.flatten() * (h[1] - h[0]) * (v[1] - v[0])  # Flatten corresponding power contributions
    e_bins = numpy.arange(energy_harmonics.min(), energy_harmonics.max() + photon_energy_bin, photon_energy_bin)
    # Bin the powers into energy bins
    power_vs_energy, bin_edges = numpy.histogram(energies, bins=e_bins, weights=powers)
    spectral_power_energy = 0.5 * (bin_edges[:-1] + bin_edges[1:]) # bin centers
    spectral_power = power_vs_energy / photon_energy_bin

    return h, v, p, code, power_density_harmonics, energy_harmonics, spectral_power, spectral_power_energy

def xoppy_calc_undulator_radiation(ELECTRONENERGY=6.04,ELECTRONENERGYSPREAD=0.001,ELECTRONCURRENT=0.2,\
                                       ELECTRONBEAMSIZEH=0.000395,ELECTRONBEAMSIZEV=9.9e-06,\
                                       ELECTRONBEAMDIVERGENCEH=1.05e-05,ELECTRONBEAMDIVERGENCEV=3.9e-06,\
                                       PERIODID=0.018,NPERIODS=222,KV=1.68,KH=0.0,KPHASE=0.0,DISTANCE=30.0,
                                       SETRESONANCE=0,HARMONICNUMBER=1,
                                       GAPH=0.003,GAPV=0.003,GAPH_CENTER=0.0,GAPV_CENTER=0.0,
                                       HSLITPOINTS=41,VSLITPOINTS=41,METHOD=2,
                                       PHOTONENERGYMIN=7982.2,PHOTONENERGYMAX=7983.2,PHOTONENERGYPOINTS=2,
                                       USEEMITTANCES=1,
                                       h5_file="",h5_entry_name="XOPPY_RADIATION",h5_initialize=True,h5_parameters={}):
    print("Inside xoppy_calc_undulator_radiation. ")

    bl = OrderedDict()
    bl['ElectronBeamDivergenceH'] = ELECTRONBEAMDIVERGENCEH
    bl['ElectronBeamDivergenceV'] = ELECTRONBEAMDIVERGENCEV
    bl['ElectronBeamSizeH'] = ELECTRONBEAMSIZEH
    bl['ElectronBeamSizeV'] = ELECTRONBEAMSIZEV
    bl['ElectronCurrent'] = ELECTRONCURRENT
    bl['ElectronEnergy'] = ELECTRONENERGY
    bl['ElectronEnergySpread'] = ELECTRONENERGYSPREAD
    bl['Kv'] = KV
    bl['Kh'] = KH
    bl['Kphase'] = KPHASE
    bl['NPeriods'] = NPERIODS
    bl['PeriodID'] = PERIODID
    bl['distance'] = DISTANCE
    bl['gapH'] = GAPH
    bl['gapV'] = GAPV
    bl['gapHcenter'] = GAPH_CENTER
    bl['gapVcenter'] = GAPV_CENTER

    if USEEMITTANCES:
        zero_emittance = False
    else:
        zero_emittance = True

    gamma = ELECTRONENERGY / (codata_mee * 1e-3)


    resonance_wavelength = (1 + (bl['Kv']**2 + bl['Kh']**2)/ 2.0) / 2 / gamma**2 * bl["PeriodID"]
    m2ev = codata.c * codata.h / codata.e      # lambda(m)  = m2eV / energy(eV)
    resonance_energy = m2ev / resonance_wavelength

    resonance_central_cone = 1.0/gamma*numpy.sqrt( (1+0.5*(KV**2+KH**2))/(2*NPERIODS*HARMONICNUMBER) )

    ring_order = 1

    resonance_ring = 1.0/gamma*numpy.sqrt( ring_order / HARMONICNUMBER * (1+0.5*(KV**2+KH**2)) )

    # autoset energy
    if SETRESONANCE == 0:
        photonEnergyMin = PHOTONENERGYMIN
        photonEnergyMax = PHOTONENERGYMAX
        photonEnergyPoints = PHOTONENERGYPOINTS
    else:
        # referred to resonance
        photonEnergyMin = resonance_energy
        photonEnergyMax = resonance_energy + 1
        photonEnergyPoints = 2

    # autoset slit

    if SETRESONANCE == 0:
        pass
    elif SETRESONANCE == 1:
        MAXANGLE = 3 * 0.69 * resonance_central_cone
        bl['gapH'] = 2 * MAXANGLE * DISTANCE
        bl['gapV'] = 2 * MAXANGLE * DISTANCE
    elif SETRESONANCE == 2:
        MAXANGLE = 2.1 * resonance_ring
        bl['gapH'] = 2 * MAXANGLE * DISTANCE
        bl['gapV'] = 2 * MAXANGLE * DISTANCE


    #TODO SPEC file can be removed
    outFile = "undulator_radiation.spec"

    # Memorandum:
    # e = array with energy in eV
    # h = array with horizontal positions in mm
    # v = array with vertical positions in mm
    # p = array with photon flux in photons/s/0.1%bw/mm^2 with shape (Ne,Nh.Nv)
    if METHOD == 0:
        code = "US"
        print("Undulator radiation calculation using US. Please wait...")
        e,h,v,p = calc3d_us(bl,fileName=outFile,fileAppend=False,hSlitPoints=HSLITPOINTS,vSlitPoints=VSLITPOINTS,
                                    photonEnergyMin=photonEnergyMin,photonEnergyMax=photonEnergyMax,
                                    photonEnergyPoints=photonEnergyPoints,zero_emittance=zero_emittance)
    if METHOD == 1:
        code = "URGENT"
        print("Undulator radiation calculation using URGENT. Please wait...")
        e,h,v,p = calc3d_urgent(bl,fileName=outFile,fileAppend=False,hSlitPoints=HSLITPOINTS,vSlitPoints=VSLITPOINTS,
                                    photonEnergyMin=photonEnergyMin,photonEnergyMax=photonEnergyMax,
                                    photonEnergyPoints=photonEnergyPoints,zero_emittance=zero_emittance)
    if METHOD == 2:
        code = "SRW"
        print("Undulator radiation calculation using SRW. Please wait...")
        e,h,v,p = calc3d_srw(bl,fileName=outFile,fileAppend=False,hSlitPoints=HSLITPOINTS,vSlitPoints=VSLITPOINTS,
                                    photonEnergyMin=photonEnergyMin,photonEnergyMax=photonEnergyMax,
                                    photonEnergyPoints=photonEnergyPoints,zero_emittance=zero_emittance)
    if METHOD == 22:
        code = "SRW"
        print("Undulator radiation calculation using SRW. Please wait...")
        e, h, v, p = calc3d_srw_step_by_step(bl, fileName=outFile, fileAppend=False, hSlitPoints=HSLITPOINTS,
                                          vSlitPoints=VSLITPOINTS,
                                          photonEnergyMin=photonEnergyMin, photonEnergyMax=photonEnergyMax,
                                          photonEnergyPoints=photonEnergyPoints, zero_emittance=zero_emittance)
    if METHOD == 3:
        # todo too slow
        code = "pySRU"
        print("Undulator radiation calculation using SRW. Please wait...")
        e,h,v,p = calc3d_pysru(bl,fileName=outFile,fileAppend=False,hSlitPoints=HSLITPOINTS,vSlitPoints=VSLITPOINTS,
                                    photonEnergyMin=photonEnergyMin,photonEnergyMax=photonEnergyMax,
                                    photonEnergyPoints=photonEnergyPoints,zero_emittance=zero_emittance)


    print ("Gamma: %f \n"%(gamma))
    print ("Resonance wavelength (1st harmonic): %g A\n"%(1e10*resonance_wavelength))
    print ("Resonance energy (1st harmonic): %g eV\n"%(resonance_energy))
    if HARMONICNUMBER != 1:
        print ("Resonance wavelength (%d harmonic): %g A\n"%(HARMONICNUMBER,1e10*resonance_wavelength/HARMONICNUMBER))
        print ("Resonance energy (%d harmonic): %g eV\n"%(HARMONICNUMBER,HARMONICNUMBER*resonance_energy))
    print ("Resonance central cone (%d harmonic): %g urad\n"%(HARMONICNUMBER,1e6*resonance_central_cone))


    print ("Resonance first ring (%d harmonic): %g urad\n"%(HARMONICNUMBER,1e6*resonance_ring))

    print("Calculated %d photon energy points from %f to %f."%(photonEnergyPoints,photonEnergyMin,photonEnergyMax))

    if zero_emittance:
        print("No emittance.")

    print("Done")

    ptot = (NPERIODS/6) * codata.value('characteristic impedance of vacuum') * \
           ELECTRONCURRENT * codata.e * 2 * numpy.pi * codata.c * gamma**2 * (KV**2 + KH**2)/ PERIODID
    print ("\nTotal power radiated by the undulator with fully opened slits [W]: %f \n"%(ptot))


    if SETRESONANCE == 0:
        pcalc =  p.sum() * codata.e * 1e3 * (h[1]-h[0]) * (v[1]-v[0]) * (e[1]-e[0])
        print ("\nTotal power from calculated spectrum (h,v,energy) grid [W]: %f \n"%pcalc)


    # fit
    try:
        print("============= Fitting power density to a 2D Gaussian. ==============\n")
        print("Please use these results with care: check if the original data looks like a Gaussian.\n")
        print("Length units are mm")
        data_to_fit = p.sum(axis=0)*(e[1]-e[0])*codata.e*1e3
        fit_parameters = fit_gaussian2d(data_to_fit,h,v)
        print(info_params(fit_parameters))
        H,V = numpy.meshgrid(h,v)
        data_fitted = twoD_Gaussian( (H,V), *fit_parameters)
        print("  Total power in the fitted data [W]: ",data_fitted.sum()*(h[1]-h[0])*(v[1]-v[0]))
        # plot_image(data_fitted.reshape((h.size,v.size)),h, v,title="FIT")
        print("====================================================\n")

    except:
        pass

    if h5_file != "":
        try:
            if h5_initialize:
                h5w = H5SimpleWriter.initialize_file(h5_file,creator="xoppy_undulators.py")
            else:
                h5w = H5SimpleWriter(h5_file,None)
            h5w.create_entry(h5_entry_name,nx_default=None)
            h5w.add_stack(e,h,v,p,stack_name="Radiation",entry_name=h5_entry_name,
                title_0="Photon energy [eV]",
                title_1="X gap [mm]",
                title_2="Y gap [mm]")
            h5w.create_entry("parameters",root_entry=h5_entry_name,nx_default=None)
            for key in h5_parameters.keys():
                h5w.add_key(key,h5_parameters[key], entry_name=h5_entry_name+"/parameters")
            print("File written to disk: %s"%h5_file)
        except:
            print("ERROR initializing h5 file")

    return e, h, v, p, code


if __name__ == "__main__":

    from srxraylib.plot.gol import plot, plot_image

    if False:
        e, f, spectral_power, cumulated_power = xoppy_calc_undulator_spectrum()
        plot(e,f)

        h, v, p, code = xoppy_calc_undulator_power_density(h5_file="test.h5",h5_initialize=True)
        plot_image(p,h,v)

        e, h, v, p, code = xoppy_calc_undulator_radiation(ELECTRONENERGY=6.0, h5_file="test.h5",h5_entry_name="first_entry",h5_initialize=True)
        e, h, v, p, code = xoppy_calc_undulator_radiation(ELECTRONENERGY=7.0, h5_file="test.h5",h5_entry_name="second_entry",h5_initialize=False)

    if False:

        h5_parameters = dict()
        h5_parameters["ELECTRONENERGY"] = 6.0
        h5_parameters["ELECTRONENERGYSPREAD"] = 0.001
        h5_parameters["ELECTRONCURRENT"] = 0.2
        h5_parameters["ELECTRONBEAMSIZEH"] = 3.01836e-05
        h5_parameters["ELECTRONBEAMSIZEV"] = 3.63641e-06
        h5_parameters["ELECTRONBEAMDIVERGENCEH"] = 4.36821e-06
        h5_parameters["ELECTRONBEAMDIVERGENCEV"] = 1.37498e-06
        h5_parameters["PERIODID"] = 0.018
        h5_parameters["NPERIODS"] = 111
        h5_parameters["KV"] = 1.6563
        h5_parameters["KH"] = 0.0
        h5_parameters["KPHASE"] = 0.0
        h5_parameters["DISTANCE"] = 30.0
        h5_parameters["GAPH"] = 0.01
        h5_parameters["GAPV"] = 0.01
        h5_parameters["HSLITPOINTS"] = 41
        h5_parameters["VSLITPOINTS"] = 41
        h5_parameters["METHOD"] = 1  # 0=urgent (fortran), 1=urgentpy (python)
        h5_parameters["USEEMITTANCES"] = 1
        h5_parameters["MASK_FLAG"] = 0
        h5_parameters["MASK_ROT_H_DEG"] = 0.0
        h5_parameters["MASK_ROT_V_DEG"] = 0.0
        h5_parameters["MASK_H_MIN"] = -1000.0
        h5_parameters["MASK_H_MAX"] = 1000.0
        h5_parameters["MASK_V_MIN"] = -1000.0
        h5_parameters["MASK_V_MAX"] = 1000.0
        h5_parameters["harmonic_max"] = 20  # maximum harmonic to calculate


        h, v, p, code, p_harmonics, e_harmonics, spectral_power, spectral_power_energy = xoppy_calc_undulator_power_density_from_harmonics(
            ELECTRONENERGY=h5_parameters["ELECTRONENERGY"],
            ELECTRONENERGYSPREAD=h5_parameters["ELECTRONENERGYSPREAD"],
            ELECTRONCURRENT=h5_parameters["ELECTRONCURRENT"],
            ELECTRONBEAMSIZEH=h5_parameters["ELECTRONBEAMSIZEH"],
            ELECTRONBEAMSIZEV=h5_parameters["ELECTRONBEAMSIZEV"],
            ELECTRONBEAMDIVERGENCEH=h5_parameters["ELECTRONBEAMDIVERGENCEH"],
            ELECTRONBEAMDIVERGENCEV=h5_parameters["ELECTRONBEAMDIVERGENCEV"],
            PERIODID=h5_parameters["PERIODID"],
            NPERIODS=h5_parameters["NPERIODS"],
            KV=h5_parameters["KV"],
            KH=h5_parameters["KH"],
            KPHASE=h5_parameters["KPHASE"],
            DISTANCE=h5_parameters["DISTANCE"],
            GAPH=h5_parameters["GAPH"],
            GAPV=h5_parameters["GAPV"],
            HSLITPOINTS=h5_parameters["HSLITPOINTS"],
            VSLITPOINTS=h5_parameters["VSLITPOINTS"],
            METHOD=h5_parameters["METHOD"],
            harmonic_max=h5_parameters["harmonic_max"],
            USEEMITTANCES=h5_parameters["USEEMITTANCES"],
            MASK_FLAG=h5_parameters["MASK_FLAG"],
            MASK_ROT_H_DEG=h5_parameters["MASK_ROT_H_DEG"],
            MASK_ROT_V_DEG=h5_parameters["MASK_ROT_V_DEG"],
            MASK_H_MIN=h5_parameters["MASK_H_MIN"],
            MASK_H_MAX=h5_parameters["MASK_H_MAX"],
            MASK_V_MIN=h5_parameters["MASK_V_MIN"],
            MASK_V_MAX=h5_parameters["MASK_V_MAX"],
            h5_file="undulator_power_density_from_harmonics.h5",
            h5_entry_name="XOPPY_POWERDENSITY",
            h5_initialize=True,
            h5_parameters=h5_parameters,
            photon_energy_bin=400.0,
        )

        #
        # plot power density vs h,v
        #
        plot_image(p,h,v)
        plot(spectral_power_energy, spectral_power)