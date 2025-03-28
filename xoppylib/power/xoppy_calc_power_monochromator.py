import numpy
import scipy.constants as codata
from xoppylib.power.power1d_calc_monochromators import \
    power1d_calc_bragg_monochromator, power1d_calc_laue_monochromator, power1d_calc_multilayer_monochromator

def List_Product(list):
    L = []
    l = 1
    for k in range(len(list[0])):
        for i in range(len(list)):
            l = l * list[i][k]
        L.append(l)
        l = 1
    return (L)

def xoppy_calc_power_monochromator(energies=None,                    # array with energies in eV
                                   source=None,                      # array with source spectral density
                                   TYPE=0,                           # 0=None, 1=Crystal Bragg, 2=Crystal Laue, 3=Multilayer
                                   ENER_SELECTED=8000.0,             # Energy to set crystal monochromator
                                   METHOD=0,                         # For crystals, in crystalpy, 0=Zachariasem, 1=Guigay
                                   THICK=1.0,                        # crystal thicknes for Laue crystals in um
                                   ML_H5_FILE="",                    # File with inputs from multilaters (from xoppy/Multilayer)
                                   ML_GRAZING_ANGLE_DEG=0.4,         # For multilayer, the grazing angle in degrees
                                   N_REFLECTIONS=1,                  # number of reflections (crystals or multilayers)
                                   FILE_DUMP=0,                      # 0=No, 1=yes
                                   polarization=0,                   # 0=sigma, 1=pi, 2=unpolarized
                                   output_file="monochromator.spec", # filename if FILE_DUMP=1
                                   ):

    if energies is None:
        energies = numpy.linspace(1000,10000,100)
        source = numpy.ones_like(energies)
    if source is None: source = numpy.ones_like(energies)

    if TYPE == 0:
        Mono_Effect = numpy.array([1] * len(energies))
    elif TYPE == 1:
        Mono_Effect, Mono_Effect_p = power1d_calc_bragg_monochromator(energies=energies,
                                                                      energy_setup=ENER_SELECTED,
                                                                      calculation_method=METHOD)
    elif TYPE == 2:
        Mono_Effect, Mono_Effect_p = power1d_calc_laue_monochromator(energies=energies,
                                                                     energy_setup=ENER_SELECTED,
                                                                     calculation_method=METHOD,
                                                                     thickness=THICK * 1e-6)
    elif TYPE == 3:
        Mono_Effect, Mono_Effect_p = power1d_calc_multilayer_monochromator(ML_H5_FILE,
                                                                           energies=energies,
                                                                           grazing_angle_deg=ML_GRAZING_ANGLE_DEG)


    if polarization == 0:
        Mono_Effect = Mono_Effect ** N_REFLECTIONS
    elif polarization == 1:
        Mono_Effect = Mono_Effect_p ** N_REFLECTIONS
    elif polarization == 2:
        Mono_Effect = ((Mono_Effect + Mono_Effect_p) / 2) ** N_REFLECTIONS


    Final_Spectrum = List_Product([source, Mono_Effect])

    Output = []
    Output.append(energies.tolist())
    Output.append(source.tolist())
    Output.append(Mono_Effect)
    Output.append(Final_Spectrum)
    Output = numpy.array(Output)

    if FILE_DUMP == 1:
        f = open(output_file, 'w')
        f.write("#S 1\n#N 4\n#L Photon energy [eV]  source  monochromator reflectivity  transmitted intensity\n")
        for i in range(Output.shape[1]):
            f.write("%g  %g  %g  %g\n" % (Output[0, i], Output[1, i], Output[2, i], Output[3, i]))
        f.close()
        print("File written to disk: %s" % output_file)

    #
    # text results
    #
    txt = "\n\n"
    txt += "*************************** power results ******************\n"
    if energies[0] != energies[-1]:
        txt += "  Source energy: start=%f eV, end=%f eV, points=%d \n"%(energies[0],energies[-1],energies.size)
    else:
        txt += "  Source energy: %f eV\n"%(energies[0])
    txt += "  Number of reflections: %d\n"%(N_REFLECTIONS)

    if energies[0] != energies[-1]:
        # I0 = source[0:-1].sum()*(energies[1]-energies[0])
        I0 = numpy.trapz(source, x=energies, axis=-1)
        P0 = numpy.trapz(source / (codata.e * energies), x=energies, axis=-1)
        txt += "\n  Incoming power (integral of spectrum): %g W (%g photons)\n" % (I0, P0)

        I1 = I0
        P1 = P0
    else:
        txt += "  Incoming power: %g W (%g photons) \n"%(source[0], source[0] / (codata.e * energies))
        I0  = source[0]
        I1 = I0
        P0 = source[0] / (codata.e * energies)
        P1 = P0

    cumulated = Final_Spectrum

    if energies[0] != energies[-1]:
        I2 = numpy.trapz( cumulated, x=energies, axis=-1)
        P2 = numpy.trapz( cumulated / (codata.e * energies) , x=energies, axis=-1)
        txt += "      Outcoming power: %f  W (%g photons)\n" % (I2, P2)
        txt += "      Absorbed power:  %f W (%g photons)\n" % (I1 - I2, P1 - P2)
        txt += "      Normalized Absorbed power: %f\n" % ((I1 - I2) / I1)
        txt += "      Normalized Outcoming Power: %f\n" % (I2 / I0)
        # txt += "      Absorbed dose %f Gy.(mm^2 beam cross section)/s\n "%((I1-I2)/(dens[i]*thick[i]*1e-6))
        I1 = I2
    else:
        I2 = cumulated[0]
        P2 = cumulated[0] / (codata.e * energies)
        txt += "      Outcoming power: %f W (%g photons)\n" % (cumulated[0], P2)
        txt += "      Absorbed power: %f W (%g photons)\n" % (I1 - I2, P1 - P2)
        txt += "      Normalized Absorbed power: %f\n" % ((I1 - I2) / I1)
        txt += "      Normalized Outcoming Power: %f\n" % (I2 / I0)
        I1 = I2

    outColTitles = ["Photon energy [eV]", "Source", "reflectivity", "Spectral power after monochromator"]
    return {"data": Output, "labels": outColTitles, "info": txt}

if __name__ == "__main__":
    print(xoppy_calc_power_monochromator(TYPE=0))
    print(xoppy_calc_power_monochromator(TYPE=1))
    print(xoppy_calc_power_monochromator(TYPE=2))
    print(xoppy_calc_power_monochromator(TYPE=3, ML_H5_FILE="/users/srio/Oasys/multilayerTiC.h5", ML_GRAZING_ANGLE_DEG=0.4))