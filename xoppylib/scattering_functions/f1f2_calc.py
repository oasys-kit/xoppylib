import numpy
import scipy.constants as codata

from xoppylib.scattering_functions.fresnel import interface_reflectivity

try: import xraylib
except: pass

from dabax.dabax_xraylib import DabaxXraylib


def f1f2_calc(descriptor, energy, theta=3.0e-3, F=0, density=None, rough=0.0, verbose=True,
              material_constants_library=None,):
    """
    calculate the elastic Photon-Atom anonalous f1 and f2  coefficients as a function of energy.
    It also gives the refractive index components delta and beta (n=1-delta - i beta),
    the absorption photoelectric coefficient and the reflectivities (s,p and unpolarized).
    :param descriptor: string with the element symbol or integer with Z
    :param energy: array with energies (eV)
    :param theta: array with grazing angles (rad)
    :param F: calculation flag:

           F=0 (default) returns a 2-col array with f1 and f2
           F=1  returns f1
           F=2  returns f2
           F=3  returns delta  [n = 1 -delta -i beta]
           F=4  returns betaf  [n = 1 -delta -i beta]
           F=5  returns Photoelectric linear absorption coefficient
           F=6  returns Photoelectric mass absorption coefficient
           F=7  returns Photoelectric Cross Section
           F=8  returns s-polarized reflectivity
           F=9  returns p-polarized reflectivity
           F=10  returns unpolarized reflectivity
           F=11  returns delta/betaf
           F=12  returns delta calculated with F1
           F=13  returns beta calculated with F2

    :param density: the density to be used for some calculations. If None, get it from xraylib
    :param rough: the roughness RMS in Angstroms for reflectivity calculations
    :return: a numpy array with results
    """

    energy = numpy.array(energy, dtype=float).reshape(-1)
    theta = numpy.array(theta, dtype=float).reshape(-1)

    if isinstance(descriptor, str):
        Z = material_constants_library.SymbolToAtomicNumber(descriptor)
        symbol = descriptor
    else:
        Z = descriptor
        symbol = material_constants_library.AtomicNumberToSymbol(descriptor)

    if density is None:
        density = material_constants_library.ElementDensity(Z)

    if verbose:
        print("f1f2_calc: using density: %f g/cm3" % density)

    if F == 0:  # F=0 (default) returns a 2-col array with f1 and f2
        out = numpy.zeros((2, energy.size))
        if isinstance(material_constants_library,DabaxXraylib):
            tmp1 = Z + material_constants_library.Fi(Z, 1e-3 * energy)
            tmp2 = - material_constants_library.Fii(Z, 1e-3 * energy)
            out[0, :] = tmp1
            out[1, :] = tmp2
        else:
            for i, ienergy in enumerate(energy):
                out[0, i] = Z + material_constants_library.Fi(Z, 1e-3 * ienergy)
                out[1, i] = - material_constants_library.Fii(Z, 1e-3 * ienergy)
    elif F == 1:  # F=1  returns f1
        if isinstance(material_constants_library,DabaxXraylib):
            out = Z + material_constants_library.Fi(Z, 1e-3 * energy)
        else:
            out = numpy.zeros_like(energy)
            for i, ienergy in enumerate(energy):
                out[i] = Z + material_constants_library.Fi(Z, 1e-3 * ienergy)
    elif F == 2:  # F=2  returns f2
        if isinstance(material_constants_library,DabaxXraylib):
            out = - material_constants_library.Fii(Z, 1e-3 * energy)
        else:
            out = numpy.zeros_like(energy)
            for i, ienergy in enumerate(energy):
                out[i] = - material_constants_library.Fii(Z, 1e-3 * ienergy)
    elif F == 3:  # F=3  returns delta  [n = 1 -delta -i beta]
        if isinstance(material_constants_library,DabaxXraylib):
            out = (1e0 - material_constants_library.Refractive_Index_Re(symbol, 1e-3 * energy, density))
        else:
            out = numpy.zeros_like(energy)
            for i, ienergy in enumerate(energy):
                out[i] = (1e0 - material_constants_library.Refractive_Index_Re(symbol, 1e-3 * ienergy, density))
    elif F == 4:  # F=4  returns betaf  [n = 1 -delta -i beta]
        if isinstance(material_constants_library,DabaxXraylib):
            out = material_constants_library.Refractive_Index_Im(symbol, 1e-3 * energy, density)
        else:
            out = numpy.zeros_like(energy)
            for i, ienergy in enumerate(energy):
                out[i] = material_constants_library.Refractive_Index_Im(symbol, 1e-3 * ienergy, density)
    elif F == 5:  # F=5  returns Photoelectric linear absorption coefficient
        if isinstance(material_constants_library,DabaxXraylib):
            out = density * material_constants_library.CS_Photo(Z, 1e-3 * energy)
        else:
            out = numpy.zeros_like(energy)

            for i, ienergy in enumerate(energy):
                out[i] = density * material_constants_library.CS_Photo(Z, 1e-3 * ienergy)
    elif F == 6:  # F=6  returns Photoelectric mass absorption coefficient
        if isinstance(material_constants_library,DabaxXraylib):
            out = material_constants_library.CS_Photo(Z, 1e-3 * energy)
        else:
            out = numpy.zeros_like(energy)
            for i, ienergy in enumerate(energy):
                out[i] = material_constants_library.CS_Photo(Z, 1e-3 * ienergy)
    elif F == 7:  # F=7  returns Photoelectric Cross Section
        if isinstance(material_constants_library,DabaxXraylib):
            out = material_constants_library.CSb_Photo(Z, 1e-3 * energy)
        else:
            out = numpy.zeros_like(energy)
            for i, ienergy in enumerate(energy):
                out[i] = material_constants_library.CSb_Photo(Z, 1e-3 * ienergy)
    elif F == 11:  # F=11  returns delta/betaf
        if isinstance(material_constants_library,DabaxXraylib):
            out = (1e0 - material_constants_library.Refractive_Index_Re(symbol, 1e-3 * energy, density))
            out /= material_constants_library.Refractive_Index_Im(symbol, 1e-3 * energy, density)
        else:
            out = numpy.zeros_like(energy)
            for i, ienergy in enumerate(energy):
                out[i] = (1e0 - material_constants_library.Refractive_Index_Re(symbol, 1e-3 * ienergy, density))
                out[i] /= material_constants_library.Refractive_Index_Im(symbol, 1e-3 * ienergy, density)

    if (F >= 8 and F <= 10) or (F >= 12 and F <= 13):  # reflectivities
        atwt = material_constants_library.AtomicWeight(Z)
        avogadro = codata.Avogadro
        toangstroms = codata.h * codata.c / codata.e * 1e10
        re = codata.e ** 2 / codata.m_e / codata.c ** 2 / (4 * numpy.pi * codata.epsilon_0) * 1e2  # in cm

        molecules_per_cc = density * avogadro / atwt
        wavelength = toangstroms / energy * 1e-8  # in cm
        k = molecules_per_cc * re * wavelength * wavelength / 2.0 / numpy.pi

        if isinstance(material_constants_library,DabaxXraylib):
            f1 = Z + material_constants_library.Fi(Z, 1e-3 * energy)
            f2 = - material_constants_library.Fii(Z, 1e-3 * energy)
        else:
            f1 = numpy.zeros_like(energy)
            f2 = numpy.zeros_like(energy)
            for i, ienergy in enumerate(energy):
                f1[i] = Z + material_constants_library.Fi(Z, 1e-3 * ienergy)
                f2[i] = - material_constants_library.Fii(Z, 1e-3 * ienergy)

        alpha = 2.0 * k * f1
        gamma = 2.0 * k * f2

        rs, rp, runp = interface_reflectivity(alpha, gamma, theta)

        if rough != 0:
            rough *= 1e-8  # to cm
            debyewaller = numpy.exp(-(4.0 * numpy.pi * numpy.sin(theta) * rough / wavelength) ** 2)
        else:
            debyewaller = 1.0
        if F == 8:  # returns s-polarized reflectivity
            out = rs * debyewaller
        elif F == 9:  # returns p-polarized reflectivity
            out = rp * debyewaller
        elif F == 10:  # returns unpolarized reflectivity
            out = runp * debyewaller
        elif F == 12:  # returns delta as calculated from f1
            out = alpha / 2
        elif F == 13:  # returns beta as calculated from f2
            out = gamma / 2

    return out


def f1f2_calc_mix(descriptor, energy, theta=3.0e-3, F=0, density=None, rough=0.0, verbose=True,
                  material_constants_library=None,):
    """
    Like f1f2_calc but for a chemical formula. S

    :param descriptor: string with the element symbol or integer with Z
    :param energy: array with energies (eV)
    :param theta: array with grazing angles (rad)
    :param F: calculation flag:

           F=0 (default) returns a 2-col array with f1 and f2
           F=1  returns f1
           F=2  returns f2
           F=3  returns delta  [n = 1 -delta -i beta]
           F=4  returns betaf  [n = 1 -delta -i beta]
           F=5  returns Photoelectric linear absorption coefficient
           F=6  returns Photoelectric mass absorption coefficient
           F=7  returns Photoelectric Cross Section
           F=8  returns s-polarized reflectivity
           F=9  returns p-polarized reflectivity
           F=10  returns unpolarized reflectivity
           F=11  returns delta/betaf
    :param density: the density to be used for some calculations.
    :param rough: the roughness RMS in Angstroms for reflectivity calculations
    :return: a numpy array with results
    """
    energy = numpy.array(energy, dtype=float).reshape(-1)

    Zarray = material_constants_library.CompoundParser(descriptor)
    # Zarray = parse_formula(descriptor)
    zetas = Zarray["Elements"]
    weights = numpy.array(Zarray["nAtoms"])
    atwt = Zarray["molarMass"]
    print("molarMass: %g" % atwt)

    if verbose: print("f1f2_calc_mix: Zs: ", zetas, " n: ", weights)

    f1 = numpy.zeros_like(energy)
    f2 = numpy.zeros_like(energy)
    for i, zi in enumerate(zetas):
        f1i = f1f2_calc(material_constants_library.AtomicNumberToSymbol(zi), energy, theta, F=1, verbose=False,
                        material_constants_library=material_constants_library)
        f2i = f1f2_calc(material_constants_library.AtomicNumberToSymbol(zi), energy, theta, F=2, verbose=False,
                        material_constants_library=material_constants_library)
        f1 += f1i * weights[i]
        f2 += f2i * weights[i]

    if F == 0:
        return numpy.vstack((f1, f2))
    elif F == 1:
        return f1
    elif F == 2:
        return f2

    if density is None:
        raise Exception("Please define density.")

    avogadro = codata.Avogadro
    toangstroms = codata.h * codata.c / codata.e * 1e10
    re = codata.e ** 2 / codata.m_e / codata.c ** 2 / (4 * numpy.pi * codata.epsilon_0) * 1e2  # in cm

    molecules_per_cc = density * avogadro / atwt
    wavelength = toangstroms / energy * 1e-8  # in cm
    k = molecules_per_cc * re * wavelength * wavelength / 2.0 / numpy.pi

    # ;
    # ; calculation of refraction index
    # ;

    delta = k * f1
    beta = k * f2
    mu = 4.0 * numpy.pi * beta / wavelength

    if F == 3: return delta
    if F == 4: return beta
    if F == 5: return mu
    if F == 6: return mu / density
    if F == 7: return mu / molecules_per_cc * 1e24
    if F == 11: return delta / beta

    #
    # interface reflectivities
    #

    alpha = 2.0 * k * f1
    gamma = 2.0 * k * f2

    rs, rp, runp = interface_reflectivity(alpha, gamma, theta)

    if rough != 0:
        rough *= 1e-8  # to cm
        debyewaller = numpy.exp(-(4.0 * numpy.pi * numpy.sin(theta) * rough / wavelength) ** 2)
    else:
        debyewaller = 1.0

    if F == 8: return rs * debyewaller  # returns s-polarized reflectivity
    if F == 9: return rp * debyewaller  # returns p-polarized reflectivity
    if F == 10: return runp * debyewaller  # returns unpolarized reflectivity

    raise Exception("Why am I here? ")


def f1f2_calc_nist(descriptor, energy, theta=3.0e-3, F=0, density=None, rough=0.0, verbose=True,
                   material_constants_library=None,):
    """
    Like f1f2_calc but for a compound defined in the NIST list
        See list here:  http://lvserver.ugent.be/xraylib-web/index.php?xrlFunction=GetCompoundDataNISTList&Element=26&ElementOrCompound=FeSO4&Compound=Ca5%28PO4%293&AugerTransa=K&AugerTransb=L2&AugerTransc=M3&LinenameSwitch=IUPAC&Linename1a=K&Linename1b=L3&Linename2=KA1_LINE&NISTcompound=Gadolinium+Oxysulfide&RadioNuclide=55Fe&Shell=K_SHELL&Energy=10.0&Theta=1.5707964&Phi=3.14159&MomentumTransfer=0.57032&CKTrans=FL12_TRANS&Density=1.0&PZ=1.0&submit=Go%21&Language=C

    :param descriptor: string with the name of compound as in NIST table
    :param energy: array with energies (eV)
    :param theta: array with grazing angles (rad)
    :param F: calculation flag:

           F=0 (default) returns a 2-col array with f1 and f2
           F=1  returns f1
           F=2  returns f2
           F=3  returns delta  [n = 1 -delta -i beta]
           F=4  returns betaf  [n = 1 -delta -i beta]
           F=5  returns Photoelectric linear absorption coefficient
           F=6  returns Photoelectric mass absorption coefficient
           F=7  returns Photoelectric Cross Section
           F=8  returns s-polarized reflectivity
           F=9  returns p-polarized reflectivity
           F=10  returns unpolarized reflectivity
           F=11  returns delta/betaf
    :param density: the density. If None, take from xraylib
    :param rough: the roughness RMS in Angstroms for reflectivity calculations
    :return: a numpy array with results
    """
    energy = numpy.array(energy, dtype=float).reshape(-1)

    # >>>> name
    # >>>> nElements
    # >>>> density
    # >>>> Elements
    # >>>> massFractions
    # {'name': 'Alanine', 'nElements': 4, 'density': 1.42, 'Elements': [1, 6, 7, 8],
    # 'massFractions': [0.07919, 0.404439, 0.157213, 0.359159]}
    Zarray = material_constants_library.GetCompoundDataNISTByName(descriptor)

    # Zarray = parse_formula(descriptor)
    zetas = Zarray["Elements"]
    fractions = numpy.array(Zarray["massFractions"])
    if density is None:
        density = Zarray["density"]

    weights = []
    for i in range(fractions.size):
        weights.append(fractions[i] / material_constants_library.AtomicWeight(zetas[i]))

    weights = numpy.array(weights)
    weights /= weights.min()

    atwt = 0.0
    for i in range(fractions.size):
        atwt += weights[i] * material_constants_library.AtomicWeight(zetas[i])

    if verbose:
        print("f1f2_calc_nist: ")
        print("    Descriptor: ", descriptor)
        print("    Zs: ", zetas)
        print("    n: ", weights)
        print("    atomic weight: ", atwt)
        print("Density: ", density)

    f1 = numpy.zeros_like(energy)
    f2 = numpy.zeros_like(energy)
    for i, zi in enumerate(zetas):
        f1i = f1f2_calc(material_constants_library.AtomicNumberToSymbol(zi), energy, theta, F=1, verbose=False,
                        material_constants_library=material_constants_library)
        f2i = f1f2_calc(material_constants_library.AtomicNumberToSymbol(zi), energy, theta, F=2, verbose=False,
                        material_constants_library=material_constants_library)
        f1 += f1i * weights[i]
        f2 += f2i * weights[i]

    if F == 0:
        return numpy.vstack((f1, f2))
    elif F == 1:
        return f1
    elif F == 2:
        return f2

    avogadro = codata.Avogadro
    toangstroms = codata.h * codata.c / codata.e * 1e10
    re = codata.e ** 2 / codata.m_e / codata.c ** 2 / (4 * numpy.pi * codata.epsilon_0) * 1e2  # in cm

    molecules_per_cc = density * avogadro / atwt
    wavelength = toangstroms / energy * 1e-8  # in cm
    k = molecules_per_cc * re * wavelength * wavelength / 2.0 / numpy.pi

    # ;
    # ; calculation of refraction index
    # ;

    delta = k * f1
    beta = k * f2
    mu = 4.0 * numpy.pi * beta / wavelength
    if F == 3: return delta
    if F == 4: return beta
    if F == 5: return mu
    if F == 6: return mu / density
    if F == 7: return mu / molecules_per_cc * 1e24
    if F == 11: return delta / beta

    #
    # interface reflectivities
    #

    alpha = 2.0 * k * f1
    gamma = 2.0 * k * f2

    rs, rp, runp = interface_reflectivity(alpha, gamma, theta)

    if rough != 0:
        rough *= 1e-8  # to cm
        debyewaller = numpy.exp(-(4.0 * numpy.pi * numpy.sin(theta) * rough / wavelength) ** 2)
    else:
        debyewaller = 1.0

    if F == 8: return rs * debyewaller  # returns s-polarized reflectivity
    if F == 9: return rp * debyewaller  # returns p-polarized reflectivity
    if F == 10: return runp * debyewaller  # returns unpolarized reflectivity

    raise Exception("Why am I here? ")