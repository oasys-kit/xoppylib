#
# file to be removed: copied to xoppy_calc_f0 (for uniformity)
#

import numpy
from dabax.common_tools import calculate_f0_from_f0coeff, atomic_number, atomic_symbols

try: import xraylib
except: pass

from dabax.dabax_xraylib import DabaxXraylib

def f0_calc(
            MAT_FLAG,
            DESCRIPTOR,
            GRIDSTART,
            GRIDEND,
            GRIDN,
            FILE_NAME="",
            charge=0.0,
            material_constants_library=None,
            ):

    qscale = numpy.linspace(GRIDSTART, GRIDEND, GRIDN)

    f0 = numpy.zeros_like(qscale)

    if MAT_FLAG == 0: # element
        descriptor = DESCRIPTOR
        # for i,iqscale in enumerate(qscale):
        Z = atomic_number(descriptor)
        if isinstance(material_constants_library, DabaxXraylib):
            coeffs = material_constants_library.f0_with_fractional_charge(Z, charge=charge)
            f0 = calculate_f0_from_f0coeff(coeffs, qscale)
        else:
            for i in range(qscale.size):
                f0[i] = material_constants_library.FF_Rayl(Z, qscale[i])

    elif MAT_FLAG == 1: # formula
        tmp = material_constants_library.CompoundParser(DESCRIPTOR)
        zetas = tmp["Elements"]
        multiplicity = tmp["nAtoms"]
        if isinstance(material_constants_library, DabaxXraylib):
            for j ,jz in enumerate(zetas):
                coeffs = material_constants_library.f0_with_fractional_charge(jz, charge=charge)
                f0 += multiplicity[j] * calculate_f0_from_f0coeff(coeffs, qscale)
        else:
            if charge != 0:
                raise Exception(NotImplementedError)

            for i in range(qscale.size):
                for j, jz in enumerate(zetas):
                    f0[i] += multiplicity[j] * material_constants_library.FF_Rayl(jz, qscale[i])

    elif MAT_FLAG == 2: # nist
        if isinstance(material_constants_library, DabaxXraylib):
            raise Exception(NotImplementedError)  # TODO: implement
        else:
            Zarray = material_constants_library.GetCompoundDataNISTByName(DESCRIPTOR)
            zetas = Zarray["Elements"]
            fractions = numpy.array(Zarray["massFractions"])

            multiplicity = []
            for i in range(fractions.size):
                Z = zetas[i]
                multiplicity.append( fractions[i] / material_constants_library.AtomicWeight(Z) )

            multiplicity = numpy.array(multiplicity)
            multiplicity /= multiplicity.min()

            atwt = 0.0
            for i in range(fractions.size):
                Z = zetas[i]
                atwt += multiplicity[i] * material_constants_library.AtomicWeight(Z)

            print("f0_calc - nist: ")
            print("    Descriptor: ", DESCRIPTOR)
            print("    Zs: ", zetas)
            print("    n: ", multiplicity)
            print("    atomic weight: ", atwt)

            for i in range(qscale.size):
                for j, jz in enumerate(zetas):
                    f0[i] += multiplicity[j] * material_constants_library.FF_Rayl(jz, qscale[i])


    else:
        raise Exception("Not implemented")

    if FILE_NAME != "":
        with open(FILE_NAME, "w") as file:
            try:
                file.write("#F %s\n " %FILE_NAME)
                file.write("\n#S 1 xoppy f0 results\n")
                file.write("#N 2\n")
                file.write("#L  q=sin(theta)/lambda [A^-1]  f0 [electron units]\n")
                for j in range(qscale.size):
                    # file.write("%19.12e  "%energy[j])
                    file.write("%19.12e  %19.12e\n " %(qscale[j] ,f0[j]))
                file.close()
                print("File written to disk: %s \n " %FILE_NAME)
            except:
                raise Exception("f0: The data could not be dumped onto the specified file!\n")
    #
    # return
    #
    return {"application" :"xoppy" ,"name" :"f0" ,"data" :numpy.vstack((qscale ,f0))
            ,"labels" :["q=sin(theta)/lambda [A^-1]" ,"f0 [electron units]"]}


if __name__ == "__main__":


    dx = DabaxXraylib()
    Si_xrl = f0_calc(0, "Si", 0, 6, 100, material_constants_library=xraylib)
    Si_dbx =   f0_calc(0, "Si", 0, 6, 100, material_constants_library=dx)

    H2O_xrl = f0_calc(1, "H2O", 0, 6, 100, material_constants_library=xraylib)
    H2O_dbx = f0_calc(1, "H2O", 0, 6, 100, material_constants_library=dx)
    #
    # H2O_xrl = XraylibDecorated.f0_calc(2, "Water, Liquid", 0, 6, 100)
    # H2O_dbx = DabaxDecorated.f0_calc(2, "Water, Liquid", 0, 6, 100)
    #

    from srxraylib.plot.gol import plot

    plot(Si_xrl["data"][0, :], Si_xrl["data"][1, :],
         Si_dbx["data"][0, :], Si_dbx["data"][1, :],
         H2O_xrl["data"][0, :], H2O_xrl["data"][1, :],
         H2O_dbx["data"][0, :], H2O_dbx["data"][1, :],
         linestyle=[None, '', None, ''],
         marker=[None, '+', None, '+'],
         color=['r', 'r', 'b', 'b'],
         legend=['Si xraylib', 'Si dabax', 'H2O xraylib', 'H2O dabax'])