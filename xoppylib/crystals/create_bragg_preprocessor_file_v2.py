
import numpy

try: import xraylib
except: pass

import scipy.constants as codata

# needed by bragg_calc
from dabax.common_tools import f0_xop

# needed by bragg_calc
from dabax.common_tools import bragg_metrictensor, atomic_symbols
from xoppylib.crystals.tools import bragg_calc2

def create_bragg_preprocessor_file_v2(interactive=True,
        DESCRIPTOR="Si", H_MILLER_INDEX=1, K_MILLER_INDEX=1, L_MILLER_INDEX=1, TEMPERATURE_FACTOR=1.0,
        E_MIN=5000.0, E_MAX=15000.0, E_STEP=100.0,
        SHADOW_FILE="bragg.dat",
        material_constants_library=None):

    """
     SHADOW preprocessor for crystals - python+xraylib version

     -"""
    if material_constants_library is None:
        try:    material_constants_library = xraylib
        except: material_constants_library = DabaxXraylib()
    # codata_e2_mc2 = 2.81794032e-15 = Classical electron radius in S.I.
    codata_e2_mc2 = codata.hbar * codata.alpha / codata.m_e / codata.c

    if interactive:
        print("bragg: SHADOW preprocessor for crystals - python+xraylib version")
        fileout = input("Name of output file : ")

        print(" bragg (python) only works now for ZincBlende Cubic structures. ")
        print(" Valid descriptor are: ")
        print("     Si (alternatively Si_NIST, Si2) ")
        print("     Ge")
        print("     Diamond")
        print("     GaAs, GaSb, GaP")
        print("     InAs, InP, InSb")
        print("     SiC")

        descriptor = input("Name of crystal descriptor : ")

        print("Miller indices of crystal plane of reeflection.")
        miller = input("H K L: ")
        miller = miller.split()
        hh = int(miller[0])
        kk = int(miller[1])
        ll = int(miller[2])

        temper = input("Temperature (Debye-Waller) factor (set 1 for default): ")
        temper = float(temper)

        emin = input("minimum photon energy (eV): ")
        emin = float(emin)
        emax = input("maximum photon energy (eV): ")
        emax = float(emax)
        estep = input("energy step (eV): ")
        estep = float(estep)

    else:
        fileout    = SHADOW_FILE
        descriptor = DESCRIPTOR
        hh         = int(H_MILLER_INDEX)
        kk         = int(K_MILLER_INDEX)
        ll         = int(L_MILLER_INDEX)
        temper     = float(TEMPERATURE_FACTOR)
        emin       = float(E_MIN)
        emax       = float(E_MAX)
        estep      = float(E_STEP)


    #
    # end input section, start calculations
    #

    out_dict = bragg_calc2(descriptor=descriptor,
                          hh=hh,kk=kk,ll=ll,temper=temper,
                          emin=emin,emax=emax,estep=estep,fileout=fileout,
                          material_constants_library=material_constants_library)

    # dump_bragg_preprocessor_file_v1(out_dict, fileout=fileout)

    return out_dict


if __name__ == "__main__":

    from xoppylib.crystals.bragg_preprocessor_file_io import bragg_preprocessor_file_v2_read
    from dabax.dabax_xraylib import DabaxXraylib
    for method in [0,1]:
        if method == 0:
            dx = DabaxXraylib()
            SHADOW_FILE = "bragg_v2_dabax.dat"
            tmp = create_bragg_preprocessor_file_v2(interactive=False, DESCRIPTOR="Si", H_MILLER_INDEX=1, K_MILLER_INDEX=1, L_MILLER_INDEX=1,
                  TEMPERATURE_FACTOR=1.0, E_MIN=5000.0, E_MAX=15000.0, E_STEP=100.0, SHADOW_FILE=SHADOW_FILE,
                  material_constants_library=dx)

        else:
            SHADOW_FILE = "bragg_v2_xraylib.dat"
            tmp = create_bragg_preprocessor_file_v2(interactive=False, DESCRIPTOR="Si", H_MILLER_INDEX=1,
                                                                   K_MILLER_INDEX=1, L_MILLER_INDEX=1,
                                                                   TEMPERATURE_FACTOR=1.0, E_MIN=5000.0, E_MAX=15000.0,
                                                                   E_STEP=100.0, SHADOW_FILE=SHADOW_FILE,
                                                                   material_constants_library=xraylib)



        tmp1 = bragg_preprocessor_file_v2_read(SHADOW_FILE)

        for key in tmp1.keys():
            print("---------------", key)

        for key in tmp.keys():
            print("---------------", key)
            oo = tmp[key]
            try:
                oo1 = tmp1[key]
                print(type(oo), type(oo1))
                if isinstance(oo, list):
                    pass
                    # print(key,oo[0][0],oo1[0][0])
                elif isinstance(oo, numpy.ndarray):
                    print(key,oo[0][0],oo1[0][0])
                else:
                    print(key,oo,oo1)
            except:
                print("key %s not found " % key)