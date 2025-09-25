
import numpy

try: import xraylib
except: pass

import scipy.constants as codata
from xoppylib.crystals.bragg_preprocessor_file_io import bragg_preprocessor_file_v1_write

def create_bragg_preprocessor_file_v1(interactive=True,
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
    cryst = material_constants_library.Crystal_GetCrystal(descriptor)
    volume = cryst['volume']
    volume_in_cm3 = volume * 1e-8 * 1e-8 * 1e-8  # in cm^3
    one_over_volume_times_electron_radius_in_cm = (1e0 / volume_in_cm3) * (codata_e2_mc2 * 1e2)

    # test crystal data - not needed

    if (cryst == None):
        raise Exception("Undefined crystal")
    print("  Unit cell dimensions are %f %f %f" % (cryst['a'], cryst['b'], cryst['c']))
    print("  Unit cell angles are %f %f %f" % (cryst['alpha'], cryst['beta'], cryst['gamma']))
    print("  Unit cell volume is %f A^3" % volume)
    print("  Atoms at:")
    print("     Z  fraction    X        Y        Z")
    for i in range(cryst['n_atom']):
        atom = cryst['atom'][i]
        print("    %3i %f %f %f %f" % (atom['Zatom'], atom['fraction'], atom['x'], atom['y'], atom['z']))
    print("  ")

    # dspacing
    dspacing = material_constants_library.Crystal_dSpacing(cryst, hh, kk, ll)
    dspacing_in_cm = dspacing * 1e-8
    atom = cryst['atom'] # Z's

    ga = (1e0 + 0j) + numpy.exp(1j * numpy.pi * (hh + kk)) \
         + numpy.exp(1j * numpy.pi * (hh + ll)) \
         + numpy.exp(1j * numpy.pi * (kk + ll))
    gb = ga * numpy.exp(1j * numpy.pi * 0.5 * (hh + kk + ll))
    ga_bar = ga.conjugate()
    gb_bar = gb.conjugate()
    zetas = numpy.array([atom[0]["Zatom"], atom[7]["Zatom"]])
    zeta_a = zetas[0]
    zeta_b = zetas[1]
    npoint = int((emax - emin) / estep + 1)
    for i,zeta in enumerate(zetas):
        xx01 = 1e0 / 2e0 / dspacing
        xx00 = xx01 - 0.1
        xx02 = xx01 + 0.1
        yy00 = material_constants_library.FF_Rayl(int(zeta), xx00)
        yy01 = material_constants_library.FF_Rayl(int(zeta), xx01)
        yy02 = material_constants_library.FF_Rayl(int(zeta), xx02)
        xx = numpy.array([xx00, xx01, xx02])
        yy = numpy.array([yy00, yy01, yy02])
        fit = numpy.polyfit(xx, yy, 2)
        if i == 0:
            fit_a = fit[::-1] # (tuple(fit[::-1].tolist()))
        elif i == 1:
            fit_b = fit[::-1] # (tuple(fit[::-1].tolist()))
        else:
            raise Exception("Unknown crystal structure")

    Energy = numpy.zeros(npoint)
    F1a = numpy.zeros(npoint)
    F1b = numpy.zeros(npoint)
    F2a = numpy.zeros(npoint)
    F2b = numpy.zeros(npoint)


    for i in range(npoint):
        energy = (emin + estep * i)
        Energy[i] = energy
        F1a[i] =     material_constants_library.Fi(int(zetas[0]), energy * 1e-3)
        F2a[i] = abs(material_constants_library.Fii(int(zetas[0]), energy * 1e-3))
        F1b[i] =     material_constants_library.Fi(int(zetas[1]), energy * 1e-3)
        F2b[i] = abs(material_constants_library.Fii(int(zetas[1]), energy * 1e-3))

    out_dict = {}
    # inputs
    out_dict["fileout"]    = fileout
    out_dict["descriptor"] = descriptor
    out_dict["hh"]         = hh
    out_dict["kk"]         = kk
    out_dict["ll"]         = ll
    out_dict["temper"]     = temper
    out_dict["emin"]       = emin
    out_dict["emax"]       = emax
    out_dict["estep"]      = estep
    # outputs
    out_dict["i_latt"] = 0
    out_dict["one_over_volume_times_electron_radius_in_cm"] = one_over_volume_times_electron_radius_in_cm
    out_dict["dspacing_in_cm"] = dspacing_in_cm
    out_dict["zeta_a"] = zeta_a
    out_dict["zeta_b"] = zeta_b
    out_dict["temper"] = temper
    out_dict["ga.real"] = ga.real
    out_dict["ga.imag"] = ga.imag
    out_dict["ga_bar.real"] = ga_bar.real
    out_dict["ga_bar.imag"] = ga_bar.imag

    out_dict["gb.real"] = gb.real
    out_dict["gb.imag"] = gb.imag
    out_dict["gb_bar.real"] = gb_bar.real
    out_dict["gb_bar.imag"] = gb_bar.imag

    out_dict["fit_a"] = fit_a
    out_dict["fit_b"] = fit_b
    out_dict["npoint"] = npoint
    out_dict["Energy"] = Energy
    out_dict["F1a"] = F1a
    out_dict["F2a"] = F2a
    out_dict["F1b"] = F1b
    out_dict["F2b"] = F2b

    bragg_preprocessor_file_v1_write(out_dict, fileout=fileout)

    return out_dict


if __name__ == "__main__":

    from xoppylib.crystals.bragg_preprocessor_file_io import bragg_preprocessor_file_v1_read
    from dabax.dabax_xraylib import DabaxXraylib

    for method in [0,1]:
        if method == 0:
            dx = DabaxXraylib()
            SHADOW_FILE = "bragg_v1_dabax.dat"
            tmp = create_bragg_preprocessor_file_v1(interactive=False, DESCRIPTOR="Si", H_MILLER_INDEX=1,
                                                                   K_MILLER_INDEX=1, L_MILLER_INDEX=1,
                                                                   TEMPERATURE_FACTOR=1.0, E_MIN=5000.0, E_MAX=15000.0,
                                                                   E_STEP=100.0, SHADOW_FILE=SHADOW_FILE,
                                                                   material_constants_library=dx)

        else:
            SHADOW_FILE = "bragg_v1_xraylib.dat"
            tmp = create_bragg_preprocessor_file_v1(interactive=False, DESCRIPTOR="Si", H_MILLER_INDEX=1,
                                                                   K_MILLER_INDEX=1, L_MILLER_INDEX=1,
                                                                   TEMPERATURE_FACTOR=1.0, E_MIN=5000.0, E_MAX=15000.0,
                                                                   E_STEP=100.0, SHADOW_FILE=SHADOW_FILE,
                                                                   material_constants_library=xraylib)



        tmp1 = bragg_preprocessor_file_v1_read(SHADOW_FILE)

        for key in tmp.keys():
            # print("---------------", key)
            oo = tmp[key]
            try:
                oo1 = tmp1[key]
                # print(type(oo), type(oo1))
                if isinstance(oo, list):
                    print(key,oo[0],oo1[0])
                elif isinstance(oo, numpy.ndarray):
                    print(key,oo[0],oo1[0])
                else:
                    print(key,oo,oo1)
            except:
                print("key %s not found " % key)