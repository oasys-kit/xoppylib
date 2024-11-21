import numpy
import xraylib
import scipy.constants as codata

# needed by bragg_calc
from xoppylib.crystals.bragg_preprocessor_file_io import bragg_preprocessor_file_v2_write
from dabax.common_tools import f0_xop, f0_xop_with_fractional_charge
from dabax.common_tools import bragg_metrictensor, lorentz, atomic_symbols


import sys
import os
import platform
from xoppylib.xoppy_util import locations
from dabax.dabax_xraylib import DabaxXraylib

#
#
#
def bragg_metrictensor(a, b, c, a1, a2, a3, RETURN_REAL_SPACE=0, RETURN_VOLUME=0, HKL=None):
    """
    Returns the metric tensor in the reciprocal space.

    Parameters
    ----------
    a: float
        unit cell a
    b: float
        unit cell b
    c: float
        unit cell c
    a1: float
        unit cell alpha [in deg]
    a2: float
        unit cell beta [in deg]
    a3: float
        unit cell gamma [in deg]
    RETURN_REAL_SPACE: int, optional
        set to 1 for returning metric tensor in real space.
    RETURN_VOLUME: int, optional
        set to 1 to return the unit cell volume in Angstroms^3.
    HKL: None, or list
        if !=None, returns the d-spacing for the corresponding [H,K,L] reflection.

    Returns
    -------
    The returned value depends on the keywords used. If RETURN_REAL_SPACE=0, RETURN_VOLUME=0, and HKL=None
             then retuns the metric tensor in reciprocal space.

    """

    # input cell a,b,c,alpha,beta,gamma; angles in degrees
    a1 *= numpy.pi / 180.0
    a2 *= numpy.pi / 180.0
    a3 *= numpy.pi / 180.0
    # ;
    # ; tensor in real space
    # ;
    g = numpy.array( [ [a*a, a*b*numpy.cos(a3), a*c*numpy.cos(a2)], \
          [a*b*numpy.cos(a3), b*b, b*c*numpy.cos(a1)], \
          [a*c*numpy.cos(a2), b*c*numpy.cos(a1), c*c]] )

    if RETURN_REAL_SPACE: return g
    # print("g: ",g)

    # ;
    # ; volume of the lattice
    # ;
    volume2 = numpy.linalg.det(g)
    volume = numpy.sqrt(volume2)

    # print("Volume of unit cell: %g A^3",volume)

    if RETURN_VOLUME: return volume

    # ;
    # ; tensor in reciprocal space
    # ;
    ginv = numpy.linalg.inv(g)
    # ;print,gInv
    #

    # itmp = where(abs(ginv) LT 1d-8)
    # IF itmp[0] NE -1 THEN ginv[itmp]=0D

    itmp = numpy.where(numpy.abs(ginv) < 1e-8)
    ginv[itmp] = 0.0

    # print("ginv: ",ginv)

    if HKL != None:
    #   ; computes d-spacing
        dd = numpy.dot( numpy.array(HKL) , numpy.dot( ginv , numpy.array(HKL)))
        #
        # print("DD: ", dd)
        dd1 = 1.0 / numpy.sqrt(dd)
        # print("D-spacing: ",dd1)
        return dd1
    else:
        return ginv

def lorentz(theta_bragg_deg, return_what=0):
    """
    This function returns the Lorentz factor, polarization factor (unpolarized beam), geometric factor,
    or a combination of them.

    Parameters
    ----------
    theta_bragg_deg: float
        Bragg angle in degrees.
    return_what, int
        0: (default) PolFac*lorentzFac
        1: PolFac
        2: lorentzFac
        3: geomFac

    Returns
    -------
    float

    """
    tr = theta_bragg_deg * numpy.pi / 180.
    polarization_factor = 0.5 * (1.0 + (numpy.cos(2.0 * tr))**2)
    lorentz_factor = 1.0 / numpy.sin(2.0 * tr)
    geometrical_factor = 1.0 * numpy.cos(tr) / numpy.sin(2.0 * tr)

    if return_what == 0:
        return polarization_factor*lorentz_factor
    elif return_what == 1:
        return polarization_factor
    elif return_what == 2:
        return lorentz_factor
    elif return_what == 3:
        return geometrical_factor
    elif return_what == 4:
        return polarization_factor*lorentz_factor*geometrical_factor

# OBSOLETE.... USE bragg_calc2() INSTEAD!
def bragg_calc(descriptor="Si", hh=1, kk=1, ll=1, temper=1.0, emin=5000.0, emax=15000.0, estep=100.0, fileout=None,
               material_constants_library=xraylib):
    """
    OBSOLETE.... USE bragg_calc2() INSTEAD!
    Preprocessor for Structure Factor (FH) calculations. It calculates the basic ingredients of FH.



    Parameters
    ----------
    descriptor: str, optional
        crystal name (as in xraylib)
    hh: int, optional
        miller index H
    kk: int, optional
        miller index K
    ll: int, optional
        miller index L
    temper: float, optional
        temperature factor (scalar <=1.0 )
    emin: float, optional
        photon energy minimum in eV
    emax: float, optional
        photon energy maximum in eV
    estep: float, optional
        photon energy step in eV
    fileout: None or str, optional
        name for the output file (default=None, no output file)
    material_constants_library: xraylib or instance of DabaxXraylib, optional
        The pointer to the material library to be used to retrieve scattering data.

    Returns
    -------
    dict

    """


    output_dictionary = {}

    codata_e2_mc2 = codata.e**2 / codata.m_e / codata.c**2 / (4*numpy.pi*codata.epsilon_0)  # in m

    # f = open(fileout,'w')

    version = "2.5"
    output_dictionary["version"] = version

    # todo: txt not longer used here... can be removed
    txt = ""
    txt += "# Bragg version, Data file type\n"
    txt += "%s 1\n" % version



    cryst = material_constants_library.Crystal_GetCrystal(descriptor)

    if cryst is None:
        raise Exception("Crystal not found in xraylib: %s" % descriptor )

    volume = cryst['volume']

    # crystal data - not needed
    icheck = 0
    if icheck:

        print ("  Unit cell dimensions are %f %f %f" % (cryst['a'],cryst['b'],cryst['c']))
        print ("  Unit cell angles are %f %f %f" % (cryst['alpha'],cryst['beta'],cryst['gamma']))
        print ("  Unit cell volume is %f A^3" % volume )
        print ("  Atoms at:")
        print ("     Z  fraction    X        Y        Z")
        for i in range(cryst['n_atom']):
            atom =  cryst['atom'][i]
            print ("    %3i %f %f %f %f" % (atom['Zatom'], atom['fraction'], atom['x'], atom['y'], atom['z']) )
        print ("  ")

    volume = volume*1e-8*1e-8*1e-8 # in cm^3
    dspacing = material_constants_library.Crystal_dSpacing(cryst, hh, kk, ll)
    rn = (1e0/volume)*(codata_e2_mc2*1e2)
    dspacing *= 1e-8 # in cm

    txt += "# RN = (e^2/(m c^2))/V) [cm^-2], d spacing [cm]\n"
    txt += "%e %e \n" % (rn , dspacing)

    output_dictionary["rn"] = rn
    output_dictionary["dspacing"] = dspacing

    atom = cryst['atom']
    list_Zatom = [ atom[i]['Zatom'] for i in range(len(atom))]
    number_of_atoms = len(list_Zatom)
    list_fraction = [ atom[i]['fraction'] for i in range(len(atom))]
    try:
        list_charge = [atom[i]['charge'] for i in range(len(atom))]
    except:
        list_charge = [0.0] * number_of_atoms
    list_x = [ atom[i]['x'] for i in range(len(atom))]
    list_y = [ atom[i]['y'] for i in range(len(atom))]
    list_z = [ atom[i]['z'] for i in range(len(atom))]

    # creates an is that contains Z, occupation and charge, that will
    # define the different sites.
    IDs = []
    number_of_atoms = len(list_Zatom)
    for i in range(number_of_atoms):
        IDs.append("Z:%2d-F:%g-C:%g" % (list_Zatom[i],list_fraction[i], list_charge[i]))

    # calculate indices of uniqte Id's sorted by Z
    unique_indexes1 = numpy.unique(IDs, return_index=True) [1]
    unique_Zatom1 = [list_Zatom[i] for i in unique_indexes1]
    # sort by Z
    ii = numpy.argsort(unique_Zatom1)
    unique_indexes = unique_indexes1[ii]

    unique_Zatom = [list_Zatom[i] for i in unique_indexes]
    unique_charge = [list_charge[i] for i in unique_indexes]
    unique_scattering_electrons = []
    for i, Zi in enumerate(unique_Zatom):
        unique_scattering_electrons.append(Zi - unique_charge[i])

    nbatom = (len(unique_Zatom))

    txt += "# Number of different element-sites in unit cell NBATOM:\n%d \n" % nbatom
    output_dictionary["nbatom"] = nbatom

    txt += "# for each element-site, the number of scattering electrons (Z_i + charge_i)\n"
    for i in unique_Zatom:
        txt += "%d "%i
    txt += "\n"
    output_dictionary["atnum"] = list(unique_scattering_electrons)

    txt += "# for each element-site, the occupation factor\n"
    unique_fraction = []
    for i in range(len(unique_indexes)):
        unique_fraction.append(list_fraction[unique_indexes[i]])
        txt += "%g "%(unique_fraction[i])
    txt += "\n"
    output_dictionary["fraction"] = unique_fraction


    txt += "# for each element-site, the temperature factor\n" # temperature parameter
    list_temper = []
    for i in range(len(unique_indexes)):
        txt += "%5.3f "%temper
        list_temper.append(temper)
    txt += "\n"
    output_dictionary["temper"] = list_temper

    #
    # Geometrical part of structure factor:  G and G_BAR
    #
    txt += "# for each type of element-site, COOR_NR=G_0\n"
    list_multiplicity = []
    for i in range(len(unique_indexes)):
        id = IDs[unique_indexes[i]]
        txt += "%d "%IDs.count(id)
        list_multiplicity.append(IDs.count(id))
    txt += "\n"
    output_dictionary["G_0"] = list_multiplicity

    txt += "# for each type of element-site, G and G_BAR (both complex)\n"
    list_g = []
    list_g_bar = []
    for i in range(len(unique_indexes)):
        id = IDs[unique_indexes[i]]
        ga = 0.0 + 0j
        for i,zz in enumerate(IDs):
            if zz == id:
                ga += numpy.exp(2j*numpy.pi*(hh*list_x[i]+kk*list_y[i]+ll*list_z[i]))
        txt += "(%g,%g) \n"%(ga.real,ga.imag)
        txt += "(%g,%g) \n"%(ga.real,-ga.imag)
        list_g.append(ga)
        list_g_bar.append(ga.conjugate())
    output_dictionary["G"] = list_g
    output_dictionary["G_BAR"] = list_g_bar

    #
    # F0 part
    #
    txt += "# for each type of element-site, the number of f0 coefficients followed by them\n"
    list_f0 = []
    for i in range(len(unique_indexes)):
        zeta = list_Zatom[unique_indexes[i]]
        tmp = f0_xop(zeta)
        txt += ("11 "+"%g "*11+"\n")%(tuple(tmp))
        list_f0.append(tmp.tolist())
    output_dictionary["f0coeff"] = list_f0


    npoint  = int( (emax - emin)/estep + 1 )
    txt += "# The number of energy points NPOINT: \n"
    txt +=  ("%i \n") % npoint
    output_dictionary["npoint"] = npoint
    txt += "# for each energy point, energy, F1(1),F2(1),...,F1(nbatom),F2(nbatom)\n"
    list_energy = []
    out_f1 = numpy.zeros( (len(unique_indexes),npoint), dtype=float)
    out_f2 = numpy.zeros( (len(unique_indexes),npoint), dtype=float)
    out_fcompton = numpy.zeros( (len(unique_indexes),npoint), dtype=float) # todo is complex?
    for i in range(npoint):
        energy = (emin+estep*i)
        txt += ("%20.11e \n") % (energy)
        list_energy.append(energy)

        for j in range(len(unique_indexes)):
            zeta = list_Zatom[unique_indexes[j]]
            f1a =  material_constants_library.Fi(int(zeta),energy*1e-3)
            f2a = -material_constants_library.Fii(int(zeta),energy*1e-3) # TODO: check the sign!!
            txt +=  (" %20.11e %20.11e 1.000 \n")%(f1a, f2a)
            out_f1[j,i] = f1a
            out_f2[j,i] = f2a
            out_fcompton[j,i] = 1.0

    output_dictionary["energy"] = list_energy
    output_dictionary["f1"] = out_f1
    output_dictionary["f2"] = out_f2
    output_dictionary["fcompton"] = out_fcompton

    if fileout != None:
        bragg_preprocessor_file_v2_write(output_dictionary, fileout)
        # with open(fileout,"w") as f:
        #     f.write(txt)
        #     print("File written to disk: %s" % fileout)

    return output_dictionary

#
#
#
def crystal_fh(input_dictionary, phot_in, theta=None, forceratio=0):
    """
    Calculates the crystal structure factors.

    Parameters
    ----------
    input_dictionary: dict
        as resulting from bragg_calc()
    phot_in: float, numpy array
        The photon energy in eV
    theta: None or float or numpy array.
        The incident angle (half of scattering angle) in rad. If None, it uses the Bragg angle.
    forceratio: int, optional
        0: calculates ratio = numpy.sin(itheta[i]) / (toangstroms / phot)
        1: calculates ratio = 1 / (2 * dspacing * 1e8)

    Returns
    -------
    dict
        a dictionary with structure factor.

    """
    rn        = input_dictionary["rn"]
    dspacing  = numpy.array(input_dictionary["dspacing"])
    nbatom    = numpy.array(input_dictionary["nbatom"])
    atnum     = numpy.array(input_dictionary["atnum"])
    temper    = numpy.array(input_dictionary["temper"])
    G_0       = numpy.array(input_dictionary["G_0"])
    G         = numpy.array(input_dictionary["G"])
    G_BAR     = numpy.array(input_dictionary["G_BAR"])
    f0coeff   = numpy.array(input_dictionary["f0coeff"])
    npoint    = numpy.array(input_dictionary["npoint"])
    energy    = numpy.array(input_dictionary["energy"])
    fp        = numpy.array(input_dictionary["f1"])
    fpp       = numpy.array(input_dictionary["f2"])
    fraction = numpy.array(input_dictionary["fraction"])

    phot_in = numpy.array(phot_in, dtype=float).reshape(-1)
    toangstroms = codata.h * codata.c / codata.e * 1e10

    if theta is None:
        itheta = numpy.arcsin(toangstroms * 1e-8 / phot_in / 2 / dspacing)
    else:
        theta = numpy.array(theta, dtype=float).reshape(-1)
        itheta = theta

    for i,phot in enumerate(phot_in):

        # print("energy= %g eV, theta = %15.13g deg"%(phot,itheta[i]*180/numpy.pi))
        if phot < energy[0] or phot > energy[-1]:
            raise Exception("Photon energy %g eV outside of valid limits [%g,%g]"%(phot,energy[0],energy[-1]))

        if forceratio == 0:
            ratio = numpy.sin(itheta[i]) / (toangstroms / phot)
        else:
            ratio = 1 / (2 * dspacing * 1e8)
        # print("Ratio: ",ratio)

        F0 = numpy.zeros(nbatom)
        F000 = numpy.zeros(nbatom)
        for j in range(nbatom):
            #icentral = int(f0coeff.shape[1]/2)
            #F0[j] = f0coeff[j,icentral]
            icentral = int(len(f0coeff[j])/2)
            F0[j] = f0coeff[j][icentral]
            # F000[j] = F0[j]
            for i in range(icentral):
                #F0[j] += f0coeff[j,i] * numpy.exp(-1.0*f0coeff[j,i+icentral+1]*ratio**2)
                F0[j] += f0coeff[j][i] * numpy.exp(-1.0*f0coeff[j][i+icentral+1]*ratio**2)
                #srio F000[j] += f0coeff[j][i]  #actual number of electrons carried by each atom, X.J. Yu, slsyxj@nus.edu.sg
            F000[j] = atnum[j] # srio
        # ;C
        # ;C Interpolate for the atomic scattering factor.
        # ;C
        for j,ienergy in enumerate(energy):
            if ienergy > phot:
                break
        nener = j - 1


        F1 = numpy.zeros(nbatom,dtype=float)
        F2 = numpy.zeros(nbatom,dtype=float)
        F = numpy.zeros(nbatom,dtype=complex)

        for j in range(nbatom):
            F1[j] = fp[j,nener] + (fp[j,nener+1] - fp[j,nener]) * \
            (phot - energy[nener]) / (energy[nener+1] - energy[nener])
            F2[j] = fpp[j,nener] + (fpp[j,nener+1] - fpp[j,nener]) * \
            (phot - energy[nener]) / (energy[nener+1] - energy[nener])

        r_lam0 = toangstroms * 1e-8 / phot
        for j in range(nbatom):
            F[j] = F0[j] + F1[j] + 1j * F2[j]
            # print("F",F)


        F_0 = 0.0 + 0.0j
        FH = 0.0 + 0.0j
        FH_BAR = 0.0 + 0.0j
        FHr = 0.0 + 0.0j
        FHi = 0.0 + 0.0j
        FH_BARr = 0.0 + 0.0j
        FH_BARi = 0.0 + 0.0j


        TEMPER_AVE = 1.0
        for j in range(nbatom):
            FH  += fraction[j] * (G[j] *   F[j] * 1.0) * temper[j]
            FHr += fraction[j] * (G[j] * (F0[j] + F1[j])* 1.0) * temper[j]
            FHi += fraction[j] * (G[j] *  F2[j] * 1.0) * temper[j]
            FN = F000[j] + F1[j] + 1j * F2[j]
            F_0 += fraction[j] * (G_0[j] *  FN  * 1.0)
            # TEMPER_AVE *= (temper[j])**(G_0[j]/(G_0.sum()))

            FH_BAR  += fraction[j] * ((G_BAR[j] * F[j] * 1.0)) * temper[j]
            FH_BARr += fraction[j] * ((G_BAR[j] * (F0[j]  + F1[j]) *1.0)) * temper[j]
            FH_BARi += fraction[j] * ((G_BAR[j] *  F2[j] * 1.0)) * temper[j]
            # print("TEMPER_AVE: ",TEMPER_AVE)


        # ;C
        # ;C multiply by the average temperature factor
        # ;C

        STRUCT = numpy.sqrt(FH * FH_BAR)

        # ;C
        # ;C   PSI_CONJ = F*( note: PSI_HBAR is PSI at -H position and is
        # ;C   proportional to fh_bar but PSI_CONJ is complex conjugate os PSI_H)
        # ;C

        psi_over_f = rn * r_lam0**2 / numpy.pi
        psi_h      = rn * r_lam0**2 / numpy.pi * FH
        psi_hr     = rn * r_lam0**2 / numpy.pi * FHr
        psi_hi     = rn * r_lam0**2 / numpy.pi * FHi
        psi_hbar   = rn * r_lam0**2 / numpy.pi * FH_BAR
        psi_hbarr  = rn * r_lam0**2 / numpy.pi * FH_BARr
        psi_hbari  = rn * r_lam0**2 / numpy.pi * FH_BARi
        psi_0      = rn * r_lam0**2 / numpy.pi * F_0
        psi_conj   = rn * r_lam0**2 / numpy.pi * FH.conjugate()

        # ;
        # ; Darwin width
        # ;
        ssvar = rn * (r_lam0**2) * STRUCT / numpy.pi / numpy.sin(2.0*itheta)
        spvar = ssvar * numpy.abs((numpy.cos(2.0*itheta)))
        ssr = ssvar.real
        spr = spvar.real

        # ;C
        # ;C computes refractive index.
        # ;C ([3.171] of Zachariasen's book)
        # ;C
        REFRAC = (1.0+0j) - r_lam0**2 * rn * F_0 / 2/ numpy.pi
        DELTA_REF = 1.0 - REFRAC.real
        ABSORP = 4.0 * numpy.pi * (-REFRAC.imag) / r_lam0

        txt = ""
        txt += '\n******************************************************'
        txt += '\n       at energy    = '+repr(phot)+' eV'
        txt += '\n                    = '+repr(r_lam0*1e8)+' Angstroms'
        txt += '\n       and at angle = '+repr(itheta*180.0/numpy.pi)+' degrees'
        txt += '\n                    = '+repr(itheta)+' rads'
        txt += '\n******************************************************'

        for j in range(nbatom):
            txt += '\n  '
            txt += '\nFor atom '+repr(j+1)+':'
            txt += '\n       fo + fp+ i fpp = '
            txt += '\n        '+repr(F0[j])+' + '+ repr(F1[j].real)+' + i'+ repr(F2[j])+" ="
            txt += '\n        '+repr(F0[j] + F1[j] + 1j * F2[j])
            txt += '\n       Z = '+repr(atnum[j])
            txt += '\n       Temperature factor = '+repr(temper[j])
        txt += '\n  '
        txt += '\n Structure factor F(0,0,0) = '+repr(F_0)
        txt += '\n Structure factor FH = '      +repr(FH)
        txt += '\n Structure factor FH_BAR = '  +repr(FH_BAR)
        txt += '\n Structure factor F(h,k,l) = '+repr(STRUCT)
        txt += '\n  '
        txt += '\n Psi_0  = '   +repr(psi_0)
        txt += '\n Psi_H  = '   +repr(psi_h)
        txt += '\n Psi_HBar  = '+repr(psi_hbar)
        txt += '\n  '
        txt += '\n Psi_H(real) Real and Imaginary parts = '   + repr(psi_hr)
        txt += '\n Psi_H(real) Modulus  = '                   + repr(numpy.abs(psi_hr))
        txt += '\n Psi_H(imag) Real and Imaginary parts = '   + repr(psi_hi)
        txt += '\n Psi_H(imag) Modulus  = '                   + repr(abs(psi_hi))
        txt += '\n Psi_HBar(real) Real and Imaginary parts = '+ repr(psi_hbarr)
        txt += '\n Psi_HBar(real) Modulus  = '                + repr(abs(psi_hbarr))
        txt += '\n Psi_HBar(imag) Real and Imaginary parts = '+ repr(psi_hbari)
        txt += '\n Psi_HBar(imag) Modulus  = '                + repr(abs(psi_hbari))
        txt += '\n  '
        txt += '\n Psi/F factor = '                           + repr(psi_over_f)
        txt += '\n  '
        txt += '\n Average Temperature factor = '             + repr(TEMPER_AVE)
        txt += '\n Refraction index = 1 - delta - i*beta'
        txt += '\n            delta = '                       + repr(DELTA_REF)
        txt += '\n             beta = '                       + repr(1.0e0*REFRAC.imag)
        txt += '\n Absorption coeff = '                       + repr(ABSORP)+' cm^-1'
        txt += '\n  '
        txt += '\n e^2/(mc^2)/V = '                           + repr(rn)+' cm^-2'
        txt += '\n d-spacing = '                              + repr(dspacing*1.0e8)+' Angstroms'
        txt += '\n SIN(theta)/Lambda = '                      + repr(ratio)
        txt += '\n  '
        txt += '\n Darwin width for symmetric s-pol [microrad] = ' + repr(2.0e6*ssr)
        txt += '\n Darwin width for symmetric p-pol [microrad] = ' + repr(2.0e6*spr)

    return {"PHOT":phot, "WAVELENGTH":r_lam0*1e-2 ,"THETA":itheta, "F_0":F_0, "FH":FH, "FH_BAR":FH_BAR,
	        "STRUCT":STRUCT, "psi_0":psi_0, "psi_h":psi_h, "psi_hbar":psi_hbar,
        	"DELTA_REF":DELTA_REF, "REFRAC":REFRAC, "ABSORP":ABSORP, "RATIO":ratio,
        	"ssr":ssr, "spr":spr, "psi_over_f":psi_over_f, "info":txt}

#
#
#
def bragg_calc2(descriptor="YB66", hh=1, kk=1, ll=1, temper=1.0,
                emin=5000.0, emax=15000.0, estep=100.0, ANISO_SEL=0,
                fileout=None,
                do_not_prototype=0, # 0=use site groups (recommended), 1=use all individual sites
                verbose=True,
                material_constants_library=xraylib,
                ):
    """
    Preprocessor for Structure Factor (FH) calculations. It calculates the basic ingredients of FH.


    Parameters
    ----------
    descriptor: str, optional
        crystal name (as in xraylib)
    hh: int, optional
        miller index H
    kk: int, optional
        miller index K
    ll: int, optional
        miller index L
    temper: float, optional
        temperature factor (scalar <=1.0 )
    emin: float, optional
        photon energy minimum in eV
    emax: float, optional
        photon energy maximum in eV
    estep: float, optional
        photon energy step in eV
    ANISO_SEL: int, optional
        0: Do not use anisotropy.
        1: Use anisotropy.
    fileout: None or str, optional
        name for the output file (default=None, no output file)
    do_not_prototype: int, optional
        for computing the structure factor, 0=sum over site groups (recommended), 1=sum over each individual sites
    verbose: int, optional
        Set to 1 for verbose output.
    material_constants_library: xraylib or instance of DabaxXraylib, optional
        The pointer to the material library to be used to retrieve scattering data.

    Returns
    -------
    dict
        a dictionary with all ingredients of the structure factor.
    """
    output_dictionary = {}

    codata_e2_mc2 = codata.e ** 2 / codata.m_e / codata.c ** 2 / (4 * numpy.pi * codata.epsilon_0)  # in m

    # f = open(fileout,'w')

    version = "2.6"
    output_dictionary["version"] = version


    txt = ""
    txt += "# Bragg version, Data file type\n"
    txt += "%s\n" % version

    cryst = material_constants_library.Crystal_GetCrystal(descriptor)

    if cryst is None:
        raise Exception("Crystal descriptor %s not found in material constants library" % descriptor)

    volume = cryst['volume']

    # test crystal data - not needed
    icheck= 0
    if icheck:
        print("  Unit cell dimensions are %f %f %f" % (cryst['a'], cryst['b'], cryst['c']))
        print("  Unit cell angles are %f %f %f" % (cryst['alpha'], cryst['beta'], cryst['gamma']))
        print("  Unit cell volume is %f A^3" % volume)
        print("  Atoms at:")
        print("     Z  fraction    X        Y        Z")
        for i in range(cryst['n_atom']):
            atom = cryst['atom'][i]
            print("    %3i %f %f %f %f" % (atom['Zatom'], atom['fraction'], atom['x'], atom['y'], atom['z']))
        print("  ")

    volume = volume * 1e-8 * 1e-8 * 1e-8  # in cm^3
    rn = (1e0 / volume) * (codata_e2_mc2 * 1e2)

    dspacing = bragg_metrictensor(cryst['a'], cryst['b'], cryst['c'], cryst['alpha'], cryst['beta'], cryst['gamma'], HKL=[hh, kk, ll])
    dspacing *= 1e-8  # in cm

    txt += "# RN = (e^2/(m c^2))/V) [cm^-2], d spacing [cm]\n"
    txt += "%e %e \n" % (rn, dspacing)

    output_dictionary["rn"] = rn
    output_dictionary["dspacing"] = dspacing

    atom = cryst['atom']
    number_of_atoms = len(atom)
    list_Zatom = [atom[i]['Zatom'] for i in range(len(atom))]

    list_fraction = [atom[i]['fraction'] for i in range(number_of_atoms)]
    try:
        list_charge = [atom[i]['charge'] for i in range(number_of_atoms)]
    except:
        list_charge = [0.0] * number_of_atoms
    list_x = [atom[i]['x'] for i in range(number_of_atoms)]
    list_y = [atom[i]['y'] for i in range(number_of_atoms)]
    list_z = [atom[i]['z'] for i in range(number_of_atoms)]

    # calculate array of temperature factor for all atoms
    #
    # Consider anisotropic temperature factor
    # X.J. Yu, slsyxj@nus.edu.sg
    # A dummy dictionary Aniso with start =0 if no aniso temperature factor input
    # start
    if 'Aniso' in cryst.keys() and cryst['Aniso'][0]['start'] > 0:  # most crystals have no Anisotropic input
        TFac = TemperFactor(1.0 / (2.0 * dspacing * 1e8), cryst['Aniso'], Miller={'h': hh, 'k': kk, 'l': ll}, \
                            cell={'a': cryst['a'], 'b': cryst['b'], 'c': cryst['c']}, n=len(atom))
        B_TFac = 1
    else:
        B_TFac = 0
    #
    #
    #
    list_temper = []
    list_temper_label = []
    if ANISO_SEL == 0:
        for i in range(number_of_atoms):
            list_temper.append(temper)
            list_temper_label.append(-1)
    elif ANISO_SEL == 1:
        if B_TFac:
            for i in range(number_of_atoms):
                list_temper.append(TFac[0, i])
                list_temper_label.append(TFac[2, i])
        else:
            raise Exception("No crystal data to calculate isotropic temperature factor for crystal %s" % descriptor)
    elif ANISO_SEL == 2:
        if B_TFac:
            for i in range(number_of_atoms):
                list_temper.append(TFac[1, i])
                list_temper_label.append(TFac[2, i])
        else:
            raise Exception("No crystal data to calculate anisotropic temperature factor for crystal %s" % descriptor)

    list_AtomicName = []
    for i in range(number_of_atoms):
        s = atomic_symbols()[atom[i]['Zatom']]
        # if sourceCryst == 1: # charge is not available in xraylib
        try: # charge is not available in xraylib
            if atom[i]['charge'] != 0.0:  # if charge is 0, s is symbol only, not B0, etc
                s = s + f'%+.6g' % atom[i]['charge']
        except:
            pass
        list_AtomicName.append(s)

    # identify the prototypical atoms
    labels_prototypical = []
    for i in range(number_of_atoms):
        labels_prototypical.append("Z=%d C=%g F=%g T=%g" % (list_Zatom[i], list_charge[i], list_fraction[i], list_temper_label[i]))

    if do_not_prototype:
        indices_prototypical = numpy.arange(number_of_atoms)  # different with diff_pat for complex crystal
    else:
        indices_prototypical = numpy.unique(labels_prototypical, return_index=True)[1]

    number_of_prototypical_atoms = len(indices_prototypical)

    f0coeffs = []
    for i in indices_prototypical:
        try:
            charge = atom[i]['charge']
        except:
            charge = 0.0
        f0coeffs.append(f0_xop_with_fractional_charge(atom[i]['Zatom'], charge))


    txt += "# Number of different element-sites in unit cell NBATOM:\n%d \n" % number_of_prototypical_atoms
    output_dictionary["nbatom"] = number_of_prototypical_atoms

    txt += "# for each element-site, the number of scattering electrons (Z_i + charge_i)\n"
    atnum_list = []
    for i in indices_prototypical:
        txt += "%f " % (list_Zatom[i] - list_charge[i])
        atnum_list.append(list_Zatom[i] - list_charge[i])
    txt += "\n"
    output_dictionary["atnum"] = atnum_list


    txt += "# for each element-site, the occupation factor\n"
    unique_fraction = [list_fraction[i] for i in indices_prototypical]
    for z in unique_fraction:
        txt += "%g " % (z)
    txt += "\n"
    output_dictionary["fraction"] = unique_fraction

    txt += "# for each element-site, the temperature factor\n"  # temperature parameter
    unique_temper = []
    for i in indices_prototypical:
        txt += "%g " % list_temper[i]
        unique_temper.append(list_temper[i])
    txt += "\n"
    output_dictionary["temper"] = unique_temper

    #
    # Geometrical part of structure factor:  G and G_BAR
    #
    txt += "# for each type of element-site, COOR_NR=G_0\n"
    list_multiplicity = []
    for i in indices_prototypical:
        if do_not_prototype:
            txt += "%d " % 1
            list_multiplicity.append(1)
        else:
            count = 0
            for j in range(number_of_atoms):
                if labels_prototypical[j] == labels_prototypical[i]: count += 1
            txt += "%d " % count
            list_multiplicity.append(count)
    txt += "\n"
    output_dictionary["G_0"] = list_multiplicity

    txt += "# for each type of element-site, G and G_BAR (both complex)\n"
    list_g = []
    list_g_bar = []
    for i in indices_prototypical:

        if do_not_prototype:
            # # ga_item = numpy.exp(2j * numpy.pi * (hh * list_x[i] + kk * list_y[i] + ll * list_z[i]))
            # ga += ga_item
            ga = numpy.exp(2j * numpy.pi * (hh * list_x[i] + kk * list_y[i] + ll * list_z[i]))
        else:
            ga = 0.0 + 0j
            for j in range(number_of_atoms):
                if labels_prototypical[j] == labels_prototypical[i]:
                # if list_AtomicName[j] == zz and list_fraction[j] == ff and list_temper[j] == tt:
                    ga_item = numpy.exp(2j * numpy.pi * (hh * list_x[j] + kk * list_y[j] + ll * list_z[j]))
                    ga += ga_item

        txt += "(%g,%g) \n" % (ga.real, ga.imag)
        txt += "(%g,%g) \n" % (ga.real, -ga.imag)
        list_g.append(ga)
        list_g_bar.append(ga.conjugate())
    output_dictionary["G"] = list_g
    output_dictionary["G_BAR"] = list_g_bar

    #
    # F0 part
    #
    txt += "# for each type of element-site, the number of f0 coefficients followed by them\n"
    for f0coeffs_item in f0coeffs:
        txt += "%d " % len(f0coeffs_item)
        for cc in f0coeffs_item:
            txt += "%g " % cc
        txt += "\n"
    output_dictionary["f0coeff"] = f0coeffs


    # X.J. Yu, use ceil to round up, otherwise we may get actual max energy less than emax
    npoint = int(numpy.ceil(((emax - emin) / estep + 1)))
    txt += "# The number of energy points NPOINT: \n"
    txt += ("%i \n") % npoint
    output_dictionary["npoint"] = npoint

    txt += "# for each energy point, energy, F1(1),F2(1),...,F1(nbatom),F2(nbatom)\n"
    list_energy = []
    out_f1 = numpy.zeros((len(indices_prototypical), npoint), dtype=float)
    out_f2 = numpy.zeros((len(indices_prototypical), npoint), dtype=float)
    out_fcompton = numpy.zeros((len(indices_prototypical), npoint), dtype=float) # todo: is complex?

    if isinstance(material_constants_library, DabaxXraylib ):
        # vectorize with DABAX
        energies = numpy.zeros(npoint)
        for i in range(npoint):
            energies[i] = (emin + estep * i)

        DABAX_F_RESULTS = []
        for j, jj in enumerate(indices_prototypical):
            DABAX_F_RESULTS.append(numpy.array( material_constants_library.FiAndFii(list_Zatom[jj], energies * 1e-3)))

        for i in range(npoint):
            energy = (emin + estep * i)
            txt += ("%20.11e \n") % (energy)
            list_energy.append(energy)

            for j, jj in enumerate(indices_prototypical):
                f1a =  (DABAX_F_RESULTS[j])[0, i]  # material_constants_library.Fi(list_Zatom[jj], energy * 1e-3)
                f2a = -(DABAX_F_RESULTS[j])[1, i]  # -material_constants_library.Fii(list_Zatom[jj], energy * 1e-3)
                txt += (" %20.11e %20.11e 1.000 \n") % (f1a, f2a)
                out_f1[j, i] = f1a
                out_f2[j, i] = f2a
                out_fcompton[j, i] = 1.0
    else:
        # make a simple loop with xraylib (fast)
        for i in range(npoint):
            energy = (emin + estep * i)
            txt += ("%20.11e \n") % (energy)
            list_energy.append(energy)

            for j,jj in enumerate(indices_prototypical):
                f1a = material_constants_library.Fi(list_Zatom[jj], energy * 1e-3)
                f2a = -material_constants_library.Fii(list_Zatom[jj], energy * 1e-3)
                txt += (" %20.11e %20.11e 1.000 \n") % (f1a, f2a)
                out_f1[j, i] = f1a
                out_f2[j, i] = f2a
                out_fcompton[j, i] = 1.0


    output_dictionary["energy"] = list_energy
    output_dictionary["f1"] = out_f1
    output_dictionary["f2"] = out_f2
    output_dictionary["fcompton"] = out_fcompton

    if fileout != None:
        bragg_preprocessor_file_v2_write(output_dictionary, fileout)

    return output_dictionary

# todo: rename
def TemperFactor(sinTheta_lambda, anisos, Miller={'h':1,'k':1,'l':1}, cell={'a':23.44,'b':23.44,'c':23.44}, n=1936):
    """
    Calculation isotropic & anisotropic temerature factors.

    Parameters
    ----------
    sinTheta_lambda: float
        Sin(theta)/lambda, lambda in units of Angstrom.
    anisos: numpy array
        array of dictionary containing anisotropic coefficients.
    Miller: dict
        The miller indices, example: {'h':1,'k':1,'l':1}.
    cell: dict
        The cell a,b,c parameters, example: {'a':23.44,'b':23.44,'c':23.44}
    n: int, optional
        number of atomic sites.

    Returns
    -------
    list
        output results in a 2-elements list: [[sotropic],[anisotropic]].

    """

    #+
    # Singapore Synchrotron Light Source (SSLS)
    # :Author: X.J. Yu, slsyxj@nus.edu.sg
    # :Name:  TemperFactor
    # :Purpose: Calculation isotropic & anisotropic temerature factors
    # :Input:
    #     Miller: Miller indices
    #     cell:  dictionary of lattice [a,b,c] in units of Angstrom
    #     sinTheta_lambda: Sin(theta)/lambda, lambda in units of Angstrom
    #     n: number of atomic sites
    #     anisos: array of dictionary containing anisotropic coefficients
    #     Out: output results in a 2-elements list: [[sotropic],[anisotropic]]
    #-

    #0: isotropic, 1: anisotropic temerature factors
    # results = numpy.zeros([2,n])
    results = numpy.zeros([3,n]) # srio adds "start"

    for i, aniso in enumerate(anisos):
        s = aniso['start'] - 1
        e = aniso['end']
        if aniso['beta11'] >= 1:
            #if beta11>=1, then beta22 is Beq, the other fields are unused
            #if Beq specified, anisotropic temperature factor same as isotropic
            Beq = aniso['beta22']
            results[1, s:e] = numpy.exp(-sinTheta_lambda * sinTheta_lambda*Beq)
        else:
            Beq = 4.0 / 3.0 * ( aniso['beta11'] * cell['a'] * cell['a'] + aniso['beta22'] * cell['b'] * cell['b']+ \
                aniso['beta33'] * cell['c'] * cell['c'] ) # this is true only for cubic, tetragonal and orthorhombic Giacovazzo pag 188
            results[1,s:e] = numpy.exp(-(aniso['beta11'] * Miller['h'] * Miller['h'] + \
                  aniso['beta22'] * Miller['k'] * Miller['k'] + aniso['beta33'] * Miller['l'] * Miller['l'] + \
                  2.0 * Miller['h'] * Miller['k'] * aniso['beta12'] + 2.0 * Miller['h'] * Miller['l'] * aniso['beta13'] + \
                                         2.0 * Miller['k'] * Miller['l'] * aniso['beta23']))
        results[0,s:e] = numpy.exp(-sinTheta_lambda * sinTheta_lambda * Beq)

        results[2, s:e] = s

    return results

def mare_calc(descriptor, H, K, L, HMAX, KMAX, LMAX, FHEDGE, DISPLAY, lambda1, deltalambda, PHI, DELTAPHI,
              material_constants_library=xraylib,verbose=0):
    """
    Calculates:

      - Spaghetti plots (lambda versis Psi for multiple crystal reflection)

      - The Umweganregung peak location plot (the diffracted wavelength lambda vs. Psi) for a given primary
        reflection,i.e., an horizontal cut of the spaghetti plot.
      - The Glitches spectrum (the negative intensity for versus the wavelength) or a vertical cut of the spaghetti plot.

      Psi is the azimutal angle of totation, i.e., the totation around
        the H vector (main reflection)

     In other words, if a crystal is set with a particular Bragg angle to match a given reflection (inputs: H,K,L) at
     a given wavelength (input: WaveLength), many other (secondary) reflections are excited when the crystal is rotated
     around the azimutal angle Psi, without changing the Bragg angle.

     The plot (WaveLength,Psi) of the possible reflections is calculated and contains all possible reflection curves
     up to a maximum reflection (input: H Max,  K Max, L Max).

     Umweg plot:

     The intersection of these curves with an horizontal line at the wavelength of the primary reflection
     (input: WaveLength) gives the position of the peaks in the unweg plot. The width of each peak depends on the
     pendent of the curve at the intersection. For that, the Psi1 and Psi2 intersection angles with a band of width
     (input: DeltaWaveLength) are calculated. With this width and the intensity of the diffraction line, it is possible
     to compute a Gaussian that "roughly" describe the peak.

     Glitches plot:

     The intersection of these curves with a vertical line at a given Psi gives the position of the peaks in the
     glitches plot. The width of each peak is the difference between the wavelength values for Psi+/-DeltaPsi
     With this width and the intensity of the diffraction line, it is possible to compute a Gaussian that "roughly"
     describe the peak.


    Parameters
    ----------
    descriptor: str
        a valid crystal name for xraylib.
    H: int
        the miller index H.
    K: int
        the miller index K.
    L: int
        the miller index L.
    HMAX: int
        the maximum miller index H.
    KMAX: int
        the maximum miller index K.
    LMAX: int
        the maximum miller index L.
    FHEDGE: float
        below this edge (structure factor value) the reflections are discarded.
    DISPLAY: int
            0: Create spaghetti plot script
            1: Create spaghetti+Umweg plot scripts
            2: Create spaghetti+Glitches plot scripts
            3: Create spaghetti+Umweg+Glitches plot scripts
    lambda1: float
        wavelength in Angstroms for Umweg plot.
    deltalambda: float
        delta wavelength in Angstroms for Umweg plot.
    PHI: float
        phi angle in deg for the Glitches plot.
    DELTAPHI: float
        delta phi angle in deg for the Glitches plot.
    material_constants_library: xraylib or instance of DabaxXraylib, optional
        The pointer to the material library to be used to retrieve scattering data.
    verbose: int, optional
        set to 1 for a more verbose output

    Returns
    -------
    str
        A script to create the plots.

    """

    list_of_scripts = []


    cryst = material_constants_library.Crystal_GetCrystal(descriptor)
    # volume = cryst['volume']
    #
    # # crystal data - not needed
    #
    # print ("  Unit cell dimensions are %f %f %f" % (cryst['a'],cryst['b'],cryst['c']))
    # print ("  Unit cell angles are %f %f %f" % (cryst['alpha'],cryst['beta'],cryst['gamma']))
    # print ("  Unit cell volume is %f A^3" % volume )
    # print ("  Atoms at:")
    # print ("     Z  fraction    X        Y        Z")
    # for i in range(cryst['n_atom']):
    #     atom =  cryst['atom'][i]
    #     print ("    %3i %f %f %f %f" % (atom['Zatom'], atom['fraction'], atom['x'], atom['y'], atom['z']) )
    # print ("  ")


    fhEdge = FHEDGE
    fhMax = -1e0
    fhMaxIndex = -1

    flg_s = 0
    flg_u = 0
    flg_g = 0


    if DISPLAY == 0:
        flg_s = 1
    elif DISPLAY == 1:
        flg_s = 1
        flg_u = 1
    elif DISPLAY == 2:
        flg_s = 1
        flg_g = 1
    elif DISPLAY == 3:
        flg_s = 1
        flg_u = 1
        flg_g = 1


    # ;
    # ; compute the metric tensor in the reciprocal space
    # ;
    ginv = bragg_metrictensor(cryst['a'],cryst['b'],cryst['c'],cryst['alpha'],cryst['beta'],cryst['gamma'])

    # ;
    # ; wavelength (for intersections: unweg pattern)
    # ;
    # lambda1 = LAMBDA # ; for intersections
    # deltalambda = DELTALAMBDA
    lambdas = numpy.array([lambda1-deltalambda, lambda1, lambda1+deltalambda])

    # ;
    # ; phi (for intersections: glitches pattern)
    # ;
    phi = PHI
    deltaPhi = DELTAPHI
    phis = numpy.array([phi-deltaPhi, phi, phi+deltaPhi])

    # ;
    # ; Main reflection
    # ;
    P = numpy.array([H,K,L],dtype=int)
    p2 = (P[0]**2 + P[1]**2 + P[2]**2)
    pn = numpy.sqrt(p2)

    # ;
    # ; Calculate Reference axis (corresponding to phi =0)
    # ; This is a vector perpendicular to P
    # ;
    mm1 = numpy.dot(ginv,P.T)
    mm2 = [mm1[1],-mm1[0],0]
    mm3 = numpy.min(numpy.abs( mm1[numpy.where(mm1 != 0)]  ))
    M0 = (mm2/mm3)

    # ;
    # ; operational reflections (for permutations)
    # ;
    pmax = numpy.array([HMAX,KMAX,LMAX],dtype=float)
    hh = numpy.arange(pmax[0])+1
    hh = numpy.concatenate((-hh,[0],hh))
    kk = numpy.arange(pmax[1])+1
    kk = numpy.concatenate((-kk,[0],kk))
    ll = numpy.arange(pmax[2])+1
    ll = numpy.concatenate((-ll,[0],ll))

    # ;
    # ; calculate the structure needed for intensity calculations
    # ;
    toangstroms = codata.h * codata.c / codata.e * 1e10
    energy = toangstroms/lambda1

    # ;
    # ; first call to bragg_inp, then calculates the intensity of the main reflection
    # ;
    fhInp = bragg_calc(descriptor,int(P[0]),int(P[1]),int(P[2]),emin=energy-100,emax=energy+100,estep=10.0)
    outInt = crystal_fh(fhInp,energy)

    bragg_angle = 180.0 / numpy.pi * numpy.arcsin(lambda1 * 1e-8/2 / fhInp['dspacing'])
    fhMain = outInt["STRUCT"].real
    intMain = lorentz(bragg_angle)*(fhMain**2)

    if verbose:
        print('Main reflection d-spacing [A]: ',fhInp["dspacing"]*1e8)
        print('Main reflection 1/2d=sin(theta)/lambda: ',1.0/(2*fhInp["dspacing"]*1e8))
        print('Main reflection Bragg angle (using lambda Umweg) [DEG]: ',outInt["THETA"]*180/numpy.pi)
        print('Main reflection Lorentz: ',lorentz(outInt["THETA"]*180/numpy.pi))
        print('Main reflection fh (real part): ',fhMain)
        print('Main reflection intensity: ',intMain)
    #
    # ;
    # ; creates abscissas for spaghettis
    # ;
    alpha = numpy.linspace(-90.0,90.0,500)

    # ;
    # ; main loop over permutations on operatinal reflections
    # ;
    out = numpy.zeros((18,15000))
    ngood = 0
    print("MARE: loop over %d reflections..."%(hh.size*kk.size*ll.size))

    norm = lambda vector: numpy.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)

    ijk = 0
    for ih in range(hh.size):
        for ik in range(kk.size):
            for il in range(ll.size):
                ijk += 1
                if verbose: print("\n-------------%d-------------,hkl: %d %d %d"%(ijk,hh[ih],kk[ik],ll[il]))
                r = numpy.array((hh[ih],kk[ik],ll[il]),dtype=int)

                rp = (r*P).sum() / p2 * P
                rp2 = (rp[0]**2 + rp[1]**2 + rp[2]**2)
                rpn = numpy.sqrt(rp2)

                p2new = numpy.dot( P , numpy.dot(ginv,P.T))
                rpnew = numpy.dot( r , numpy.dot(ginv,P.T)) / p2new
                rpnew = rpnew * P

                #   ;
                #   ; Alpha0
                #   ;
                cos_alpha0 = ((r-rp)*M0).sum() / norm(r-rp)/norm(M0)
                alpha0rad = numpy.arccos(cos_alpha0)

                #   ; NOTA BENE: alpha0 is calculating using the orthonormal scalar
                #   ; product. Should this be changed using the metric tensor for a
                #   ; generic structure?
                alpha0 = alpha0rad * 180 / numpy.pi

                #   ;
                #   ; k
                #   ;

                knew1 = 0.5 *  ( numpy.dot( r , numpy.dot( ginv , r.T))  - numpy.dot( r , numpy.dot( ginv , P.T)) )
                knew22 = numpy.dot(r , numpy.dot(ginv , r.T)) - numpy.dot(rpnew, numpy.dot(ginv , rpnew.T))
                knew2 = numpy.sqrt( knew22 )
                knew = knew1 / knew2

                if numpy.abs(knew22) > 1e-8:
                    goodRef = 1
                else:
                    goodRef = 0


                #   ;
                #   ; computes intensity
                #   ;

                fhInp = bragg_calc(descriptor,int(r[0]),int(r[1]),int(r[2]),emin=energy-100,emax=energy+100,estep=10.0,fileout=None)

                fhInp["f1"] *= 0.0
                fhInp["f2"] *= 0.0

                outInt = crystal_fh(fhInp,energy,forceratio=1)

                if outInt["STRUCT"].real < fhEdge:
                    goodRef = 0

                if goodRef == 1:
                    ngood += 1
                    braggAngleUmweg = outInt["THETA"] * 180 / numpy.pi
                    beta = alpha - alpha0
                    y3 = 1.0 / numpy.sqrt( (knew / numpy.cos(beta * numpy.pi / 180))**2 + p2new / 4 )
                    if verbose: print("Bragg angle (for Umweg): %g"%braggAngleUmweg)

                    theta1 = knew**2 / ((1/lambdas)**2 - p2new / 4)
                    if numpy.abs(theta1[1] > 1):
                        theta2 = [-1000,-1000,-1000]
                        theta3 = [-1000,-1000,-1000]
                    else:
                        theta1 = numpy.arccos(numpy.sqrt(theta1))
                        theta1 = theta1*180/numpy.pi
                        theta2 = alpha0 - theta1
                        theta3 = alpha0 + theta1 - 180
                    #     ;
                    #     ; lambda values for phi intervals (for glitches)
                    #     ;
                    lambdaIntersec = 1.0 / numpy.sqrt( (knew/numpy.cos((phis-alpha0)*numpy.pi/180))**2+p2new/4 )

                    if verbose: print("lambdaIntersec:  ",repr(lambdaIntersec))
                    if verbose: print(("d-spacing [A]: %g"%fhInp["dspacing"]))

                    braggAngleGlitches = lambdaIntersec[1]/2/fhInp["dspacing"]/1e8

                    if numpy.abs(braggAngleGlitches) <= 1:
                        braggAngleGlitches = numpy.arcsin(braggAngleGlitches)*180/numpy.pi
                    else:
                        braggAngleGlitches = 0

                    if verbose: print("Bragg angle (for Glitches): %g"%braggAngleGlitches)
                    #     ;
                    #     ; print/store results
                    #     ;
                    out[0,ngood-1]=r[0]
                    out[1,ngood-1]=r[1]
                    out[2,ngood-1]=r[2]
                    out[3,ngood-1]=alpha0
                    out[4,ngood-1]=knew
                    out[5,ngood-1]=p2new/4
                    out[6,ngood-1]=theta2[0]
                    out[7,ngood-1]=theta2[1]
                    out[8,ngood-1]=theta2[2]
                    out[9,ngood-1]=theta3[0]
                    out[10,ngood-1]=theta3[1]
                    out[11,ngood-1]=theta3[2]
                    out[12,ngood-1]=lambdaIntersec[0]
                    out[13,ngood-1]=lambdaIntersec[1]
                    out[14,ngood-1]=lambdaIntersec[2]
                    out[15,ngood-1]=braggAngleUmweg
                    out[16,ngood-1]=braggAngleGlitches
                    out[17,ngood-1]=(outInt["STRUCT"]).real

                    if outInt["STRUCT"].real > fhMax:
                        fhMax = outInt["STRUCT"].real
                        fhMaxIndex = ngood - 1

    if ngood == 0:
        print("Warning: No good reflections found.")
        return None


    out = out[:,0:(ngood)].copy()
    #
    # ;
    # ; common header for scripts
    # ;
    #
    txt0 = ""
    txt0 += "#\n"
    txt0 += "# xoppy/python macro created by MARE \n"
    txt0 += "# xoppy/mare multiple diffraction  \n"
    txt0 += "#  \n"
    txt0 += "# inputs:  \n"
    # txt0 += "# crystal index: %d\n"%(CRYSTAL)
    txt0 += "# crystal name: %s  \n"%(descriptor)
    txt0 += "# Main reflection: %d %d %d\n"%(H,K,L)
    txt0 += "# Max reflections: %d %d %d\n"%(HMAX,KMAX,LMAX)
    txt0 += "# Wavelength = %g A \n"%(lambda1)
    txt0 += "# Delta Wavelength = %g A\n"%(deltalambda)
    txt0 += "# Phi = %g deg \n"%(PHI)
    txt0 += "# Delta Phi = %g deg\n"%(DELTAPHI)
    txt0 += "# Display: %d \n"%(DISPLAY)
    txt0 += "# Using reflections with fh > %d \n"%(fhEdge)
    txt0 += "# \n"
    txt0 += "# Computed parameters: \n"
    txt0 += "# Number of good reflections: %d \n"%(ngood)
    txt0 += "# M vector (corresponding to phi=0)  %d %d %d \n"%(M0[0],M0[1],M0[2])
    txt0 += "# Intensity of main reflection: %g \n"%(intMain)
    txt0 += "# Structure Factor fh of main reflection: %g \n"%(fhMain)
    txt0 += "# Reflection with maximum intensity: \n"
    txt0 += "#            number: %d \n"%(fhMaxIndex)
    txt0 += "#            miller indices: %d %d %d \n"%(
        int(out[0,fhMaxIndex]),int(out[1,fhMaxIndex]),int(out[2,fhMaxIndex])   )
    txt0 += "#            fh value: %g \n"%(fhMax)



    # ;
    # ; plot script with spaghettis
    # ;

    txt = txt0
    txt += "import numpy\n"
    txt += "import matplotlib.pylab as plt\n"
    txt += "import matplotlib.pylab as plt\n"
    txt += "fig = plt.figure()\n"
    txt += "ax = fig.add_subplot(111)\n"
    txt += "parms = {'n':500, 'xmin':-90.,'xmax':90.,'A_or_eV':0,'ymin':0.,'ymax':3.5}\n"
    txt += "alpha = numpy.linspace(parms['xmin'],parms['xmax'],parms['n'],)\n"
    txt += "ytitle='Photon energy [eV]' if parms['A_or_eV'] == 1 else 'Wavelength [A]'\n"
    txt += "plt.title('MARE-spaghetti, Main diffraction: %d %d %d %s')\n"%(H,K,L,descriptor)
    txt += "plt.xlabel('Azimuthal angle [deg]')\n"
    txt += "plt.ylabel(ytitle)\n"

    txt += "lambdas = numpy."+repr(lambdas)+"\n"
    txt += "phis = numpy."+repr(phis)+"\n"
    txt += "yy =12398.419/lambdas if parms['A_or_eV'] == 1 else lambdas\n"

    for i in range(ngood):
        txt += "# --------------------------------\n"
        txt += "# Reflection nr: %d \n"%(i+1)
        txt += "#           h           k           l      alpha0           k        p2/4         th2         th2         th2         th3         th3         th3      lambda      lambda      lambda     BrgAngU     BrgAngG          fh\n"
        txt += ("#"+"%12d"*3+"%12.6g"*15+"\n")%(tuple(out[:,i]))
        txt += "y3 = 1.0/numpy.sqrt((%g/numpy.cos((alpha-%g)*numpy.pi/180))**2 + %g)\n"%(out[4,i],out[3,i],out[5,i])
        txt += "if parms['A_or_eV'] == 1: y3=12398.419/y3\n"
        txt += "fg = plt.plot(alpha,y3)\n"
        txt += "ilabel = int(numpy.random.rand()*(parms['n']-1))\n"%()
        txt += "ax.text(alpha[ilabel],y3[ilabel],'%d %d %d',color=fg[0].get_color())\n"%(int(out[0,i]),int(out[1,i]),int(out[2,i]))

    txt += "plt.show()\n"

    list_of_scripts.append(txt)
    if verbose: print(txt)

    # ;
    # ; plot macro with umweg pattern
    # ;
    #
    if flg_u:

        txt1 = txt0
        txt1 += "import numpy\n"
        txt1 += "import matplotlib.pylab as plt\n"
        txt1 += "import matplotlib.pylab as plt\n"
        txt1 += "fig = plt.figure()\n"
        txt1 += "ax = fig.add_subplot(111)\n"
        txt1 += "parms = {'n':500, 'xmin':-90.,'xmax':90.,'A_or_eV':0,'ymin':0.,'ymax':0}\n"
        txt1 += "alpha = numpy.linspace(parms['xmin'],parms['xmax'],parms['n'],)\n"
        txt1 += "umweg = alpha*0\n"
        txt1 += "plt.title('MARE-umweg, Main diffraction: %d %d %d %s at %g A')\n"%(H,K,L,descriptor,lambda1)
        txt1 += "plt.xlabel('Azimuthal angle [deg]')\n"
        txt1 += "plt.ylabel('Approximated intensity')\n"

        for i in range(ngood):
            txt1 += "# --------------------------------\n"
            txt1 += "# Reflection nr: %d \n"%(i+1)
            txt1 += "#           h           k           l      alpha0           k        p2/4         th2         th2         th2         th3         th3         th3      lambda      lambda      lambda     BrgAngU     BrgAngG          fh\n"
            txt1 += ("#"+"%12d"*3+"%12.6g"*15+"\n")%(tuple(out[:,i]))

            intens = out[17,i]**2 *lorentz(out[15,i])
            txt1 += "theta2 = numpy.array([%g,%g,%g])\n"%(out[6,i],out[7,i],out[8,i])
            txt1 += "theta3 = numpy.array([%g,%g,%g])\n"%(out[9,i],out[10,i],out[11,i])

            if numpy.abs(out[8,i]-out[6,i]) > 1e-6:
                ymax = intens/numpy.abs(out[8,i]-out[6,i])
                txt1 += "intens = %g**2 * %g\n"%(out[17,i],lorentz(out[15,i]))
                txt1 += "umweg +=  (intens/numpy.abs(theta2[2]-theta2[0]))*numpy.exp(-(alpha-theta2[1])**2/numpy.abs(theta2[2]-theta2[0])**2) \n"
            if numpy.abs(out[11,i]-out[9,i]) > 1e-6:
                ymax = intens/numpy.abs(out[8,i]-out[6,i])
                txt1 += "intens = %g**2 * %g\n"%(out[17,i],lorentz(out[15,i]))
                txt1 += "umweg +=  (intens/numpy.abs(theta3[2]-theta3[0]))*numpy.exp(-(alpha-theta3[1])**2/numpy.abs(theta3[2]-theta3[0])**2) \n"

        txt1 += "plt.plot(alpha,umweg)\n"

        txt1 += "plt.show()\n"
        #
        list_of_scripts.append(txt1)
        if verbose: print(txt1)




    # ;
    # ; plot macro with glitches pattern
    # ;
    if flg_g:

        txt2 = txt0
        txt2 += "import numpy\n"
        txt2 += "import matplotlib.pylab as plt\n"
        txt2 += "import matplotlib.pylab as plt\n"
        txt2 += "fig = plt.figure()\n"
        txt2 += "ax = fig.add_subplot(111)\n"
        txt2 += "parms = {'n':500, 'xmin':0.5,'xmax':3.5,'A_or_eV':0,'ymin':0.,'ymax':0}\n"
        txt2 += "xmin = parms['xmin']\n"
        txt2 += "xmax = parms['xmax']\n"
        txt2 += "if parms['A_or_eV'] == 1: xmin = 12398.419/xmin\n"
        txt2 += "if parms['A_or_eV'] == 1: xmax = 12398.419/xmax\n"

        txt2 += "xx = numpy.linspace(xmin,xmax,parms['n'],)\n"
        txt2 += "yy = xx*0\n"
        txt2 += "plt.title('MARE-glitches, Main diffraction: %d %d %d %s at %g deg')\n"%(H,K,L,descriptor,phis[1])
        txt2 += "xtitle='Wavelength [A]' if parms['A_or_eV']==0 else 'Photon energy [eV]'\n"
        txt2 += "plt.xlabel(xtitle)\n"
        txt2 += "plt.ylabel('Approximated intensity')\n"

        for i in range(ngood):
            txt2 += "# --------------------------------\n"
            txt2 += "# Reflection nr: %d \n"%(i+1)
            txt2 += "#           h           k           l      alpha0           k        p2/4         th2         th2         th2         th3         th3         th3      lambda      lambda      lambda     BrgAngU     BrgAngG          fh\n"
            txt2 += ("#"+"%12d"*3+"%12.6g"*15+"\n")%(tuple(out[:,i]))
            txt2 += "lambdas = numpy.array([%g,%g,%g])\n"%(out[12,i],out[13,i],out[14,i])
            txt2 += "intens = %g**2 * %g\n"%(out[17,i],lorentz(out[16,i]))
            if numpy.abs(out[14,i]-out[12,i]) > 1e-6:
                txt2 += "yy = yy + (intens/numpy.abs(lambdas[2]-lambdas[0]))*numpy.exp(-(xx-lambdas[1])**2/numpy.abs(lambdas[2]-lambdas[0])**2)\n"

        txt2 += "plt.plot(xx,-yy)\n"

        txt2 += "plt.show()\n"

        list_of_scripts.append(txt2)
        if verbose: print(txt2)


    return(list_of_scripts)



def calc_temperature_factor(temperature, crystal='Si', debyeTemperature=644.92,
                            millerIndex=[1, 1, 1], atomicMass=28.09,
                            dSpacing=3.1354162886330583,
                            material_constants_library=xraylib
                          ):
    """
    Calculates the (Debye) temperature factor for single crystals. See [1]_, [2]_.

    Parameters
    ----------
    temperature: float
        Crystal temperature in Kelvin (positive number).
    crystal: str, optional
        Crystal descriptor (e.g. Si, Ge, ...).
    debyeTemperature: float, optional
        Debye temperature of the crystal material in Kelvin.
    millerIndex: list or numpy array, optional
        Miller indexes of the crystal orientation. For use with xraylib only.
    atomicMass: float, optional
        Atomic mass of the crystal element (amu unit). if atomicMass == 0, get from xraylib.
    dSpacing: float, optional
        dSpacing in Angstroms, given the crystal and millerIndex . if dSpacing == 0, get from xraylib.
    material_constants_library: xraylib or instance of DabaxXraylib, optional
        The pointer to the material library to be used to retrieve scattering data.

    Returns
    -------
    float
        The temperature factor.

    References
    ----------

    .. [1]: A. Freund, Nucl. Instrum. and Meth. 213 (1983) 495-501

    .. [2]: M. Sanchez del Rio and R. J. Dejus, "Status of XOP: an x-ray optics software toolkit",
         SPIE proc. vol. 5536 (2004) pp.171-174

    """

    # Written: Sergio Lordano, M. Sanchez del Rio based on XOP/IDL routine (2022-01-21)

    # Examples:
    # ---------
    #     ### using xraylib:
    #
    #     >>> calc_temperature_factor(80, crystal='Si', millerIndex=[3,1,1], debyeTemperature=644.92, dSpacing=0, atomicMass=0)
    #     0.983851994268226
    #
    #
    #     ### forcing it to use given dSpacing and atomicMass:
    #
    #     >>> calc_temperature_factor(80, crystal='Si', millerIndex=[1,1,1], debyeTemperature=644.92, atomicMass=28.09, dSpacing=3.1354163)
    #     0.9955698950510736


    def debyeFunc(x):
        return x / (numpy.exp(-x) - 1)

    def debyePhi(y):
        from scipy.integrate import quad
        integral = quad(lambda x: debyeFunc(x), 0, y)[0]
        return (1 / y) * integral

    planck = codata.h           # 6.62607015e-34    # codata.Planck
    Kb = codata.Boltzmann       # 1.380649e-23      # codata.Boltzmann
    atomicMassTokg = codata.m_u # 1.6605390666e-27  # codata.atomic_mass

    try:
        h, k, l = millerIndex
        crystalDict = material_constants_library.Crystal_GetCrystal(crystal)
        if (dSpacing == 0):
            dSpacing = material_constants_library.Crystal_dSpacing(crystalDict, h, k, l)
        if (atomicMass == 0):
            atomicMass = material_constants_library.AtomicWeight(xraylib.SymbolToAtomicNumber(crystal))

    except:
        print("material_constants_library not available. Please give dSpacing and atomicMass manually.")
        if ((dSpacing == 0) or (atomicMass == 0)):
            return numpy.nan

    atomicMass *= atomicMassTokg  # converting to [kg]
    dSpacing *= 1e-10  # converting to [m]

    x = debyeTemperature / (-1 * temperature)  # invert temperature sign (!!!)

    B0 = (3 * planck ** 2) / (2 * Kb * debyeTemperature * atomicMass)

    BT = 4 * B0 * debyePhi(x) / x

    ratio = 1 / (2 * dSpacing)

    M = (B0 + BT) * ratio ** 2

    temperatureFactor = numpy.exp(-M)

    return temperatureFactor


# used in crystalpy
def run_diff_pat(
    bragg_dict,
    preprocessor_file="xcrystal.bra",
    descriptor       = 'Si',
    MOSAIC           = 0,
    GEOMETRY         = 0,
    SCAN             = 2,
    UNIT             = 1,
    SCANFROM         = -100,
    SCANTO           = 100,
    SCANPOINTS       = 200,
    ENERGY           = 8000.0,
    ASYMMETRY_ANGLE  = 0.0,
    THICKNESS        = 0.7,
    MOSAIC_FWHM      = 0.1,
    RSAG             = 125.0,
    RMER             = 1290.0,
    ANISOTROPY       = 0,
    POISSON          = 0.22,
    CUT              = "2 -1 -1 ; 1 1 1 ; 0 0 0",
    FILECOMPLIANCE   = "mycompliance.dat",
    MILLER_INDEX_H   = 1, # needed if ANISOTROPY=1
    MILLER_INDEX_K   = 1, # needed if ANISOTROPY=1
    MILLER_INDEX_L   = 1, # needed if ANISOTROPY=1
    # material_constants_library=None,
    ):

    for file in ["diff_pat.dat", "diff_pat.gle", "diff_pat.par", "diff_pat.xop"]:
        try:
            os.remove(os.path.join(locations.home_bin_run(), file))
        except:
            pass

    if (GEOMETRY == 1) or (GEOMETRY == 3):
        if ASYMMETRY_ANGLE == 0.0:
            print(
                "xoppy_calc_xcrystal: WARNING: In xcrystal the asymmetry angle is the angle between Bragg planes and crystal surface," +
                "in BOTH Bragg and Laue geometries.")



    with open("xoppy.inp", "wt") as f:
        f.write("%s\n" % preprocessor_file)
        f.write("%d\n" % MOSAIC)
        f.write("%d\n" % GEOMETRY)

        if MOSAIC == 1:
            f.write("%g\n" % MOSAIC_FWHM)
            f.write("%g\n" % THICKNESS)
        else:
            f.write("%g\n" % THICKNESS)
            f.write("%g\n" % ASYMMETRY_ANGLE)

        scan_flag = 1 + SCAN

        f.write("%d\n" % scan_flag)

        f.write("%19.9f\n" % ENERGY)

        if scan_flag <= 3:
            f.write("%d\n" % UNIT)

        f.write("%g\n" % SCANFROM)
        f.write("%g\n" % SCANTO)
        f.write("%d\n" % SCANPOINTS)

        if MOSAIC > 1:  # bent
            f.write("%g\n" % RSAG)
            f.write("%g\n" % RMER)
            f.write("0\n")

            if ((descriptor == "Si") or (descriptor == "Si2") or (descriptor == "Si_NIST") or (
                    descriptor == "Ge") or descriptor == "Diamond"):
                pass
            else:  # not Si,Ge,Diamond
                if ((ANISOTROPY == 1) or (ANISOTROPY == 2)):
                    raise Exception(
                        "Anisotropy data not available for this crystal. Either use isotropic or use external compliance file. Please change and run again'")

            f.write("%d\n" % ANISOTROPY)

            if ANISOTROPY == 0:
                f.write("%g\n" % POISSON)
            elif ANISOTROPY == 1:
                # elas%crystalindex =  irint('CrystalIndex: 0,1,2=Si,3=Ge,4=Diamond: ')
                if descriptor == "Si":
                    f.write("0\n")
                elif descriptor == "Si2":
                    f.write("1\n")
                elif descriptor == "Si_NIST":
                    f.write("2\n")
                elif descriptor == "Ge":
                    f.write("3\n")
                elif descriptor == "Diamond":
                    f.write("4\n")
                else:
                    raise Exception(
                        "Cannot calculate anisotropy data for %s this crystal. Use Si, Ge, Diamond." % descriptor)

                f.write("%g\n" % ASYMMETRY_ANGLE)
                f.write("%d\n" % MILLER_INDEX_H)
                f.write("%d\n" % MILLER_INDEX_K)
                f.write("%d\n" % MILLER_INDEX_L)
            elif ANISOTROPY == 2:
                # elas%crystalindex =  irint('CrystalIndex: 0,1,2=Si,3=Ge,4=Diamond: ')
                if descriptor == "Si":
                    f.write("0\n")
                elif descriptor == "Si2":
                    f.write("1\n")
                elif descriptor == "Si_NIST":
                    f.write("2\n")
                elif descriptor == "Ge":
                    f.write("3\n")
                elif descriptor == "Diamond":
                    f.write("4\n")
                else:
                    raise Exception(
                        "Cannot calculate anisotropy data for %s this crystal. Use Si, Ge, Diamond." % descriptor)

                f.write("%g\n" % ASYMMETRY_ANGLE)
                # TODO: check syntax for CUT: Cut syntax is: valong_X valong_Y valong_Z ; vnorm_X vnorm_Y vnorm_Z ; vperp_x vperp_Y vperp_Z
                f.write("%s\n" % CUT.split(";")[0])
                f.write("%s\n" % CUT.split(";")[1])
                f.write("%s\n" % CUT.split(";")[2])
            elif ANISOTROPY == 3:
                f.write("%s\n" % FILECOMPLIANCE)

    if platform.system() == "Windows":
        command = "\"" + os.path.join(locations.home_bin(), 'diff_pat.exe\" < xoppy.inp')
    else:
        command = "'" + os.path.join(locations.home_bin(), 'diff_pat') + "' < xoppy.inp"
    print("Running command '%s' in directory: %s " % (command, locations.home_bin_run()))
    print("\n--------------------------------------------------------\n")
    os.system(command)
    print("\n--------------------------------------------------------\n")

if __name__ == "__main__":

    from dabax.dabax_xraylib import DabaxXraylib
    dx = DabaxXraylib()

    if False:
        tmp = bragg_calc(descriptor="Si",hh=1,kk=1,ll=1,temper=1.0,emin=5000.0,emax=15000.0,estep=100.0,
                   fileout="bragg_v2_xraylib.dat", material_constants_library=xraylib)

        tmp = bragg_calc(descriptor="Si",hh=1,kk=1,ll=1,temper=1.0,emin=5000.0,emax=15000.0,estep=100.0,
                   fileout="bragg_v2_dabax.dat", material_constants_library=dx)


        tmp = bragg_calc2(descriptor="Si",hh=1,kk=1,ll=1,temper=1.0,emin=5000.0,emax=15000.0,estep=100.0,
                   fileout="bragg_v2_xraylib.dat", material_constants_library=xraylib )

        tmp = bragg_calc2(descriptor="Si",hh=1,kk=1,ll=1,temper=1.0,emin=5000.0,emax=15000.0,estep=100.0,
                   fileout="bragg_v2_dabax.dat", material_constants_library=dx)


        tmp = bragg_calc2(descriptor="YB66", hh=1, kk=1, ll=1, temper=1.0,
                    emin=5000.0, emax=15000.0, estep=100.0,
                    ANISO_SEL=0,
                    fileout="bragg_yb66.dat",
                    do_not_prototype=0, # 0=use site groups (recommended), 1=use all individual sites
                    verbose=True,
                    material_constants_library=dx)


    if False:
        from xoppylib.xoppy_xraylib_util import mare_calc as mare_calc_old
        m1 = mare_calc_old("Si", 2, 2, 2, 3, 3, 3, 2e-8, 3, 1.54, 0.01, -20.0, 0.1)
        m2 = mare_calc("Si", 2, 2, 2, 3, 3, 3, 2e-8, 3, 1.54, 0.01, -20.0, 0.1,
                       material_constants_library=dx)
        print(m2)


    if False:
        t1 = calc_temperature_factor(80, crystal='Si', millerIndex=[3, 1, 1], debyeTemperature=644.92, dSpacing=0,
                                     atomicMass=0)


        ### forcing it to use given dSpacing and atomicMass:

        t2 = calc_temperature_factor(80, crystal='Si', millerIndex=[1, 1, 1], debyeTemperature=644.92, atomicMass=28.09,
                                     dSpacing=3.1354163)
        print("temperature factor Si: ", t1, t2)
        assert (numpy.abs(t1 - 0.983851994268226) < 1e-3)
        assert (numpy.abs(t2 - 0.9955698950510736) < 1e-3)


    if True:


        bragg_dictionary = bragg_calc2(descriptor="YB66", hh=1, kk=1, ll=1, temper=1.0,
                    emin=5000.0, emax=15000.0, estep=100.0,
                    ANISO_SEL=0,
                    fileout="bragg_yb66.dat",
                    do_not_prototype=0, # 0=use site groups (recommended), 1=use all individual sites
                    verbose=True,
                    material_constants_library=dx)


        ienergy = 6000.0
        dic2 = crystal_fh(bragg_dictionary, ienergy)
        print("Energy=%g eV FH=(%g,%g)" % (ienergy, dic2["STRUCT"].real, dic2["STRUCT"].imag))