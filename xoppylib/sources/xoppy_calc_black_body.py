"""
XOPPY black body radiation spectrum calculator.
"""

import numpy
import scipy.constants as codata

#
# Black body
#
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

def xoppy_calc_black_body(
    TITLE       = "Thermal source: Planck distribution",
    TEMPERATURE = 1200000.0,
    E_MIN       = 10.0,
    E_MAX       = 1000.0,
    NPOINTS     = 500,
    ):

    #
    # text info
    #
    kb = codata.Boltzmann / codata.e # eV/K
    txt = ' \n'
    txt += 'Results of Black Body Radiation: Planck distribution\n'
    txt += 'TITLE: %s'%TITLE
    txt += ' \n'
    txt += '-------------------------------------------------------------\n'
    txt += 'Temperature           = %g K\n'%(TEMPERATURE)
    txt += 'Minimum photon energy = %g eV\n'%(E_MIN)
    txt += 'Maximum photon energy = %g eV\n'%(E_MAX)
    txt += '-------------------------------------------------------------\n'
    txt += 'Kb*T                = %g eV\n'%(TEMPERATURE*kb)
    txt += 'Peak at 2.822*Kb*T  = %g eV\n'%(2.822*TEMPERATURE*kb)
    txt += '-------------------------------------------------------------\n'

    # print(txt)

    #
    # calculation data
    #
    e_ev = numpy.linspace(E_MIN,E_MAX,NPOINTS)
    e_kt = e_ev/(TEMPERATURE*kb)
    brightness=3.146e11*(TEMPERATURE*kb)**3*e_kt**3/(numpy.exp(e_kt)-1)
    a3 = numpy.zeros((4,NPOINTS))
    a3[0,:] = e_ev
    a3[1,:] = e_kt
    a3[2,:] = brightness
    a3[3,:] = brightness*1e3*codata.e

    labels = ["Photon energy [eV]","Photon energy/(Kb*T)", "Brightness [Photons/sec/mm2/mrad2/0.1%bw]", "Spectral Power [Watts/eV/mrad2/mm2]"]

    return {"application":"xoppy","name":"black_body","data":a3.T,"labels":labels,"info":txt}