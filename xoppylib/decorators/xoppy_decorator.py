#
# material constants libraries (dabax and xraylib) decorated with xoppy functions
#
from dabax.dabax_xraylib import DabaxXraylib

try: import xraylib
except: pass

from xoppylib.crystals.tools import bragg_calc2, bragg_calc, crystal_fh, mare_calc
from xoppylib.scattering_functions import f0_calc


class XoppyDecorator(object):

    def f0_calc(self,
            MAT_FLAG,
            DESCRIPTOR,
            GRIDSTART,
            GRIDEND,
            GRIDN,
            FILE_NAME="",
            charge=0.0):

        return f0_calc(MAT_FLAG,
                DESCRIPTOR,
                GRIDSTART,
                GRIDEND,
                GRIDN,
                FILE_NAME=FILE_NAME,
                charge=charge,
                material_constants_library=self,)

    #
    # def f1f2_calc(self, descriptor, energy, theta=3.0e-3, F=0, density=None, rough=0.0, verbose=True):
    #     raise NotImplementedError
    #
    # def f1f2_calc_mix(self, descriptor, energy, theta=3.0e-3, F=0, density=None, rough=0.0, verbose=True):
    #     raise NotImplementedError
    #
    # def f1f2_calc_nist(self, descriptor, energy, theta=3.0e-3, F=0, density=None, rough=0.0, verbose=True):
    #     raise NotImplementedError
    #
    # def cross_calc(self, descriptor, energy, calculate=0, unit=None, density=None, verbose=True):
    #     raise NotImplementedError
    #
    # def cross_calc_mix(self, descriptor, energy, calculate=0, unit=None, parse_or_nist=0, density=None, verbose=True):
    #     raise NotImplementedError
    #
    # def cross_calc_nist(self, descriptor0, energy, calculate=0, unit=None, density=None, verbose=True):
    #     raise NotImplementedError
    #
    # def xpower_calc(self, energies=numpy.linspace(1000.0, 50000.0, 100), source=numpy.ones(100),
    #                 substance=["Be"], flags=[0], dens=["?"], thick=[0.5], angle=[3.0], roughness=0.0,
    #                 output_file=None):
    #     raise NotImplementedError
    #

    #
    def crystal_fh(self, input_dictionary, phot_in, theta=None, forceratio=0):
        return crystal_fh(input_dictionary, phot_in, theta=None, forceratio=0,
                          material_constants_library=self)

    def mare_calc(self, descriptor, H, K, L, HMAX, KMAX, LMAX, FHEDGE, DISPLAY, lambda1, deltalambda, PHI, DELTAPHI,
                  verbose=0):
        return mare_calc(self, descriptor, H, K, L, HMAX, KMAX, LMAX, FHEDGE, DISPLAY, lambda1, deltalambda, PHI, DELTAPHI,
                  material_constants_library=self, verbose=verbose)

    def bragg_calc2(self, descriptor="YB66", hh=1, kk=1, ll=1, temper=1.0, emin=5000.0, emax=15000.0, estep=100.0,
                    ANISO_SEL=0,
                    fileout=None,
                    do_not_prototype=0,  # 0=use site groups (recommended), 1=use all individual sites
                    verbose=True,
                    ):

        return bragg_calc2(descriptor=descriptor, hh=hh, kk=kk, ll=ll, temper=temper,
                           emin=emin, emax=emax, estep=estep,
                    ANISO_SEL=ANISO_SEL,
                    fileout=fileout,
                    do_not_prototype=do_not_prototype,  # 0=use site groups (recommended), 1=use all individual sites
                    verbose=verbose,
                    material_constants_library=self)

    def bragg_calc(self, descriptor="Si", hh=1, kk=1, ll=1, temper=1.0,
                   emin=5000.0, emax=15000.0, estep=100.0, fileout=None,):
        return bragg_calc(descriptor=descriptor, hh=hh, kk=kk, ll=ll, temper=temper,
                          emin=emin, emax=emax, estep=estep,
                          fileout=fileout,
                          material_constants_library=self)
