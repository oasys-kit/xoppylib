"""
DabaxXraylib instance decorated with xoppy calculation functions.
"""
#
# dabax decorated with xoppy functions
#
from dabax.dabax_xraylib import DabaxXraylib
from xoppylib.decorators.xoppy_decorator import XoppyDecorator

class DabaxDecorated(DabaxXraylib, XoppyDecorator):
    def __init__(self,
                 dabax_repository=None,
                 file_f0="f0_InterTables.dat",
                 file_f1f2="f1f2_Windt.dat",
                 file_CrossSec="CrossSec_EPDL97.dat",
                 file_Crystals="Crystals.dat",
                 ):
        DabaxXraylib.__init__(self,
                dabax_repository=dabax_repository,
                file_f0=file_f0,
                file_f1f2=file_f1f2,
                file_CrossSec=file_CrossSec,
                file_Crystals=file_Crystals,
                              )


if __name__ == "__main__":

    dx = DabaxDecorated()

    # print(dx.f1f2_calc("Si", 8000, theta=3.0e-3, F=0, density=None, rough=0.0, verbose=True))


    tmp = dx.bragg_calc(descriptor="Si",hh=1,kk=1,ll=1,temper=1.0,emin=5000.0,emax=15000.0,estep=100.0,
               fileout="bragg_v2_dabax.dat")

    tmp = dx.bragg_calc2(descriptor="YB66", hh=1, kk=1, ll=1, temper=1.0,
                emin=5000.0, emax=15000.0, estep=100.0,
                ANISO_SEL=0,
                fileout="bragg_yb66.dat",
                do_not_prototype=0, # 0=use site groups (recommended), 1=use all individual sites
                verbose=False)