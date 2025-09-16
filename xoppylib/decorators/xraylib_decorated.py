#
# xraylib decorated with xoppy functions
#
try: import xraylib
except: pass

from xoppylib.decorators.xoppy_decorator import XoppyDecorator


class XraylibDecorated(XoppyDecorator):

    # copied and modified from C:\Users\srio\Miniconda3\Lib\site-packages\xraylib.py
    def Crystal_MakeCopy(self, crystal):
        return xraylib.Crystal_MakeCopy(crystal)

    def Crystal_Free(self, crystal):
        return xraylib.Crystal_Free(crystal)

    def Crystal_GetCrystal(self, material):
        return xraylib.Crystal_GetCrystal(material)

    def Bragg_angle(self, crystal, energy, i_miller, j_miller, k_miller):
        return xraylib.Bragg_angle(crystal, energy, i_miller, j_miller, k_miller)

    def Q_scattering_amplitude(self, crystal, energy, i_miller, j_miller, k_miller, rel_angle):
        return xraylib.Q_scattering_amplitude(crystal, energy, i_miller, j_miller, k_miller, rel_angle)

    def Atomic_Factors(self, Z, energy, q, debye_factor):
        return xraylib.Atomic_Factors(Z, energy, q, debye_factor)

    def Crystal_F_H_StructureFactor(self, crystal, energy, i_miller, j_miller, k_miller, debye_factor, rel_angle):
        return xraylib.Crystal_F_H_StructureFactor(crystal, energy, i_miller, j_miller, k_miller, debye_factor,
                                                    rel_angle)

    def Crystal_F_H_StructureFactor_Partial(self, crystal, energy, i_miller, j_miller, k_miller, debye_factor, rel_angle,
                                            f0_flag, f_prime_flag, f_prime2_flag):
        return xraylib.Crystal_F_H_StructureFactor_Partial(crystal, energy, i_miller, j_miller, k_miller, debye_factor,
                                                            rel_angle, f0_flag, f_prime_flag, f_prime2_flag)

    def Crystal_UnitCellVolume(self, crystal):
        return xraylib.Crystal_UnitCellVolume(crystal)

    def Crystal_dSpacing(self, crystal, i_miller, j_miller, k_miller):
        return xraylib.Crystal_dSpacing(crystal, i_miller, j_miller, k_miller)

    def Crystal_GetCrystalsList(self, ):
        return xraylib.Crystal_GetCrystalsList()

    def GetCompoundDataNISTByName(self, compoundString):
        return xraylib.GetCompoundDataNISTByName(compoundString)

    def GetCompoundDataNISTByIndex(self, compoundIndex):
        return xraylib.GetCompoundDataNISTByIndex(compoundIndex)

    def GetCompoundDataNISTList(self, ):
        return xraylib.GetCompoundDataNISTList()

    def CompoundParser(self, compound):
        return xraylib.CompoundParser(compound)

    def GetRadioNuclideDataByName(self, radioNuclideString):
        return xraylib.GetRadioNuclideDataByName(radioNuclideString)

    def GetRadioNuclideDataByIndex(self, radioNuclideIndex):
        return xraylib.GetRadioNuclideDataByIndex(radioNuclideIndex)

    def GetRadioNuclideDataList(self, ):
        return xraylib.GetRadioNuclideDataList()

    def XRayInit(self, ):
        return xraylib.XRayInit()

    def SetHardExit(self, hard_exit):
        return xraylib.SetHardExit(hard_exit)

    def SetExitStatus(self, exit_status):
        return xraylib.SetExitStatus(exit_status)

    def GetExitStatus(self, ):
        return xraylib.GetExitStatus()

    def SetErrorMessages(self, status):
        return xraylib.SetErrorMessages(status)

    def GetErrorMessages(self, ):
        return xraylib.GetErrorMessages()

    def AtomicWeight(self, Z):
        return xraylib.AtomicWeight(Z)

    def ElementDensity(self, Z):
        return xraylib.ElementDensity(Z)

    def CS_Total(self, Z, E):
        return xraylib.CS_Total(Z, E)

    def CS_Photo(self, Z, E):
        return xraylib.CS_Photo(Z, E)

    def CS_Rayl(self, Z, E):
        return xraylib.CS_Rayl(Z, E)

    def CS_Compt(self, Z, E):
        return xraylib.CS_Compt(Z, E)

    def CS_KN(self, E):
        return xraylib.CS_KN(E)

    def CS_Energy(self, Z, E):
        return xraylib.CS_Energy(Z, E)

    def CSb_Total(self, Z, E):
        return xraylib.CSb_Total(Z, E)

    def CSb_Photo(self, Z, E):
        return xraylib.CSb_Photo(Z, E)

    def CSb_Rayl(self, Z, E):
        return xraylib.CSb_Rayl(Z, E)

    def CSb_Compt(self, Z, E):
        return xraylib.CSb_Compt(Z, E)

    def DCS_Thoms(self, theta):
        return xraylib.DCS_Thoms(theta)

    def DCS_KN(self, E, theta):
        return xraylib.DCS_KN(E, theta)

    def DCS_Rayl(self, Z, E, theta):
        return xraylib.DCS_Rayl(Z, E, theta)

    def DCS_Compt(self, Z, E, theta):
        return xraylib.DCS_Compt(Z, E, theta)

    def DCSb_Rayl(self, Z, E, theta):
        return xraylib.DCSb_Rayl(Z, E, theta)

    def DCSb_Compt(self, Z, E, theta):
        return xraylib.DCSb_Compt(Z, E, theta)

    def DCSP_Thoms(self, theta, phi):
        return xraylib.DCSP_Thoms(theta, phi)

    def DCSP_KN(self, E, theta, phi):
        return xraylib.DCSP_KN(E, theta, phi)

    def DCSP_Rayl(self, Z, E, theta, phi):
        return xraylib.DCSP_Rayl(Z, E, theta, phi)

    def DCSP_Compt(self, Z, E, theta, phi):
        return xraylib.DCSP_Compt(Z, E, theta, phi)

    def DCSPb_Rayl(self, Z, E, theta, phi):
        return xraylib.DCSPb_Rayl(Z, E, theta, phi)

    def DCSPb_Compt(self, Z, E, theta, phi):
        return xraylib.DCSPb_Compt(Z, E, theta, phi)

    def FF_Rayl(self, Z, q):
        return xraylib.FF_Rayl(Z, q)

    def SF_Compt(self, Z, q):
        return xraylib.SF_Compt(Z, q)

    def MomentTransf(self, E, theta):
        return xraylib.MomentTransf(E, theta)

    def LineEnergy(self, Z, line):
        return xraylib.LineEnergy(Z, line)

    def FluorYield(self, Z, shell):
        return xraylib.FluorYield(Z, shell)

    def CosKronTransProb(self, Z, trans):
        return xraylib.CosKronTransProb(Z, trans)

    def EdgeEnergy(self, Z, shell):
        return xraylib.EdgeEnergy(Z, shell)

    def JumpFactor(self, Z, shell):
        return xraylib.JumpFactor(Z, shell)

    def CS_FluorLine(self, Z, line, E):
        return xraylib.CS_FluorLine(Z, line, E)

    def CSb_FluorLine(self, Z, line, E):
        return xraylib.CSb_FluorLine(Z, line, E)

    def RadRate(self, Z, line):
        return xraylib.RadRate(Z, line)

    def ComptonEnergy(self, E0, theta):
        return xraylib.ComptonEnergy(E0, theta)

    def Fi(self, Z, E):
        return xraylib.Fi(Z, E)

    def Fii(self, Z, E):
        return xraylib.Fii(Z, E)

    def CS_Photo_Total(self, Z, E):
        return xraylib.CS_Photo_Total(Z, E)

    def CSb_Photo_Total(self, Z, E):
        return xraylib.CSb_Photo_Total(Z, E)

    def CS_Photo_Partial(self, Z, shell, E):
        return xraylib.CS_Photo_Partial(Z, shell, E)

    def CSb_Photo_Partial(self, Z, shell, E):
        return xraylib.CSb_Photo_Partial(Z, shell, E)

    def CS_FluorLine_Kissel(self, Z, line, E):
        return xraylib.CS_FluorLine_Kissel(Z, line, E)

    def CSb_FluorLine_Kissel(self, Z, line, E):
        return xraylib.CSb_FluorLine_Kissel(Z, line, E)

    def CS_FluorLine_Kissel_Cascade(self, Z, line, E):
        return xraylib.CS_FluorLine_Kissel_Cascade(Z, line, E)

    def CSb_FluorLine_Kissel_Cascade(self, Z, line, E):
        return xraylib.CSb_FluorLine_Kissel_Cascade(Z, line, E)

    def CS_FluorLine_Kissel_Nonradiative_Cascade(self, Z, line, E):
        return xraylib.CS_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E)

    def CSb_FluorLine_Kissel_Nonradiative_Cascade(self, Z, line, E):
        return xraylib.CSb_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E)

    def CS_FluorLine_Kissel_Radiative_Cascade(self, Z, line, E):
        return xraylib.CS_FluorLine_Kissel_Radiative_Cascade(Z, line, E)

    def CSb_FluorLine_Kissel_Radiative_Cascade(self, Z, line, E):
        return xraylib.CSb_FluorLine_Kissel_Radiative_Cascade(Z, line, E)

    def CS_FluorLine_Kissel_no_Cascade(self, Z, line, E):
        return xraylib.CS_FluorLine_Kissel_no_Cascade(Z, line, E)

    def CSb_FluorLine_Kissel_no_Cascade(self, Z, line, E):
        return xraylib.CSb_FluorLine_Kissel_no_Cascade(Z, line, E)

    def CS_Total_Kissel(self, Z, E):
        return xraylib.CS_Total_Kissel(Z, E)

    def CSb_Total_Kissel(self, Z, E):
        return xraylib.CSb_Total_Kissel(Z, E)

    def ElectronConfig(self, Z, shell):
        return xraylib.ElectronConfig(Z, shell)

    def CS_Total_CP(self, compound, E):
        return xraylib.CS_Total_CP(compound, E)

    def CS_Photo_CP(self, compound, E):
        return xraylib.CS_Photo_CP(compound, E)

    def CS_Rayl_CP(self, compound, E):
        return xraylib.CS_Rayl_CP(compound, E)

    def CS_Compt_CP(self, compound, E):
        return xraylib.CS_Compt_CP(compound, E)

    def CSb_Total_CP(self, compound, E):
        return xraylib.CSb_Total_CP(compound, E)

    def CSb_Photo_CP(self, compound, E):
        return xraylib.CSb_Photo_CP(compound, E)

    def CSb_Rayl_CP(self, compound, E):
        return xraylib.CSb_Rayl_CP(compound, E)

    def CSb_Compt_CP(self, compound, E):
        return xraylib.CSb_Compt_CP(compound, E)

    def DCS_Rayl_CP(self, compound, E, theta):
        return xraylib.DCS_Rayl_CP(compound, E, theta)

    def DCS_Compt_CP(self, compound, E, theta):
        return xraylib.DCS_Compt_CP(compound, E, theta)

    def DCSb_Rayl_CP(self, compound, E, theta):
        return xraylib.DCSb_Rayl_CP(compound, E, theta)

    def DCSb_Compt_CP(self, compound, E, theta):
        return xraylib.DCSb_Compt_CP(compound, E, theta)

    def DCSP_Rayl_CP(self, compound, E, theta, phi):
        return xraylib.DCSP_Rayl_CP(compound, E, theta, phi)

    def DCSP_Compt_CP(self, compound, E, theta, phi):
        return xraylib.DCSP_Compt_CP(compound, E, theta, phi)

    def DCSPb_Rayl_CP(self, compound, E, theta, phi):
        return xraylib.DCSPb_Rayl_CP(compound, E, theta, phi)

    def DCSPb_Compt_CP(self, compound, E, theta, phi):
        return xraylib.DCSPb_Compt_CP(compound, E, theta, phi)

    def CS_Photo_Total_CP(self, compound, E):
        return xraylib.CS_Photo_Total_CP(compound, E)

    def CSb_Photo_Total_CP(self, compound, E):
        return xraylib.CSb_Photo_Total_CP(compound, E)

    def CS_Total_Kissel_CP(self, compound, E):
        return xraylib.CS_Total_Kissel_CP(compound, E)

    def CSb_Total_Kissel_CP(self, compound, E):
        return xraylib.CSb_Total_Kissel_CP(compound, E)

    def CS_Energy_CP(self, compound, E):
        return xraylib.CS_Energy_CP(self, compound, E)

    def Refractive_Index_Re(self, compound, E, density):
        return xraylib.Refractive_Index_Re(compound, E, density)

    def Refractive_Index_Im(self, compound, E, density):
        return xraylib.Refractive_Index_Im(compound, E, density)

    def Refractive_Index(self, compound, E, density):
        return xraylib.Refractive_Index(compound, E, density)

    def ComptonProfile(self, Z, pz):
        return xraylib.ComptonProfile(Z, pz)

    def ComptonProfile_Partial(self, Z, shell, pz):
        return xraylib.ComptonProfile_Partial(Z, shell, pz)

    def AtomicLevelWidth(self, Z, shell):
        return xraylib.AtomicLevelWidth(Z, shell)

    def AugerRate(self, Z, auger_trans):
        return xraylib.AugerRate(Z, auger_trans)

    def AugerYield(self, Z, shell):
        return xraylib.AugerYield(Z, shell)


if __name__ == "__main__":
    xrl = XraylibDecorated()

    # print(xrl.f1f2_calc("Si", 8000, theta=3.0e-3, F=0, density=None, rough=0.0, verbose=True))

    tmp = xrl.bragg_calc(descriptor="Si",hh=1,kk=1,ll=1,temper=1.0,emin=5000.0,emax=15000.0,estep=100.0,
               fileout="bragg_v2_xraylib.dat")
