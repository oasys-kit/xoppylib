"""
Fresnel reflectivity and interface reflectivity for X-ray mirrors.
"""
import numpy
import scipy.constants as codata

def reflectivity_fresnel(refraction_index_delta=1e-5,refraction_index_beta=0.0,\
                 grazing_angle_mrad=3.0,roughness_rms_A=0.0,photon_energy_ev=10000.0):
    """
    Calculates the reflectivity of an interface using Fresnel formulas.

    Code adapted from XOP and SHADOW

    :param refraction_index_delta: scalar or array with delta (n=1-delta+i beta)
    :param refraction_index_beta: scalar or array with beta (n=1-delta+i beta)
    :param grazing_angle_mrad: scalar with grazing angle in mrad
    :param roughness_rms_A: scalar with roughness rms in Angstroms
    :param photon_energy_ev: scalar or array with photon energies in eV
    :return: (rs,rp,runp) the s-polarized, p-pol and unpolarized reflectivities
    """
    # ;
    # ; calculation of reflectivity (piece of code adapted from shadow/abrefc)
    # ;
    #
    theta1 = grazing_angle_mrad*1e-3     # in rad
    rough1 = roughness_rms_A

    # ; epsi = 1 - alpha - i gamma
    # alpha = 2.0D0*k*f1
    # gamma = 2.0D0*k*f2
    alpha = 2*refraction_index_delta
    gamma = 2*refraction_index_beta

    rho = (numpy.sin(theta1))**2 - alpha
    rho += numpy.sqrt((numpy.sin(theta1)**2 - alpha)**2 + gamma**2)
    rho = numpy.sqrt(rho/2)

    rs1 = 4*(rho**2)*(numpy.sin(theta1) - rho)**2 + gamma**2
    rs2 = 4*(rho**2)*(numpy.sin(theta1) + rho)**2 + gamma**2
    rs = rs1/rs2

    ratio1 = 4*rho**2 * (rho*numpy.sin(theta1)-numpy.cos(theta1)**2)**2 + gamma**2*numpy.sin(theta1)**2
    ratio2 = 4*rho**2 * (rho*numpy.sin(theta1)+numpy.cos(theta1)**2)**2 + gamma**2*numpy.sin(theta1)**2
    ratio = ratio1/ratio2

    rp = rs*ratio
    runp = 0.5 * (rs + rp)

    m2ev = codata.c * codata.h / codata.e

    wavelength_m = m2ev / photon_energy_ev
    debyewaller = numpy.exp( -(4.0*numpy.pi*numpy.sin(theta1)*rough1/(wavelength_m*1e10))**2 )

    return rs*debyewaller, rp*debyewaller, runp*debyewaller

def interface_reflectivity(alpha,gamma,theta1):
    """
    Calculates the reflectivity of an interface using Fresnel formulas.

    Code adapted from XOP and SHADOW
    :param alpha: the array with alpha values (alpha=2 delta, n=1-delta+i beta)
    :param gamma: the array with gamma (gamma=2 beta)
    :param theta1: a scalar with the grazing angle in rad
    :return:
    """

    rho =  numpy.sin(theta1)**2 - alpha
    rho += numpy.sqrt( ( (numpy.sin(theta1))**2 - alpha)**2 + gamma**2)
    rho = numpy.sqrt(rho / 2)
    # ;** Computes now the reflectivity for s-pol


    rs1 = 4 * (rho**2) * (numpy.sin(theta1) - rho)**2 + gamma**2
    rs2 = 4 * (rho**2) * (numpy.sin(theta1) + rho)**2 + gamma**2
    rs = rs1 / rs2

    # ;** Computes now the polarization ratio

    ratio1 = 4 * rho**2 * (rho * numpy.sin(theta1) - numpy.cos(theta1) ** 2)**2 + gamma**2 * numpy.sin(theta1)**2
    ratio2 = 4 * rho**2 * (rho * numpy.sin(theta1) + numpy.cos(theta1) ** 2)**2 + gamma**2 * numpy.sin(theta1)**2
    ratio = ratio1 / ratio2

    rp = rs * ratio
    runp = 0.5 * (rs + rp)

    return rs,rp,runp