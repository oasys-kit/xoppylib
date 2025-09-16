import numpy
import scipy.constants as codata

from xoppylib.xoppy_xraylib_util import descriptor_kind_index, density
from xoppylib.scattering_functions.fresnel import reflectivity_fresnel

# using: CS_Total_CP Refractive_Index_Re Refractive_Index_Im
try: import xraylib
except: pass

from dabax.dabax_xraylib import DabaxXraylib

def power1d_calc(energies=numpy.linspace(1000.0,50000.0,100), source=numpy.ones(100),
                substance=["Be"], flags=[0], dens=["?"], thick=[0.5], angle=[3.0], roughness=0.0,
                output_file=None, material_constants_library=None,):
    """
    Apply reflectivities/transmittivities of optical elements on a source spectrum

    :param energies: the array with photon energies in eV
    :param source: the spectral intensity or spectral power
    :param substance: a list with descriptors of each optical element  material
    :param flags: a list with 0 (filter or attenuator) or 1 (mirror) for all optical elements
    :param dens: a list with densities of o.e. materials. "?" is accepted for looking in the database
    :param thick: a list with the thickness in mm for all o.e.'s. Only applicable for filters
    :param angle: a list with the grazing angles in mrad for all o.e.'s. Only applicable for mirrors
    :param roughness:a list with the roughness RMS in A for all o.e.'s. Only applicable for mirrors
    :param output_file: name of the output file (default=None, no output file)
    :return: a dictionary with the results
    """


    nelem = len(substance)

    for i in range(nelem):
        kind = descriptor_kind_index(substance[i], material_constants_library=material_constants_library)
        if kind == -1:
            raise Exception("Bad descriptor/formula: %s"%substance[i])

        try:
            rho = float(dens[i])
        except:
            rho = density(substance[i], material_constants_library=material_constants_library)


        print("Density for %s: %g g/cm3"%(substance[i],rho))

        dens[i] = rho



    outArray = numpy.hstack( energies )
    outColTitles = ["Photon Energy [eV]"]
    outArray = numpy.vstack((outArray,source))
    outColTitles.append("Source")

    txt = "\n\n"
    txt += "*************************** power results ******************\n"
    if energies[0] != energies[-1]:
        txt += "  Source energy: start=%f eV, end=%f eV, points=%d \n"%(energies[0],energies[-1],energies.size)
    else:
        txt += "  Source energy: %f eV\n"%(energies[0])
    txt += "  Number of optical elements: %d\n"%(nelem)

    if energies[0] != energies[-1]:
        # I0 = source[0:-1].sum()*(energies[1]-energies[0])
        I0 = numpy.trapz(source, x=energies, axis=-1)
        P0 = numpy.trapz(source / (codata.e * energies), x=energies, axis=-1)
        txt += "\n  Incoming power (integral of spectrum): %g W (%g photons)\n" % (I0, P0)

        I1 = I0
        P1 = P0
    else:
        txt += "  Incoming power: %g W (%g photons) \n"%(source[0], source[0] / (codata.e * energies))
        I0  = source[0]
        I1 = I0
        P0 = source[0] / (codata.e * energies)
        P1 = P0



    cumulated = source

    for i in range(nelem):
        #info oe
        if flags[i] == 0:
            txt += '      *****   oe '+str(i+1)+'  [Filter] *************\n'
            txt += '      Material:  %s\n'%(substance[i])
            txt += '      Density:   %f g/cm^3\n'%(dens[i])
            txt += '      thickness: %f mm\n'%(thick[i])
        else:
            txt += '      *****   oe '+str(i+1)+'  [Mirror] *************\n'
            txt += '      Material:   %s\n'%(substance[i])
            txt += '      Density:    %f g/cm^3\n'%(dens[i])
            txt += '      grazing angle: %f mrad\n'%(angle[i])
            txt += '      roughness: %f A\n'%(roughness[i])


        if flags[i] == 0: # filter

            if isinstance(material_constants_library, DabaxXraylib):
                tmp = material_constants_library.CS_Total_CP(substance[i],energies/1000.0)
            else:
                tmp = numpy.zeros(energies.size)
                for j,energy in enumerate(energies):
                    tmp[j] = material_constants_library.CS_Total_CP(substance[i],energy/1000.0)

            trans = numpy.exp(-tmp*dens[i]*(thick[i]/10.0))
            outArray = numpy.vstack((outArray,tmp))
            outColTitles.append("[oe %i] Total CS cm2/g"%(1+i))

            outArray = numpy.vstack((outArray,tmp*dens[i]))
            outColTitles.append("[oe %i] Mu cm^-1"%(1+i))


            outArray = numpy.vstack((outArray,trans))
            outColTitles.append("[oe %i] Transmitivity "% (1+i))

            outArray = numpy.vstack((outArray,1.0-trans))
            outColTitles.append("[oe %i] Absorption "% (1+i))

            absorbed = cumulated * (1.0-trans)
            cumulated *= trans

        if flags[i] == 1: # mirror
            if isinstance(material_constants_library, DabaxXraylib):
                tmp = material_constants_library.Refractive_Index_Re(substance[i],energies/1000.0,dens[i])
            else:
                tmp = numpy.zeros(energies.size)
                for j,energy in enumerate(energies):
                    tmp[j] = material_constants_library.Refractive_Index_Re(substance[i],energy/1000.0,dens[i])
            delta = 1.0 - tmp
            outArray = numpy.vstack((outArray,delta))
            outColTitles.append("[oe %i] 1-Re[n]=delta"%(1+i))

            if isinstance(material_constants_library, DabaxXraylib):
                beta = material_constants_library.Refractive_Index_Im(substance[i],energies/1000.0,dens[i])
            else:
                beta = numpy.zeros(energies.size)
                for j,energy in enumerate(energies):
                    beta[j] = material_constants_library.Refractive_Index_Im(substance[i],energy/1000.0,dens[i])
            outArray = numpy.vstack((outArray,beta))
            outColTitles.append("[oe %i] Im[n]=beta"%(1+i))

            outArray = numpy.vstack((outArray,delta/beta))
            outColTitles.append("[oe %i] delta/beta"%(1+i))

            (rs,rp,runp) = reflectivity_fresnel(refraction_index_beta=beta,refraction_index_delta=delta,\
                                        grazing_angle_mrad=angle[i],roughness_rms_A=roughness[i],\
                                        photon_energy_ev=energies)
            outArray = numpy.vstack((outArray,rs))
            outColTitles.append("[oe %i] Reflectivity-s"%(1+i))

            outArray = numpy.vstack((outArray,1.0-rs))
            outColTitles.append("[oe %i] Transmitivity"%(1+i))

            absorbed = cumulated * (1.0 - rs)
            cumulated *= rs

        if energies[0] != energies[-1]:
            I2 = numpy.trapz( cumulated, x=energies, axis=-1)
            P2 = numpy.trapz( cumulated / (codata.e * energies) , x=energies, axis=-1)
            txt += "      Outcoming power: %f  W (%g photons)\n" % (I2, P2)
            txt += "      Absorbed power:  %f W (%g photons)\n" % (I1 - I2, P1 - P2)
            txt += "      Normalized Outcoming Power: %f\n" % (I2 / I0)
            if flags[i] == 0:
                pass
                txt += "      Absorbed dose %f Gy.(mm^2 beam cross section)/s\n "%((I1-I2)/(dens[i]*thick[i]*1e-6))
            I1 = I2
        else:
            I2 = cumulated[0]
            P2 = cumulated[0] / (codata.e * energies)
            txt += "      Outcoming power: %f W (%g photons)\n" % (cumulated[0], P2)
            txt += "      Absorbed power: %f W (%g photons)\n" % (I1 - I2, P1 - P2)
            txt += "      Normalized Outcoming Power: %f\n" % (I2 / I0)
            I1 = I2

        outArray = numpy.vstack((outArray,absorbed))
        outColTitles.append("Spectral power absorbed in oe #%i"%(1+i))

        outArray = numpy.vstack((outArray,cumulated))
        outColTitles.append("Spectral power after oe #%i"%(1+i))

    ncol = len(outColTitles)
    npoints = energies.size

    if output_file is not None:
        f = open(output_file,"w")
        f.write("#F "+output_file+"\n")
        f.write("\n")
        f.write("#S 1 power: properties of optical elements\n")

        txt2 = txt.splitlines()
        for i in range(len(txt2)):
            f.write("#UINFO %s\n"%(txt2[i]))

        f.write("#N %d\n"%(ncol))
        f.write("#L")
        for i in range(ncol):
            f.write("  "+outColTitles[i])
        f.write("\n")

        for i in range(npoints):
                f.write((" %e "*ncol+"\n")%(tuple(outArray[:,i].tolist())))

        f.close()
        print("File written to disk: " + output_file)

    return {"data":outArray,"labels":outColTitles,"info":txt}