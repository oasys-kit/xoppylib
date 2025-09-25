import numpy
import h5py

from crystalpy.diffraction.GeometryType import BraggDiffraction, LaueDiffraction
from crystalpy.diffraction.DiffractionSetupXraylib import DiffractionSetupXraylib
from crystalpy.diffraction.DiffractionSetupDabax import DiffractionSetupDabax
from crystalpy.diffraction.Diffraction import Diffraction
from crystalpy.util.Vector import Vector
from crystalpy.util.Photon import Photon

from xoppylib.mlayer import MLayer

try: import xraylib
except: pass
from dabax.dabax_xraylib import DabaxXraylib

def power1d_calc_multilayer_monochromator(filename,
                                          energies=numpy.linspace(7900, 8100, 200),
                                          grazing_angle_deg=0.4,
                                          material_constants_library=None,
                                          verbose=1):
    try:
        f = h5py.File(filename, 'r')
        density1 = f["MLayer/parameters/density1"][()]
        density2 = f["MLayer/parameters/density2"][()]
        densityS = f["MLayer/parameters/densityS"][()]
        gamma1 = numpy.array(f["MLayer/parameters/gamma1"])
        material1 = f["MLayer/parameters/material1"][()]
        material2 = f["MLayer/parameters/material2"][()]
        materialS = f["MLayer/parameters/materialS"][()]
        mlroughness1 = numpy.array(f["MLayer/parameters/mlroughness1"])
        mlroughness2 = numpy.array(f["MLayer/parameters/mlroughness2"])
        roughnessS = f["MLayer/parameters/roughnessS"][()]
        # np = f["MLayer/parameters/np"][()]
        npair = numpy.abs(f["MLayer/parameters/npair"][()])
        thick = numpy.array(f["MLayer/parameters/thick"])

        f.close()

        if isinstance(material1, bytes): material1 = material1.decode("utf-8")
        if isinstance(material2, bytes): material2 = material2.decode("utf-8")
        if isinstance(materialS, bytes): materialS = materialS.decode("utf-8")

        if verbose:
            print("===== data read from file: %s ======" % filename)
            print("density1      = ", density1      )
            print("density2      = ", density2      )
            print("densityS      = ", densityS      )
            print("gamma1        = ", gamma1        )
            print("material1  (even, closer to substrte) = ", material1     )
            print("material2  (odd)                      = ", material2     )
            print("materialS  (substrate)                = ", materialS     )
            print("mlroughness1  = ", mlroughness1  )
            print("mlroughness2  = ", mlroughness2  )
            print("roughnessS    = ", roughnessS    )
            # print("np            = ", np            )
            print("npair         = ", npair         )
            print("thick         = ", thick         )
            print("Calculation for %d points of energy in [%.3f, %.3f] eV for theta_grazing=%.3f deg" % (
                energies.size, energies[0], energies[-1], grazing_angle_deg))
            print("=====================================\n\n")
    except:
        raise Exception("Error reading file: %s" % filename)

    try:
        if isinstance(material_constants_library, DabaxXraylib):
            use_xraylib_or_dabax = 1
            dabax = material_constants_library
            if verbose:
                print(dabax.info())
        else: # xraylib
            use_xraylib_or_dabax = 0
            dabax = None

        out = MLayer.initialize_from_bilayer_stack(
            material_S=materialS, density_S=densityS, roughness_S=roughnessS,
            material_E=material1, density_E=density1, roughness_E=mlroughness1[0],
            material_O=material2, density_O=density2, roughness_O=mlroughness2[0],
            bilayer_pairs=npair,
            bilayer_thickness=thick[0],
            bilayer_gamma=gamma1[0],
            use_xraylib_or_dabax=use_xraylib_or_dabax,
            dabax=dabax,
        )

        rs, rp, _, _ = out.scan(h5file="",
            energyN=energies.size, energy1=energies[0], energy2=energies[-1],
            thetaN=1, theta1=grazing_angle_deg, theta2=grazing_angle_deg,
            verbose=verbose)

    except:
        raise Exception("Error calculating reflectivity: %s" % filename)

    return numpy.abs(rs.flatten())**2, numpy.abs(rp.flatten())**2


def power1d_calc_bragg_monochromator(h_miller=1, k_miller=1, l_miller=1,
                        energy_setup=8000.0, energies=numpy.linspace(7900, 8100, 200),
                        calculation_method=0,
                        crystal_descriptor="Si",
                        material_constants_library=None):

    r = numpy.zeros_like(energies)
    r_p = numpy.zeros_like(energies)
    harmonic = 1

    if material_constants_library is None:
        try:    material_constants_library = xraylib
        except: material_constants_library = DabaxXraylib()

    if isinstance(material_constants_library, DabaxXraylib):
        diffraction_setup_r = DiffractionSetupDabax(geometry_type=BraggDiffraction(),  # GeometryType object
                                                      crystal_name=crystal_descriptor,  # string
                                                      thickness=1,  # meters
                                                      miller_h=harmonic * h_miller,  # int
                                                      miller_k=harmonic * k_miller,  # int
                                                      miller_l=harmonic * l_miller,  # int
                                                      asymmetry_angle=0,  # radians
                                                      azimuthal_angle=0.0,
                                                      dabax=material_constants_library)  # radians
    else: # xraylib
        diffraction_setup_r = DiffractionSetupXraylib(geometry_type=BraggDiffraction(),  # GeometryType object
                                               crystal_name=crystal_descriptor,  # string
                                               thickness=1,        # meters
                                               miller_h=harmonic * h_miller,  # int
                                               miller_k=harmonic * k_miller,  # int
                                               miller_l=harmonic * l_miller,  # int
                                               asymmetry_angle=0,    # radians
                                               azimuthal_angle=0.0)  # radians


    bragg_angle = diffraction_setup_r.angleBragg(energy_setup)
    print("Bragg angle for Si%d%d%d at E=%f eV is %f deg" % (
        h_miller, k_miller, l_miller, energy_setup, bragg_angle * 180.0 / numpy.pi))
    nharmonics = int( energies.max() / energy_setup)

    if nharmonics < 1:
        nharmonics = 1
    print("Calculating %d harmonics" % nharmonics)

    for harmonic in range(1,nharmonics+1,2): # calculate only odd harmonics
        print("\nCalculating harmonic: ", harmonic)

        if isinstance(material_constants_library, DabaxXraylib):
            diffraction_setup_r = DiffractionSetupDabax(geometry_type=BraggDiffraction(),  # GeometryType object
                                                        crystal_name=crystal_descriptor,  # string
                                                        thickness=1,  # meters
                                                        miller_h=harmonic * h_miller,  # int
                                                        miller_k=harmonic * k_miller,  # int
                                                        miller_l=harmonic * l_miller,  # int
                                                        asymmetry_angle=0,  # radians
                                                        azimuthal_angle=0.0,
                                                        dabax=material_constants_library)  # radians

        else:
            diffraction_setup_r = DiffractionSetupXraylib(geometry_type=BraggDiffraction(),  # GeometryType object
                                                          crystal_name=crystal_descriptor,  # string
                                                          thickness=1,  # meters
                                                          miller_h=harmonic * h_miller,  # int
                                                          miller_k=harmonic * k_miller,  # int
                                                          miller_l=harmonic * l_miller,  # int
                                                          asymmetry_angle=0,  # radians
                                                          azimuthal_angle=0.0)  # radians

        diffraction = Diffraction()

        deviation = 0.0  # angle_deviation_min + ia * angle_step
        angle = deviation + bragg_angle

        # calculate the components of the unitary vector of the incident photon scan
        # Note that diffraction plane is YZ
        yy = numpy.cos(angle)
        zz = - numpy.abs(numpy.sin(angle))


        ri = numpy.zeros_like(energies)
        ri_p = numpy.zeros_like(energies)

        photon = Photon(energy_in_ev=energies, direction_vector=Vector(numpy.full(energies.size, 0.0),
                                                                       numpy.full(energies.size, float(yy)),
                                                                       numpy.full(energies.size, float(zz)))
                        )

        # perform the calculation
        coeffs_r = diffraction.calculateDiffractedComplexAmplitudes(diffraction_setup_r, photon, calculation_method=calculation_method)
        # note the power 2 to get intensity (**2) for a single reflection

        ri = numpy.abs( coeffs_r['S'] ) ** 2
        ri_p = numpy.abs( coeffs_r['P'] ) ** 2
        r = r + ri
        r_p = r_p + ri_p

        print("Max reflectivity S-polarized: ", ri.max(), " at energy: ", energies[ri.argmax()])
        print("Max reflectivity P-polarized: ", ri_p.max(), " at energy: ", energies[ri_p.argmax()])

    return numpy.array(r, dtype=float), numpy.array(r_p, dtype=float)

def power1d_calc_laue_monochromator(h_miller=1, k_miller=1, l_miller=1,
                        energy_setup=8000.0, energies=numpy.linspace(7900, 8100, 200),
                        calculation_method=0, thickness=10e-6,
                        crystal_descriptor="Si",
                        material_constants_library=None):

    r = numpy.zeros_like(energies)
    r_p = numpy.zeros_like(energies)
    harmonic = 1
    if material_constants_library is None:
        try:    material_constants_library = xraylib
        except: material_constants_library = DabaxXraylib()

    if isinstance(material_constants_library, DabaxXraylib):
        diffraction_setup_r = DiffractionSetupDabax(geometry_type=LaueDiffraction(),  # GeometryType object
                                               crystal_name=crystal_descriptor,    # string
                                               thickness=thickness,  # meters
                                               miller_h=harmonic * h_miller,  # int
                                               miller_k=harmonic * k_miller,  # int
                                               miller_l=harmonic * l_miller,  # int
                                               asymmetry_angle=numpy.pi/2,  # radians
                                               azimuthal_angle=0, # radians
                                               dabax=material_constants_library)
    else:
        diffraction_setup_r = DiffractionSetupXraylib(geometry_type=LaueDiffraction(),  # GeometryType object
                                               crystal_name=crystal_descriptor,    # string
                                               thickness=thickness,  # meters
                                               miller_h=harmonic * h_miller,  # int
                                               miller_k=harmonic * k_miller,  # int
                                               miller_l=harmonic * l_miller,  # int
                                               asymmetry_angle=numpy.pi/2,  # radians
                                               azimuthal_angle=0)  # radians


    bragg_angle = diffraction_setup_r.angleBragg(energy_setup)
    print("Bragg angle for Si%d%d%d at E=%f eV is %f deg" % (
        h_miller, k_miller, l_miller, energy_setup, bragg_angle * 180.0 / numpy.pi))
    nharmonics = int( energies.max() / energy_setup)

    if nharmonics < 1:
        nharmonics = 1
    print("Calculating %d harmonics" % nharmonics)

    for harmonic in range(1, nharmonics+1, 2): # calculate only odd harmonics
        print("\nCalculating harmonic: ", harmonic)

        if isinstance(material_constants_library, DabaxXraylib):
            diffraction_setup_r = DiffractionSetupDabax(geometry_type=LaueDiffraction(),  # GeometryType object
                                                        crystal_name=crystal_descriptor,  # string
                                                        thickness=thickness,  # meters
                                                        miller_h=harmonic * h_miller,  # int
                                                        miller_k=harmonic * k_miller,  # int
                                                        miller_l=harmonic * l_miller,  # int
                                                        asymmetry_angle=numpy.pi / 2,  # radians
                                                        azimuthal_angle=0,  # radians
                                                        dabax=material_constants_library)
        else:
            diffraction_setup_r = DiffractionSetupXraylib(geometry_type=LaueDiffraction(),  # GeometryType object
                                                          crystal_name=crystal_descriptor,  # string
                                                          thickness=thickness,  # meters
                                                          miller_h=harmonic * h_miller,  # int
                                                          miller_k=harmonic * k_miller,  # int
                                                          miller_l=harmonic * l_miller,  # int
                                                          asymmetry_angle=numpy.pi / 2,  # radians
                                                          azimuthal_angle=0)  # radians

        diffraction = Diffraction()

        deviation = 0.0  # angle_deviation_min + ia * angle_step
        angle = deviation + numpy.pi / 2 + bragg_angle

        # calculate the components of the unitary vector of the incident photon scan
        # Note that diffraction plane is YZ
        yy = numpy.cos(angle)
        zz = - numpy.abs(numpy.sin(angle))

        photon = Photon(energy_in_ev=energies, direction_vector=Vector(numpy.full(energies.size, 0.0),
                                                                       numpy.full(energies.size, float(yy)),
                                                                       numpy.full(energies.size, float(zz))))


        # perform the calculation
        coeffs_r = diffraction.calculateDiffractedComplexAmplitudes(diffraction_setup_r, photon, calculation_method=calculation_method)


        # note the power 2 to get intensity (**2) for a single reflection
        ri = numpy.abs( coeffs_r['S'] ) ** 2
        ri_p = numpy.abs( coeffs_r['P'] ) ** 2
        r = r + ri
        r_p = r_p + ri_p

        print("Max reflectivity S-polarized: ", ri.max(), " at energy: ", energies[ri.argmax()])
        print("Max reflectivity P-polarized: ", ri_p.max(), " at energy: ", energies[ri_p.argmax()])

    return numpy.array(r, dtype=float), numpy.array(r_p, dtype=float)


if __name__ == "__main__":
    if 1: # mlayer
        from dabax.dabax_xraylib import DabaxXraylib
        energies = numpy.linspace(3000, 30000, 20)
        rs, rp = power1d_calc_multilayer_monochromator("/users/srio/Oasys/multilayerTiC.h5",
                                                       energies=energies, grazing_angle_deg=0.4,
                                                       material_constants_library=DabaxXraylib())
        from srxraylib.plot.gol import plot
        plot(energies, rs, energies, rp)

