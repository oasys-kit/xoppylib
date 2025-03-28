import numpy
import h5py

from crystalpy.diffraction.GeometryType import BraggDiffraction, LaueDiffraction
from crystalpy.diffraction.DiffractionSetupXraylib import DiffractionSetupXraylib
from crystalpy.diffraction.Diffraction import Diffraction
from crystalpy.util.Vector import Vector
from crystalpy.util.Photon import Photon

from xoppylib.mlayer import MLayer

def power1d_calc_multilayer_monochromator(filename,
                                          energies=numpy.linspace(7900, 8100, 200),
                                          grazing_angle_deg=0.4,
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
        np = f["MLayer/parameters/np"][()]
        npair = f["MLayer/parameters/npair"][()]
        thick = numpy.array(f["MLayer/parameters/thick"])

        f.close()

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
            print("np            = ", np            )
            print("npair         = ", npair         )
            print("thick         = ", thick         )
            print("Calculation for %d points of energy in [%.3f, %.3f] eV for theta_grazing=%.3f deg" % (
                energies.size, energies[0], energies[-1], grazing_angle_deg))
            print("=====================================\n\n")

        out = MLayer.initialize_from_bilayer_stack(
            material_S=materialS, density_S=densityS, roughness_S=roughnessS,
            material_E=material1, density_E=density1, roughness_E=mlroughness1[0],
            material_O=material2, density_O=density2, roughness_O=mlroughness2[0],
            bilayer_pairs=np,
            bilayer_thickness=thick[0],
            bilayer_gamma=gamma1[0],
        )
        rs, rp = out.scan_energy( energies, theta1=grazing_angle_deg, h5file="", verbose=verbose)

    except:
        raise Exception("Error reading file: %s" % filename)

    return rs, rp


def power1d_calc_bragg_monochromator(h_miller=1, k_miller=1, l_miller=1,
                        energy_setup=8000.0, energies=numpy.linspace(7900, 8100, 200),
                        calculation_method=0):

    r = numpy.zeros_like(energies)
    r_p = numpy.zeros_like(energies)
    harmonic = 1

    diffraction_setup_r = DiffractionSetupXraylib(geometry_type=BraggDiffraction(),  # GeometryType object
                                           crystal_name="Si",  # string
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
        ri = numpy.zeros_like(energies)
        ri_p = numpy.zeros_like(energies)
        for i in range(energies.size):
            try:
                diffraction_setup_r = DiffractionSetupXraylib(geometry_type=BraggDiffraction(),  # GeometryType object
                                                       crystal_name="Si",  # string
                                                       thickness=1,  # meters
                                                       miller_h=harmonic * h_miller,  # int
                                                       miller_k=harmonic * k_miller,  # int
                                                       miller_l=harmonic * l_miller,  # int
                                                       asymmetry_angle=0,  # radians
                                                       azimuthal_angle=0.0) # radians

                diffraction = Diffraction()

                energy = energies[i]
                deviation = 0.0  # angle_deviation_min + ia * angle_step
                angle = deviation + bragg_angle

                # calculate the components of the unitary vector of the incident photon scan
                # Note that diffraction plane is YZ
                yy = numpy.cos(angle)
                zz = - numpy.abs(numpy.sin(angle))
                photon = Photon(energy_in_ev=energy, direction_vector=Vector(0.0, yy, zz))

                # perform the calculation
                coeffs_r = diffraction.calculateDiffractedComplexAmplitudes(diffraction_setup_r, photon, calculation_method=calculation_method)
                # note the power 2 to get intensity (**2) for a single reflection

                r[i] += numpy.abs( coeffs_r['S'] ) ** 2
                ri[i] = numpy.abs( coeffs_r['S'] ) ** 2
                r_p[i] += numpy.abs( coeffs_r['P'] ) ** 2
                ri_p[i] = numpy.abs( coeffs_r['P'] ) ** 2
            except:
                print("Failed to calculate reflectivity at E=%g eV for %d%d%d reflection" % (energy,
                                        harmonic*h_miller, harmonic*k_miller, harmonic*l_miller))
        print("Max reflectivity S-polarized: ", ri.max(), " at energy: ", energies[ri.argmax()])
        print("Max reflectivity P-polarized: ", ri_p.max(), " at energy: ", energies[ri_p.argmax()])

    return r, r_p

def power1d_calc_laue_monochromator(h_miller=1, k_miller=1, l_miller=1,
                        energy_setup=8000.0, energies=numpy.linspace(7900, 8100, 200),
                        calculation_method=0, thickness=10e-6):

    r = numpy.zeros_like(energies)
    r_p = numpy.zeros_like(energies)
    harmonic = 1
    diffraction_setup_r = DiffractionSetupXraylib(geometry_type=LaueDiffraction(),  # GeometryType object
                                           crystal_name="Si",    # string
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
        ri = numpy.zeros_like(energies)
        ri_p = numpy.zeros_like(energies)
        for i in range(energies.size):
            try:
                diffraction_setup_r = DiffractionSetupXraylib(geometry_type=LaueDiffraction(),  # GeometryType object
                                                       crystal_name="Si",  # string
                                                       thickness=thickness,  # meters
                                                       miller_h=harmonic * h_miller,  # int
                                                       miller_k=harmonic * k_miller,  # int
                                                       miller_l=harmonic * l_miller,  # int
                                                       asymmetry_angle=numpy.pi/2,  # radians
                                                       azimuthal_angle=0)  # radians

                diffraction = Diffraction()

                energy = energies[i]
                deviation = 0.0  # angle_deviation_min + ia * angle_step
                angle = deviation + numpy.pi/2 + bragg_angle

                # calculate the components of the unitary vector of the incident photon scan
                # Note that diffraction plane is YZ
                yy = numpy.cos(angle)
                zz = - numpy.abs(numpy.sin(angle))
                photon = Photon(energy_in_ev=energy, direction_vector=Vector(0.0, yy, zz))

                # perform the calculation
                coeffs_r = diffraction.calculateDiffractedComplexAmplitudes(diffraction_setup_r, photon, calculation_method=calculation_method)
                # note the power 2 to get intensity
                r[i] += numpy.abs( coeffs_r['S'] ) ** 2
                ri[i] = numpy.abs( coeffs_r['S'] ) ** 2

                r_p[i] += numpy.abs( coeffs_r['P'] ) ** 2
                ri_p[i] = numpy.abs( coeffs_r['P'] ) ** 2
            except:
                print("Failed to calculate reflectivity at E=%g eV for %d%d%d reflection" % (energy,
                                        harmonic*h_miller, harmonic*k_miller, harmonic*l_miller))
        print("Max reflectivity S-polarized: ", ri.max(), " at energy: ", energies[ri.argmax()])
        print("Max reflectivity P-polarized: ", ri_p.max(), " at energy: ", energies[ri_p.argmax()])
    return r, r_p


if __name__ == "__main__":
    energies = numpy.linspace(3000, 30000, 200)
    rs, rp = power1d_calc_multilayer_monochromator("/users/srio/Oasys/multilayerTiC.h5",
                                                   energies=energies, grazing_angle_deg=0.4)
    from srxraylib.plot.gol import plot
    plot(energies, rs, energies, rp)
