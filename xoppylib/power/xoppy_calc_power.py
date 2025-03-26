import numpy
from xoppylib.power.power1d_calc import power1d_calc

def xoppy_calc_power(
                energies                   = None,
                source                     = None,
                substance                  = None,
                thick                      = None,
                angle                      = None,
                dens                       = None,
                roughness                  = None,
                flags                      = None,
                nelements                  = 1,
                FILE_DUMP                  = 0,
                material_constants_library = None,
):

    if energies  is None:
        energies  = numpy.linspace(1000,10000,100)
        source = numpy.ones(100)
    if source    is None: source    = numpy.ones_like(energies)
    if substance is None: substance = ['Si'] * 5
    if thick     is None: thick     = [1.0] * 5
    if angle     is None: angle     = [0.0] * 5
    if dens      is None: dens      = ['?'] * 5
    if roughness is None: roughness = [0.0] * 5
    if flags     is None: flags     = [1] * 5



    substance = substance[0:nelements]
    thick     =     thick[0:nelements]
    angle     =     angle[0:nelements]
    dens      =      dens[0:nelements]
    roughness = roughness[0:nelements]
    flags     =     flags[0:nelements]


    if FILE_DUMP:
        output_file = "power.spec"
    else:
        output_file = None

    out_dictionary = power1d_calc(energies=energies, source=source, substance=substance,
                flags=flags, dens=dens, thick=thick, angle=angle, roughness=roughness,
                output_file=output_file, material_constants_library=material_constants_library)

    return out_dictionary

if __name__ == "__main__":
    import xraylib
    print(xoppy_calc_power(material_constants_library=xraylib))