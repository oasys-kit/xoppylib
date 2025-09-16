import numpy

try: import xraylib
except: pass

from xoppylib.scattering_functions.cross_calc import cross_calc, cross_calc_mix, cross_calc_nist

def xoppy_calc_crosssec(
    descriptor                 = "Si",
    density                    = "?",
    MAT_FLAG                   = 2,
    CALCULATE                  = 1,
    GRID                       = 0,
    GRIDSTART                  = 100.0,
    GRIDEND                    = 10000.0,
    GRIDN                      = 200,
    UNIT                       = 0,
    DUMP_TO_FILE               = 0,
    FILE_NAME                  = "CrossSec.dat",
    material_constants_library = None,
):

    if GRID == 0:
        energy = numpy.arange(0,500)
        elefactor = numpy.log10(10000.0 / 30.0) / 300.0
        energy = 10.0 * 10**(energy * elefactor)
    elif GRID == 1:
        if GRIDN == 1:
            energy = numpy.array([GRIDSTART])
        else:
            energy = numpy.linspace(GRIDSTART,GRIDEND,GRIDN)
    elif GRID == 2:
        energy = numpy.array([GRIDSTART])

    if MAT_FLAG == 0: # element
        out =  cross_calc(descriptor,energy,calculate=CALCULATE,density=density,
                          material_constants_library=material_constants_library)
    elif MAT_FLAG == 1: # compound parse
        out =  cross_calc_mix(descriptor,energy,calculate=CALCULATE,density=density,
                          material_constants_library=material_constants_library)
    elif MAT_FLAG == 2: # NIST compound
        out =  cross_calc_nist(descriptor,energy,calculate=CALCULATE,
                          material_constants_library=material_constants_library)

    calculate_items = ['Total','PhotoElectric','Rayleigh','Compton','Total minus Rayleigh']
    unit_items = ['barn/atom','cm^2','cm^2/g','cm^-1']
    if energy.size > 1:
        tmp_x = out[0,:].copy()
        tmp_y = out[UNIT+1,:].copy()
        tmp = numpy.vstack((tmp_x,tmp_y))
        labels = ["Photon energy [eV]","%s cross section [%s]"%(calculate_items[CALCULATE],unit_items[UNIT])]
        to_return = {"application":"xoppy","name":"xcrosssec","data":tmp,"labels":labels}
    else:
        tmp = numpy.vstack((out[0,0], out[UNIT+1, 0]))
        labels = ["Photon energy [eV]","%s cross section [%s]"%(calculate_items[CALCULATE],unit_items[UNIT])]
        txt = "xoppy_calc_xcrosssec: Calculated %s cross section: %g %s"%(calculate_items[CALCULATE],out[UNIT+1,0],unit_items[UNIT])
        print(txt)
        to_return  = {"application":"xoppy","name":"xcrosssec","data":tmp,"labels":labels,"info":txt}

    if DUMP_TO_FILE:
        with open(FILE_NAME, "w") as file:
            try:
                file.write("#F %s\n"%FILE_NAME)
                file.write("\n#S 1 xoppy CrossSec results\n")
                file.write("#N 5\n")
                tmp = "#L  Photon energy [eV]"
                for unit_item in unit_items:
                    tmp += "  %s [%s]"%(calculate_items[CALCULATE],unit_item)
                tmp += "\n"
                file.write(tmp)
                for j in range(out.shape[1]):
                    # file.write("%19.12e  "%energy[j])
                    file.write(("%19.12e  "*out.shape[0]+"\n")%tuple(out[i,j] for i in range(out.shape[0])))
                file.close()
                print("File written to disk: %s \n"%FILE_NAME)
            except:
                raise Exception("CrossSec: The data could not be dumped onto the specified file!\n")

    return to_return

if __name__ == "__main__":
    from dabax.dabax_xraylib import DabaxXraylib
    tmp = xoppy_calc_crosssec(
            descriptor="Water, Liquid",
            density=None,
            MAT_FLAG=2,
            CALCULATE=1,
            GRID=0,
            GRIDSTART=100.0,
            GRIDEND=10000.0,
            GRIDN=200,
            UNIT=0,
            DUMP_TO_FILE=0,
            FILE_NAME="CrossSec.dat",
            material_constants_library=DabaxXraylib(),
    )