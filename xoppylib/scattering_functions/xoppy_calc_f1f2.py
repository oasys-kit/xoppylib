import numpy
from xoppylib.scattering_functions.f1f2_calc import f1f2_calc, f1f2_calc_mix, f1f2_calc_nist
# from xoppylib.xoppy_xraylib_util import f1f2_calc, f1f2_calc_mix, f1f2_calc_nist

def xoppy_calc_f1f2(
    descriptor                 = "Si",
    density                    = "?",
    MAT_FLAG                   = 0,
    CALCULATE                  = 1,
    GRID                       = 0,
    GRIDSTART                  = 5000.0,
    GRIDEND                    = 25000.0,
    GRIDN                      = 100,
    THETAGRID                  = 0,
    ROUGH                      = 0.0,
    THETA1                     = 2.0,
    THETA2                     = 5.0,
    THETAN                     = 50,
    DUMP_TO_FILE               = 0,
    FILE_NAME                  = "f1f2.dat",
    material_constants_library = None,
):

    if GRID == 0: # standard energy grid
        energy = numpy.arange(0,500)
        elefactor = numpy.log10(10000.0 / 30.0) / 300.0
        energy = 10.0 * 10**(energy * elefactor)
    elif GRID == 1: # user energy grid
        if GRIDN == 1:
            energy = numpy.array([GRIDSTART])
        else:
            energy = numpy.linspace(GRIDSTART,GRIDEND,GRIDN)
    elif GRID == 2: # single energy point
        energy = numpy.array([GRIDSTART])

    if THETAGRID == 0:
        theta = numpy.array([THETA1])
    else:
        theta = numpy.linspace(THETA1,THETA2,THETAN)


    CALCULATE_items=['f1', 'f2', 'delta', 'beta', 'mu [cm^-1]', 'mu [cm^2/g]', 'Cross Section [barn]', 'reflectivity-s', 'reflectivity-p', 'reflectivity-unpol', 'delta/beta ']

    out = numpy.zeros((energy.size,theta.size))
    for i,itheta in enumerate(theta):
        if MAT_FLAG == 0: # element
            tmp = f1f2_calc(descriptor,energy,1e-3*itheta,F=1+CALCULATE,rough=ROUGH,density=density,
                            material_constants_library=material_constants_library)
            out[:,i] = tmp
        elif MAT_FLAG == 1:  # compound
            tmp = f1f2_calc_mix(descriptor,energy,1e-3*itheta,F=1+CALCULATE,rough=ROUGH,density=density,
                            material_constants_library=material_constants_library)
            out[:,i] = tmp
        elif MAT_FLAG == 2:  # nist list
            tmp = f1f2_calc_nist(descriptor,energy,1e-3*itheta,F=1+CALCULATE,rough=ROUGH,density=density,
                            material_constants_library=material_constants_library)
            out[:,i] = tmp

    if ((energy.size == 1) and (theta.size == 1)):
        info = "** Single value calculation E=%g eV, theta=%g mrad, Result(F=%d)=%g "%(energy[0],theta[0],1+CALCULATE,out[0,0])
        labels = ["Energy [eV]",CALCULATE_items[CALCULATE]]
        tmp = numpy.vstack((energy,out[:,0]))
        out_dict = {"application":"xoppy","name":"xf12","info":info, "data":tmp,"labels":labels}
    elif theta.size == 1:
        tmp = numpy.vstack((energy,out[:,0]))
        labels = ["Energy [eV]",CALCULATE_items[CALCULATE]]
        out_dict = {"application":"xoppy","name":"xf12","data":tmp,"labels":labels}
    elif energy.size == 1:
        tmp = numpy.vstack((theta,out[0,:]))
        labels = ["Theta [mrad]",CALCULATE_items[CALCULATE]]
        out_dict = {"application":"xoppy","name":"xf12","data":tmp,"labels":labels}
    else:
        labels = [r"energy[eV]",r"theta [mrad]"]
        out_dict = {"application":"xoppy","name":"xf12","data2D":out,"dataX":energy,"dataY":theta,"labels":labels}

    #
    #
    #
    if DUMP_TO_FILE:
        with open(FILE_NAME, "w") as file:
            try:
                file.write("#F %s\n"%FILE_NAME)
                file.write("\n#S 1 xoppy f1f2 results\n")

                print("data shape",out_dict["data"].shape)
                print("labels: ",out_dict["labels"])
                file.write("#N 2\n")
                file.write("#L  %s  %s\n"%(out_dict["labels"][0],out_dict["labels"][1]))
                out = out_dict["data"]
                for j in range(out.shape[1]):
                    file.write(("%19.12e  "*out.shape[0]+"\n")%tuple(out[i,j] for i in range(out.shape[0])))
                file.close()
                print("File written to disk: %s \n"%FILE_NAME)
            except:
                raise Exception("CrossSec: The data could not be dumped onto the specified file!\n")


    return out_dict

if __name__ == "__main__":

    #
    # script to make the calculations (created by XOPPY:xf1f2)
    #
    from xoppylib.scattering_functions.xoppy_calc_f1f2 import xoppy_calc_f1f2

    try: import xraylib
    except: print("xraylib is not available.")

    from dabax.dabax_xraylib import DabaxXraylib

    out_dict = xoppy_calc_f1f2(
        descriptor="Si",
        density=2.33,
        MAT_FLAG=0,
        CALCULATE=1,
        GRID=0,
        GRIDSTART=5000.0,
        GRIDEND=25000.0,
        GRIDN=100,
        THETAGRID=0,
        ROUGH=0.0,
        THETA1=2.0,
        THETA2=5.0,
        THETAN=50,
        DUMP_TO_FILE=0,
        FILE_NAME="f1f2.dat",
        material_constants_library=DabaxXraylib(file_f1f2="f1f2_Sasaki.dat"),
    )

    #
    # example plot
    #
    if True:
        from srxraylib.plot.gol import plot, plot_image

        try:
            plot(out_dict["data"][0, :], out_dict["data"][-1, :],
                 xtitle=out_dict["labels"][0], ytitle=out_dict["labels"][1], title="xf1f2",
                 xlog=True, ylog=True, show=True)
        except:
            plot_image(out_dict["data2D"], out_dict["dataX"], out_dict["dataY"],
                       xtitle='Energy [eV]', ytitle='Theta [mrad]', title='Reflectivity',
                       aspect='auto', show=True)
    #
    # end script
    #

