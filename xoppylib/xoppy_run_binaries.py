import sys
import os
import platform
from xoppylib.xoppy_util import locations
import numpy
import scipy.constants as codata

#
# WS
#
def xoppy_calc_ws(ENERGY=7.0, CUR=100.0, PERIOD=8.5, N=28.0, KX=0.0, KY=8.739999771118164, \
                  EMIN=1000.0, EMAX=100000.0, NEE=2000, D=30.0, XPC=0.0, YPC=0.0, XPS=2.0, YPS=2.0, NXP=10, NYP=10):
    print("Inside xoppy_calc_ws. ")

    try:
        with open("ws.inp", "wt") as f:
            f.write("inputs from xoppy \n")
            f.write("%f     %f\n" % (ENERGY, CUR))
            f.write("%f  %d  %f  %f\n" % (PERIOD, N, KX, KY))
            f.write("%f  %f   %d\n" % (EMIN, EMAX, NEE))
            f.write("%f  %f  %f  %f  %f  %d  %d\n" % (D, XPC, YPC, XPS, YPS, NXP, NYP))
            f.write("%d  \n" % (4))

        if platform.system() == "Windows":
            command = "\"" + os.path.join(locations.home_bin(), 'ws.exe') + "\""
        else:
            command = "'" + os.path.join(locations.home_bin(), 'ws') + "'"
        print("Running command '%s' in directory: %s \n" % (command, locations.home_bin_run()))
        # TODO try to capture the text output of the external code
        os.system(command)

        # write spec file
        txt = open("ws.out").readlines()
        outFile = os.path.join(locations.home_bin_run(), "ws.spec")
        f = open(outFile, "w")

        f.write("#F ws.spec\n")
        f.write("\n")
        f.write("#S 1 ws results\n")
        f.write("#N 4\n")
        f.write("#L  Energy(eV)  Flux(photons/s/0.1%bw)  Spectral power(W/eV)  Cumulated power(W)\n")
        cum = 0.0
        estep = (EMAX - EMIN) / (NEE - 1)
        for i in txt:
            tmp = i.strip(" ")
            if tmp[0].isdigit():
                tmp1 = numpy.fromstring(tmp, dtype=float, sep=' ')
                cum += tmp1[1] * codata.e * 1e3
                f.write("%f  %g  %g  %g \n" % (tmp1[0], tmp1[1], tmp1[1] * codata.e * 1e3, cum * estep))
            else:
                f.write("#UD " + tmp)
        f.close()
        print("File written to disk: ws.spec")

        # print output file
        # for line in txt:
        #     print(line, end="")
        print("Results written to file: %s" % os.path.join(locations.home_bin_run(), 'ws.out'))

        return outFile
    except Exception as e:
        raise e

#
# TUNING CURVES
#
def xoppy_calc_xtc(
    ENERGY         = 7.0,
    CURRENT        = 100.0,
    ENERGY_SPREAD  = 0.00096,
    SIGX           = 0.274,
    SIGY           = 0.011,
    SIGX1          = 0.0113,
    SIGY1          = 0.0036,
    PERIOD         = 3.23,
    NP             = 70,
    EMIN           = 2950.0,
    EMAX           = 13500.0,
    N              = 40,
    HARMONIC_FROM  = 1,
    HARMONIC_TO    = 15,
    HARMONIC_STEP  = 2,
    HELICAL        = 0,
    METHOD         = 1,
    NEKS           = 100, ):


    for file in ["tc.inp","tc.out"]:
        try:
            os.remove(os.path.join(locations.home_bin_run(),file))
        except:
            pass

    with open("tc.inp", "wt") as f:
        f.write("TS called from xoppy\n")
        f.write("%10.3f %10.2f %10.6f %s\n"%(ENERGY,CURRENT,ENERGY_SPREAD,"Ring-Energy(GeV) Current(mA) Beam-Energy-Spread"))
        f.write("%10.4f %10.4f %10.4f %10.4f %s\n"%(SIGX,SIGY,SIGX1,SIGY1,"Sx(mm) Sy(mm) Sx1(mrad) Sy1(mrad)"))
        f.write("%10.3f %d %s\n"%(PERIOD,NP,"Period(cm) N"))
        f.write("%10.1f %10.1f %d %s\n"%(EMIN,EMAX,N,"Emin Emax Ne"))
        f.write("%d %d %d %s\n"%(HARMONIC_FROM,HARMONIC_TO,HARMONIC_STEP,"Hmin Hmax Hstep"))
        f.write("%d %d %d %d %s\n"%(HELICAL,METHOD,1,NEKS,"Helical Method Print_K Neks"))
        f.write("foreground\n")


    if platform.system() == "Windows":
        command = "\"" + os.path.join(locations.home_bin(),'tc.exe') + "\""
    else:
        command = "'" + os.path.join(locations.home_bin(), 'tc') + "'"


    print("Running command '%s' in directory: %s "%(command, locations.home_bin_run()))
    print("\n--------------------------------------------------------\n")
    os.system(command)
    print("Output file: %s"%("tc.out"))
    print("\n--------------------------------------------------------\n")

    #
    # parse result files to exchange object
    #


    with open("tc.out","r") as f:
        lines = f.readlines()

    # print output file
    # for line in lines:
    #     print(line, end="")


    # remove returns
    lines = [line[:-1] for line in lines]
    harmonics_data = []

    # separate numerical data from text
    floatlist = []
    harmoniclist = []
    txtlist = []
    for line in lines:
        try:
            tmp = line.strip()

            if tmp.startswith("Harmonic"):
                harmonic_number = int(tmp.split("Harmonic")[1].strip())

                if harmonic_number != HARMONIC_FROM:
                    harmonics_data[-1][1] = harmoniclist
                    harmoniclist = []

                harmonics_data.append([harmonic_number, None])

            tmp = float(line.strip()[0])

            floatlist.append(line)
            harmoniclist.append(line)
        except:
            txtlist.append(line)

    harmonics_data[-1][1] = harmoniclist

    data = numpy.loadtxt(floatlist)

    for index in range(0, len(harmonics_data)):
        # print (harmonics_data[index][0], harmonics_data[index][1])
        harmonics_data[index][1] = numpy.loadtxt(harmonics_data[index][1])

    return data, harmonics_data


def xoppy_calc_xtcap(
        ENERGY        = 6.0,
        CURRENT       = 200.0,
        ENERGY_SPREAD = 0.00138,
        SIGX          = 0.0148,
        SIGY          = 0.0037,
        SIGX1         = 0.0029,
        SIGY1         = 0.0015,
        PERIOD        = 2.8,
        NP            = 84,
        EMIN          = 3217.0,
        EMAX          = 11975.0,
        N             = 50,
        DISTANCE      = 30.0,
        XPS           = 1.0,
        YPS           = 1.0,
        XPC           = 0.0,
        YPC           = 0.0,
        HARMONIC_FROM = 1,
        HARMONIC_TO   = 7,
        HARMONIC_STEP = 2,
        HRED          = 0,
        HELICAL       = 0,
        NEKS          = 100,
        METHOD        = 0,
        BSL           = 0,
    ):

    for file in ["tcap.inp","tcap.out","tcap.log"]:
        try:
            os.remove(os.path.join(locations.home_bin_run(),file))
        except:
            pass

    with open("tcap.inp", "wt") as f:
        f.write("TCAP called from xoppy\n")
        f.write("%10.3f %10.2f %10.6f %s\n"%(ENERGY,CURRENT,ENERGY_SPREAD,"Ring-Energy(GeV) Current(mA) Beam-Energy-Spread"))
        f.write("%10.4f %10.4f %10.4f %10.4f %s\n"%(SIGX,SIGY,SIGX1,SIGY1,"Sx(mm) Sy(mm) Sx1(mrad) Sy1(mrad)"))
        f.write("%10.3f %d %s\n"%(PERIOD,NP,"Period(cm) N"))
        f.write("%10.1f %10.1f %d %s\n"%(EMIN,EMAX,N,"Emin Emax Ne"))
        f.write("%10.3f %10.3f %10.3f %10.3f %10.3f %d %d %s\n"%(DISTANCE,XPC,YPC,XPS,YPS,10,10,"d xpc ypc xps yps nxp nyp"))
        f.write("%d %d %d %d %s\n"%(HARMONIC_FROM,HARMONIC_TO,HARMONIC_STEP,HRED,"Hmin Hmax Hstep Hreduction"))
        f.write("%d %d %d %d %d %s\n"%(HELICAL,METHOD,1,NEKS,BSL,"Helical Method Print_K Neks Bsl-Subtr "))
        f.write("foreground\n")


    if platform.system() == "Windows":
        command = "\"" + os.path.join(locations.home_bin(),'tcap.exe') + "\""
    else:
        command = "'" + os.path.join(locations.home_bin(), 'tcap') + "'"


    print("Running command '%s' in directory: %s "%(command, locations.home_bin_run()))
    print("\n--------------------------------------------------------\n")
    # os.system(command)

    #
    #   catch the optut and write the output to a log file as well as print it.
    #
    retvalue = os.popen(command).read()
    print(retvalue)

    with open("tcap.log", "wt") as f:
        f.write(retvalue)

    print("Output file: '%s/tcap.out'"%(os.getcwd()) )
    print("\n--------------------------------------------------------\n")

    #
    # parse result files to exchange object
    #
    with open("tcap.out","r") as f:
        lines = f.readlines()

    # remove returns
    lines = [line[:-1] for line in lines]
    harmonics_data = []

    # separate numerical data from text
    floatlist = []
    harmoniclist = []
    txtlist = []
    for line in lines:
        try:
            tmp = line.strip()

            if tmp.startswith("Harmonic"):
#                   harmonic_number = int(tmp.split("Harmonic")[1].strip())
                harmonic_number = int(tmp.split()[1])

                if harmonic_number != HARMONIC_FROM:
                    harmonics_data[-1][1] = harmoniclist
                    harmoniclist = []

                harmonics_data.append([harmonic_number, None])

            tmp = float(line.strip()[0])

            floatlist.append(line)
            harmoniclist.append(line)
        except:
            txtlist.append(line)

    harmonics_data[-1][1] = harmoniclist

    data = numpy.loadtxt(floatlist)

    for index in range(0, len(harmonics_data)):
        # print (harmonics_data[index][0], harmonics_data[index][1])
        harmonics_data[index][1] = numpy.loadtxt(harmonics_data[index][1])


    return data, harmonics_data


#
# YAUP
#
def run_external_binary(binary="ls", post_command="", info=""):

    if platform.system() == "Windows":
        command = "\"" + os.path.join(locations.home_bin(), '%s.exe' % binary) + "\""
    else:
        command = "'" + os.path.join(locations.home_bin(), binary) + "'"

    command += " " + post_command
    print("Running command '%s' in directory: %s " % (command, locations.home_bin_run()))
    print("\n--------------------------------------------------------\n")

    os.system(command)

    if info != "":
        print(info)
    print("\n--------------------------------------------------------\n")

def xoppy_calc_yaup(
        #yaup
        TITLE            = "YAUP EXAMPLE (ESRF BL-8)",
        PERIOD           = 4.0,
        NPER             = 42,
        NPTS             = 40,
        EMIN             = 3000.0,
        EMAX             = 30000.0,
        NENERGY          = 100,
        ENERGY           = 6.04,
        CUR              = 0.1,
        SIGX             = 0.426,
        SIGY             = 0.085,
        SIGX1            = 0.017,
        SIGY1            = 0.0085,
        D                = 30.0,
        XPC              = 0.0,
        YPC              = 0.0,
        XPS              = 2.0,
        YPS              = 2.0,
        NXP              = 69,
        NYP              = 69,
        MODE             = 4,
        NSIG             = 2,
        TRAJECTORY       = "new+keep",
        XSYM             = "yes",
        HANNING          = 0,
        BFILE            = "undul.bf",
        TFILE            = "undul.traj",
        # B field
        BFIELD_FLAG      = 1,
        BFIELD_ASCIIFILE = "",
        PERIOD_BFIELD    = 4.0,
        NPER_BFIELD      = 42,
        NPTS_BFIELD      = 40,
        IMAGNET          = 0,
        ITYPE            = 0,
        K                = 1.38,
        GAP              = 2.0,
        GAPTAP           = 10.0,
        FILE             = "undul.bf",
        I2TYPE           = 0,
        A1               = 0.5,
        A2               = 1.0,
    ):


    for file in ["bfield.inp","bfield.out","bfield.dat","u2txt_bfield.inp",
                 "yaup.inp", "yaup-0.out","undul.bf",
                 "u2txt_traj.inp","undul_traj.dat"]:
        try:
            os.remove(os.path.join(locations.home_bin_run(),file))
        except:
            print("Failed to remove file: %s " %  (os.path.join(locations.home_bin_run(),file)) )

    if BFIELD_FLAG == 0:

        #TODO: test this option...
        message = ''
        message += 'This option takes an ASCII file and convert it to YAUP format.'
        message += 'The text file should be column-formatted, and contain three colums:'
        message += ' z, B(z), and phi(z), where the z s are equidistant with step '
        message += ' PERIOD/NPTS. See HELP/YAUP for definitions of PERIOD and NPTS.'
        message += ' There should be NPTS*NPER+1 lines in the ASCII file.'

        print(message)

        f = open('txt2u.inp', 'w')
        f.write("%s\n" % (BFIELD_ASCIIFILE) )
        f.write("%s\n" % (BFILE))
        f.write("%g\n" % (PERIOD_BFIELD))
        f.write("%d\n" % (PERIOD_BFIELD))
        f.write("%d\n" % (NPTS_BFIELD) )
        f.close

        run_external_binary(binary="txt2u", post_command="< txt2u.inp",
                                 info="Output file should be: %s" % BFILE)

    elif BFIELD_FLAG == 1:
        with open("bfield.inp", "wt") as f:
            f.write("%g\n" % (PERIOD_BFIELD))
            f.write("%d\n" % (NPER_BFIELD))
            f.write("%d\n" % (NPTS_BFIELD))
            f.write("%d\n" % (1 + ITYPE))
            if ITYPE == 0:
                f.write("%g\n" % (K))
            elif ITYPE == 1:
                f.write("%g\n" % (GAP))
                f.write("%g\n" % (GAPTAP))
            f.write("%s\n" % (FILE))



        with open("u2txt_bfield.inp", "wt") as f:
            f.write("1\n")
            f.write("%s\n" % (FILE))
            f.write("bfield.dat\n")

        if IMAGNET == 0:
            run_external_binary(binary="bfield", post_command="< bfield.inp > bfield.out", info="Output file: bfield.out")
        elif IMAGNET == 1:
            run_external_binary(binary="bfield2", post_command="< bfield.inp > bfield.out", info="Output file: bfield.out")

        run_external_binary(binary="u2txt", post_command="< u2txt_bfield.inp", info="Output file should be bfield.dat")

    elif BFIELD_FLAG == 2:
        n = NPER
        lambdau = PERIOD_BFIELD
        npts_per = NPTS_BFIELD
        if I2TYPE == 0: # L.Rebuffi 29/1/2026
            b1 = A1
            b2 = A2
        else:
            b1 = A1/0.934/PERIOD_BFIELD
            b2 = A2/0.934/PERIOD_BFIELD


        und_len = lambdau * npts_per
        z = numpy.arange( n * npts_per + 1) / float( n * npts_per)
        z *= und_len

        bmod = numpy.arange(n * npts_per + 1) / float( n * npts_per) * (b2 - b1) + b1
        berr = numpy.arange(n * npts_per + 1) * 0.0
        bphase = 2.0 * numpy.pi / lambdau * z
        btot = bmod * numpy.sin(bphase)


        f = open("bfield.dat", 'w')
        f.write('# Columns: z(cm), ampl(tesla), phserr, total(tesla)\n')
        f.write('# total = ampl * sin ( twopi/period*z + phserr ) \n')
        f.write('# period= %g; nper= %d; npts=%d \n' % (lambdau, n, npts_per))
        for i in range(z.size):
            f.write("%g  %g  %g  %g\n" % (z[i], bmod[i], berr[i], btot[i]))
        f.close()
        print("File written to disk: bfield.dat")

        f = open("bfield2.dat", 'w')
        for i in range(z.size):
            if i != 0: f.write("\n")
            f.write("%g  %g  %g" % (z[i], bmod[i], bphase[i]))
        f.close()
        print("File written to disk: bfield.dat")


        with open("txt2u.inp", "w") as f:
            f.write("bfield2.dat\n")
            f.write("%s\n" % BFILE)
            f.write("%g\n" % (PERIOD_BFIELD))
            f.write("%d\n" % (NPER_BFIELD))
            f.write("%d\n" % (NPTS_BFIELD))

        run_external_binary("txt2u", " < txt2u.inp", "File written to disk should be: %s " % BFILE )


    input = "\n"
    input += ";Magnet parameters\n"
    input += "PERIOD=%g NPER=%d NPTS=%d\n" % (PERIOD, NPER, NPTS)
    input += "\n"
    input += ";Photon energy\n"
    input += "EMIN=%g EMAX=%g NE=%d\n" % (EMIN, EMAX, NENERGY)
    input += "\n"
    input += ";Storage ring\n"
    input += "ENERGY=%g CURRENT=%g\n" % (ENERGY, CUR)
    input += " SIGX=%g SIGY=%g\n" % (SIGX, SIGY)
    input += "SIGX1=%g SIGY1=%g\n" % (SIGX1, SIGY1)
    input += "\n"
    input += ";Pinhole (mm or mrad)\n"
    input += "DISTANCE=%g\n" % D
    input += "XPC=%g XPS=%g NXP=%d\n" % (XPC, XPS, NXP)
    input += "YPC=%g YPS=%g NYP=%d\n" % (YPC, YPS, NYP)
    input += "\n"
    input += ";Calculation parameter\n"
    input += "MODE=%d NSIG=%d   TRAJECTORY=new+keep\n" % (MODE, NSIG)
    input += "XSYM=yes  HANNING=%d\n" % HANNING
    input += "\n"
    input += ";Filenames\n"
    input += 'BFILE="undul.bf"\n'
    input += 'TFILE="undul.traj"\n'
    input += "\n"
    input += "END\n"

    with open("yaup.inp", "wt") as f:
        f.write(input)

    run_external_binary(binary="yaup", post_command="", info="Output file should be XXX")

    with open("u2txt_traj.inp", "wt") as f:
        f.write("2\n")
        f.write("%s\n" % (TFILE))
        f.write("undul_traj.dat\n")

    run_external_binary(binary="u2txt", post_command="< u2txt_traj.inp", info="Output file should be undul_traj.dat")
    #
    # add spectral power and cumulated power
    #

    results = numpy.loadtxt("yaup-0.out", skiprows=33)
    e = results[:,0]
    f = results[:,1]

    power_in_spectrum = f.sum() * 1e3 * codata.e * (e[1] - e[0])
    print("\nPower from integral of spectrum: %8.3f W" % (power_in_spectrum))
    codata_mee = codata.m_e * codata.c ** 2 / codata.e  # electron mass in eV
    gamma = ENERGY * 1e9 / codata_mee
    ptot = (NPER / 6) * codata.value('characteristic impedance of vacuum') * \
           CUR * codata.e * 2 * numpy.pi * codata.c * gamma ** 2 * (K ** 2 ) / (PERIOD * 1e-2)
    print("\nTotal power radiated by the undulator with fully opened slits [W]: %g \n" % (ptot))
    print("\nRatio Power from integral of spectrum over Total emitted power: %5.4f" % (power_in_spectrum / ptot))

    spectral_power = f * codata.e * 1e3

    try:
        cumulated_power = spectral_power.cumsum() * numpy.abs(e[0] - e[1])
    except:
        cumulated_power = 0.0
    run_external_binary(binary="u2txt", post_command="< u2txt_traj.inp",
                             info="Output file should be undul_traj.dat")

    return e,f,spectral_power,cumulated_power

#
# X-ray tubes
#

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

def xoppy_calc_xtube_w(VOLTAGE=100.0,RIPPLE=0.0,AL_FILTER=0.0):
    print("Inside xoppy_calc_xtube_w. ")

    for file in ["tasmip_tmp.dat"]:
        try:
            os.remove(os.path.join(locations.home_bin_run(),file))
        except:
            pass

    try:
        with open("xoppy.inp","wt") as f:
            f.write("%f\n%f\n%f\n"%(VOLTAGE,RIPPLE,AL_FILTER))


        if platform.system() == "Windows":
            command = "\"" + os.path.join(locations.home_bin(),'tasmip.exe\" < xoppy.inp')
        else:
            command = "'" + os.path.join(locations.home_bin(), 'tasmip') + "' < xoppy.inp"

        print("Running command '%s' in directory: %s \n"%(command,locations.home_bin_run()))
        print("\n--------------------------------------------------------\n")
        os.system(command)
        print("\n--------------------------------------------------------\n")

        print("\nOutput file: %s/tasmip_tmp.dat\n" % (locations.home_bin_run()))
        return "tasmip_tmp.dat"
    except Exception as e:
        raise e


def xoppy_calc_xtubes(ITUBE=0,VOLTAGE=30.0):
    print("Inside xoppy_calc_xtubes. ")

    for file in ["xtubes_tmp.dat"]:
        try:
            os.remove(os.path.join(locations.home_bin_run(),file))
        except:
            pass

    try:
        with open("xoppy.inp","wt") as f:
            f.write("%d\n%f\n"%(ITUBE+1,VOLTAGE))

        if platform.system() == "Windows":
            command = "\"" + os.path.join(locations.home_bin(),'xtubes.exe\" < xoppy.inp')
        else:
            command = "'" + os.path.join(locations.home_bin(), "xtubes") + "' < xoppy.inp"

        print("Running command '%s' in directory: %s "%(command, locations.home_bin_run()))
        print("\n--------------------------------------------------------\n")
        os.system(command)
        print("\n--------------------------------------------------------\n")

        print("\nOutput file: %s/xtubes_tmp.dat\n" % (locations.home_bin_run()))
        return os.path.join(locations.home_bin_run(), "xtubes_tmp.dat")
    except Exception as e:
        raise e


#
# inpro
#

def xoppy_calc_inpro(CRYSTAL_MATERIAL=0,MODE=0,ENERGY=8000.0,MILLER_INDEX_H=1,MILLER_INDEX_K=1,MILLER_INDEX_L=1,\
                      ASYMMETRY_ANGLE=0.0,THICKNESS=500.0,TEMPERATURE=300.0,NPOINTS=100,SCALE=0,XFROM=-50.0,XTO=50.0):
    print("Inside xoppy_calc_xinpro. ")

    try:
        with open("xoppy.inp", "wt") as f:
            f.write("%s\n"% (os.path.join(locations.home_data(), "inpro" + os.sep)))
            if MODE == 0:
                f.write("+1\n")
            elif MODE == 1:
                f.write("-1\n")
            elif MODE == 2:
                f.write("+2\n")
            elif MODE == 3:
                f.write("-1\n")
            else:
                f.write("ERROR!!\n")

            f.write("%f\n%d\n"%(THICKNESS,CRYSTAL_MATERIAL+1))
            f.write("%s\n%f\n"%("EV",ENERGY))
            f.write("%d\n%d\n%d\n"%(MILLER_INDEX_H,MILLER_INDEX_K,MILLER_INDEX_L))
            f.write("%f\n%f\n%s\n"%(ASYMMETRY_ANGLE,TEMPERATURE, "inpro.dat"))
            if SCALE == 0:
                f.write("1\n")
            else:
                f.write("%d\n%f\n%f\n"%(2,XFROM,XTO))
            f.write("%d\n"%(NPOINTS))


        for file in ["inpro.par","inpro.dat","inpro.spec"]:
            try:
                os.remove(os.path.join(locations.home_bin_run(),file))
            except:
                pass


        if platform.system() == "Windows":
            command = "\"" + os.path.join(locations.home_bin(),'inpro.exe\" < xoppy.inp')
        else:
            command = "'" + os.path.join(locations.home_bin(), 'inpro') + "' < xoppy.inp"
        print("Running command '%s' in directory: %s "%(command, locations.home_bin_run()))
        print("\n--------------------------------------------------------\n")
        os.system(command)
        print("\n--------------------------------------------------------\n")

        #add SPEC header
        txt = open("inpro.dat").read()
        outFile = "inpro.spec"

        f = open(outFile,"w")
        f.write("#F inpro.spec\n")
        f.write("\n")
        f.write("#S 1 inpro results\n")
        f.write("#N 3\n")
        f.write("#L Theta-TetaB  s-polarized reflectivity  p-polarized reflectivity\n")
        f.write(txt)
        f.close()
        print("File written to disk: inpro.dat, inpro.par, inpro.spec")

        #show calculated parameters in standard output
        txt_info = open("inpro.par").read()
        for line in txt_info:
            print(line,end="")


        return outFile
    except Exception as e:
        raise e

#
# xcom
#


def xoppy_calc_xcom(NAME="Pyrex Glass",SUBSTANCE=3,DESCRIPTION="SiO2:B2O3:Na2O:Al2O3:K2O",\
                     FRACTION="0.807:0.129:0.038:0.022:0.004",GRID=1,GRIDINPUT=0,\
                     GRIDDATA="0.0804:0.2790:0.6616:1.3685:2.7541",ELEMENTOUTPUT=0):
    print("Inside xoppy_calc_xxcom. ")

    files = ["xcom.out","xcom.spec"]
    for file in files:
        try:
            os.remove(os.path.join(locations.home_bin_run(),file))
        except:
            pass

    try:
        with open("xoppy.inp","wt") as f:
            f.write(os.path.join(locations.home_data(), 'xcom')+ os.sep + "\n" )
            f.write( NAME+"\n" )
            f.write("%d\n"%(1+SUBSTANCE))
            if (1+SUBSTANCE) != 4:
                f.write( DESCRIPTION+"\n")
                if (1+SUBSTANCE) <= 2:
                    f.write("%d\n"%(1+ELEMENTOUTPUT))
            else:
                nn = DESCRIPTION.split(":")
                mm = FRACTION.split(":")
                f.write("%d\n"%( len(nn)))
                for i in range(len(nn)):
                    f.write(nn[i]+"\n")
                    f.write(mm[i]+"\n")
                f.write("1\n")
            f.write("%d\n"%(1+GRID))
            if (1+GRID) != 1:
                f.write("%d\n"%(1+GRIDINPUT))
                if (1+GRIDINPUT) == 1:
                    nn = GRIDDATA.split(":")
                    f.write("%d\n"%( len(nn)))
                    for i in nn:
                        f.write(i+"\n")
                    if (1+GRID) != 1:
                        f.write("N\n")
            f.write("%s\n" % "xcom.out")
            f.write("1\n")
            f.close()

        if platform.system() == "Windows":
            command = "\"" + os.path.join(locations.home_bin(),'xcom.exe\" < xoppy.inp')
        else:
            command = "'" + os.path.join(locations.home_bin(),'xcom') + "' < xoppy.inp"
        print("Running command '%s' in directory: %s "%(command,locations.home_bin_run()))
        print("\n--------------------------------------------------------\n")
        os.system(command)
        print("\n--------------------------------------------------------\n")
        # write spec file

        if (1+SUBSTANCE) <= 2:
            if (1+ELEMENTOUTPUT) == 1:
                titles = "Photon Energy [Mev]  Coherent scat [b/atom]  " \
                         "Incoherent scat [b/atom]  Photoel abs [b/atom]  " \
                         "Pair prod in nucl field [b/atom]  Pair prod in elec field [b/atom]  " \
                         "Tot atten with coh scat [b/atom]  Tot atten w/o coh scat [b/atom]"
            elif (1+ELEMENTOUTPUT) == 2:
                titles = "Photon Energy [Mev]  Coherent scat [b/atom]  " \
                         "Incoherent scat [b/atom]  Photoel abs [b/atom]  " \
                         "Pair prod in nucl field [b/atom]  Pair prod in elec field [b/atom]  " \
                         "Tot atten with coh scat [cm2/g]  Tot atten w/o coh scat [cm2/g]"
            elif (1+ELEMENTOUTPUT) == 3:
                titles = "Photon Energy [Mev]  Coherent scat [cm2/g]  " \
                         "Incoherent scat [cm2/g]  Photoel abs [cm2/g]  " \
                         "Pair prod in nucl field [cm2/g]  Pair prod in elec field [cm2/g]  " \
                         "Tot atten with coh scat [cm2/g]  Tot atten w/o coh scat [cm2/g]"
            else:
                titles = "Photon Energy [Mev]  Coherent scat [cm2/g]  " \
                         "Incoherent scat [cm2/g]  Photoel abs [cm2/g]  " \
                         "Pair prod in nucl field [cm2/g]  Pair prod in elec field [cm2/g]  " \
                         "Tot atten with coh scat [cm2/g]  Tot atten w/o coh scat [cm2/g]"
        else:
           titles = "Photon Energy [Mev]  Coherent scat [cm2/g]  " \
                    "Incoherent scat [cm2/g]  Photoel abs [cm2/g]  " \
                    "Pair prod in nucl field [cm2/g]  Pair prod in elec field [cm2/g]  " \
                    "Tot atten with coh scat [cm2/g]  Tot atten w/o coh scat [cm2/g]"


        txt = open("xcom.out").readlines()

        # copy to standard output
        for line in txt:
            print(line,end="")

        outFile = "xcom.spec"

        f = open(outFile, "w")

        f.write("#F xcom.spec\n")
        f.write("\n")
        f.write("#S 1 xcom results\n")
        f.write("#N 8\n")
        f.write("#L  "+titles+"\n")
        for i in txt:
            tmp = i.strip(" ")
            if tmp[0].isdigit():
               f.write(tmp)
            else:
               f.write("#UD "+tmp)
        f.close()
        print("File written to disk: xcom.spec")

        return outFile
    except Exception as e:
        raise e

#
# powder_fml
#

def xoppy_calc_powder_fml(
    FILE   = os.path.join(locations.home_data(), "cif" + os.sep + "icsd_31142_sepiolite_BraunerPreisinger.cif"),
    TITLE  = "powder pattern using crysFML",
    LAMBDA = 1.54056,
    JOB    = 0,
    U      = 0.0002,
    V      = -0.0002,
    W      = 0.012,
    X      = 0.0019,
    LS     = 1900.0,
    THMIN  = 1.0,
    STEP   = 0.05,
    THMAX  = 135.0,
    ):

    files = ["xpowder_fml.par","xpowder_fml.ref","xpowder_fml.out"]
    for file in files:
        try:
            os.remove(os.path.join(locations.home_bin_run(),file))
        except:
            pass

    with open("xoppy.inp", "wt") as f:
        f.write("%s\n" % (FILE))
        f.write("%s\n" % (TITLE))
        f.write("%g\n" % (LAMBDA))
        f.write("%s\n" % (JOB))
        f.write("%g\n" % (U))
        f.write("%g\n" % (V))
        f.write("%g\n" % (W))
        f.write("%g\n" % (X))
        f.write("%g\n" % (LS))
        f.write("%g\n" % (THMIN))
        f.write("%g\n" % (STEP))
        f.write("%s\n" % (THMAX))

    if platform.system() == "Windows":
        command = "\"" + os.path.join(locations.home_bin(),'xpowder_fml.exe\" < xoppy.inp')
    else:
        command = "'" + os.path.join(locations.home_bin(), 'xpowder_fml') + "' < xoppy.inp"
    print("Running command '%s' in directory: %s "%(command, locations.home_bin_run()))
    print("\n--------------------------------------------------------\n")
    os.system(command)
    print("\n--------------------------------------------------------\n")

    print("Files written to disk:\n    xpowder_fml.par (text output)\n    xpowder_fml.ref (reflections)\n    xpowder_fml.out (diffractogram)",)

    return files