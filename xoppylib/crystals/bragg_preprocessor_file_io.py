"""
I/O utilities for Bragg crystal preprocessor files (read/write version 1 and 2).
"""

import numpy


# todo: return txt
def bragg_preprocessor_file_v1_write(out_dict, fileout=""):
    if fileout != "":
        f = open(fileout, 'wt')

        f.write("%i " % out_dict["i_latt"])  # flag ZincBlende
        # f.write("%e " % ((1e0 / volume_in_cm3) * (codata_e2_mc2 * 1e2))) # 1/V*electronRadius
        f.write("%e " % (out_dict["one_over_volume_times_electron_radius_in_cm"]))
        f.write("%e " % out_dict["dspacing_in_cm"])
        f.write("\n")
        f.write("%i " % out_dict["zeta_a"])
        f.write("%i " % out_dict["zeta_b"])
        f.write("%e " % out_dict["temper"])  # temperature parameter
        f.write("\n")
        f.write("(%20.11e,%20.11e ) \n" % (out_dict["ga.real"], out_dict["ga.imag"]))
        f.write("(%20.11e,%20.11e ) \n" % (out_dict["ga_bar.real"], out_dict["ga_bar.imag"]))
        f.write("(%20.11e,%20.11e ) \n" % (out_dict["gb.real"], out_dict["gb.imag"]))
        f.write("(%20.11e,%20.11e ) \n" % (out_dict["gb_bar.real"], out_dict["gb_bar.imag"]))
        f.write("%e %e %e  \n" % (out_dict["fit_a"][0], out_dict["fit_a"][1], out_dict["fit_a"][2]  ))
        f.write("%e %e %e  \n" % (out_dict["fit_b"][0], out_dict["fit_b"][1], out_dict["fit_b"][2]  ))
        f.write(("%i \n") % out_dict["npoint"])
        for i in range(out_dict["npoint"]):
            f.write(("%20.11e %20.11e %20.11e \n %20.11e %20.11e \n") % ( \
                out_dict["Energy"][i],
                out_dict["F1a"][i],
                out_dict["F2a"][i],
                out_dict["F1b"][i],
                out_dict["F2b"][i]
            ))
        f.close()
        print("File written to disk: %s" % fileout)


def bragg_preprocessor_file_v2_write(output_dictionary, fileout=None):

    txt = ""
    txt += "# Bragg version, Data file type\n"
    txt += "%s 1\n" % output_dictionary["version"]
    txt += "# RN = (e^2/(m c^2))/V) [cm^-2], d spacing [cm]\n"
    txt += "%e %e \n" % (output_dictionary["rn"] , output_dictionary["dspacing"])

    txt += "# Number of different element-sites in unit cell NBATOM:\n%d \n" % output_dictionary["nbatom"]


    txt += "# for each element-site, the number of scattering electrons (Z_i + charge_i)\n"
    for i in output_dictionary['atnum']:
        txt += "%g "%i
    txt += "\n"

    txt += "# for each element-site, the occupation factor\n"
    for i in output_dictionary["fraction"]:
        txt += "%g "%i
    txt += "\n"

    txt += "# for each element-site, the temperature factor\n" # temperature parameter
    for i in output_dictionary["temper"]:
        txt += "%g "%i
    txt += "\n"

    #
    # Geometrical part of structure factor:  G and G_BAR
    #
    txt += "# for each type of element-site, COOR_NR=G_0\n"


    for i in output_dictionary["G_0"]:
        txt += "%g "%i
    txt += "\n"

    #
    txt += "# for each type of element-site, G and G_BAR (both complex)\n"

    for i,ga in enumerate(output_dictionary["G"]):
        ga_bar = output_dictionary["G_BAR"][i]
        txt += "(%g,%g) \n"%(ga.real,ga.imag)
        txt += "(%g,%g) \n"%(ga_bar.real,ga_bar.imag)

    #
    # F0 part
    #
    txt += "# for each type of element-site, the number of f0 coefficients followed by them\n"
    for i in range(len(output_dictionary['f0coeff'])):
        coeff = output_dictionary['f0coeff'][i]
        nn = len(coeff)
        txt += ("%d "%(nn)+"%g "*nn+"\n")%(tuple(coeff))


    txt += "# The number of energy points NPOINT: \n"
    txt +=  ("%i \n") % output_dictionary["npoint"]
    txt += "# for each energy point, energy, F1(1),F2(1),...,F1(nbatom),F2(nbatom)\n"

    for i in range(output_dictionary["npoint"]):
        txt += ("%20.11e \n") % (output_dictionary["energy"][i])

        for j in range(output_dictionary['nbatom']):
            f1a = output_dictionary['f1'][j,i]
            f2a = output_dictionary['f2'][j,i]
            fcompton = output_dictionary['fcompton'][j,i]
            txt +=  (" %20.11e %20.11e %20.11e \n")%(f1a, f2a, fcompton)


    if fileout != None:
        with open(fileout,"w") as f:
            f.write(txt)
            print("File written to disk: %s" % fileout)

    return txt

def bragg_preprocessor_file_v1_read(filename):

    try:
        f = open(filename, 'r')
        lines = f.read().splitlines()
        f.close()
    except:
        print("Fail to read file: %s " %  filename)
        raise Exception("Fail to read file: %s " %  filename)
    out_dict = {}


    line_index = 0
    variables = __parse_line(lines[line_index])
    out_dict["i_latt"] = int(variables[0])
    out_dict["one_over_volume_times_electron_radius_in_cm"] = float(variables[1])
    out_dict["dspacing_in_cm"] = float(variables[2])

    line_index += 1
    variables = __parse_line(lines[line_index])
    out_dict["zeta_a"] = int(variables[0])
    out_dict["zeta_b"] = int(variables[1])
    out_dict["temper"] = float(variables[2])

    # line_index += 1

    line_index += 1
    variables = __parse_line(lines[line_index], remove=["(",")",","])
    out_dict["ga.real"] = float(variables[0])
    out_dict["ga.imag"] = float(variables[1])

    line_index += 1
    # print(">>>>>>>>>> variables", variables)
    variables = __parse_line(lines[line_index], remove=["(", ")", ","])
    out_dict["ga_bar.real"] = float(variables[0])
    out_dict["ga_bar.imag"] = float(variables[1])

    line_index += 1
    # print(">>>>>>>>>> variables", variables)
    variables = __parse_line(lines[line_index], remove=["(", ")", ","])
    out_dict["gb.real"] = float(variables[0])
    out_dict["gb.imag"] = float(variables[1])

    line_index += 1
    line = lines[line_index]
    line = line.replace("(", "")
    line = line.replace(")", "")
    line = line.replace(" ", "")
    variables = line.split(",")
    # print(">>>>>>>>>> variables", variables)
    out_dict["gb_bar.real"] = float(variables[0])
    out_dict["gb_bar.imag"] = float(variables[1])

    line_index += 1
    variables = __parse_line(lines[line_index])
    # print(">>>>>>>>>> variables", variables)
    out_dict["fit_a"] = []
    for variable in variables:
        out_dict["fit_a"].append(float(variable))

    line_index += 1
    variables = __parse_line(lines[line_index])
    # print(">>>>>>>>>> variables", variables)
    out_dict["fit_b"] = []
    for variable in variables:
        out_dict["fit_b"].append(float(variable))

    line_index += 1
    variables = __parse_line(lines[line_index])
    npoint = int(variables[0])
    out_dict["npoint"] = npoint

    line_index += 1
    variables = __parse_line(" ".join(lines[line_index:]))


    Energy = numpy.zeros(npoint)
    F1a = numpy.zeros(npoint)
    F2a = numpy.zeros(npoint)
    F1b = numpy.zeros(npoint)
    F2b = numpy.zeros(npoint)
    iacc = -1
    for i in range(npoint):
        iacc += 1
        Energy[i] = variables[iacc]
        iacc += 1
        F1a[i] = variables[iacc]
        iacc += 1
        F2a[i] = variables[iacc]
        iacc += 1
        F1b[i] = variables[iacc]
        iacc += 1
        F2b[i] = variables[iacc]

    out_dict["Energy"] = Energy
    out_dict["F1a"] = F1a
    out_dict["F2a"] = F2a
    out_dict["F1b"] = F1b
    out_dict["F2b"] = F2b

    # not in file
    out_dict["emin"] = Energy[0]
    out_dict["emax"] = Energy[-1]
    out_dict["estep"] = Energy[1] - Energy[0]
    out_dict["fileout"] = filename
    return out_dict

def bragg_preprocessor_file_v2_read(filename=""):

    f = open(filename, 'r')
    lines = f.read().splitlines()
    f.close()

    out_dict = {}

    line_index = 2


    line_index += 1 # jump comment
    variables = __parse_line(lines[line_index])
    # print(">>>>>>>>>> variables rn dspacing", variables)
    out_dict["rn"] = float(variables[0])
    out_dict["dspacing"] = float(variables[1])

    line_index += 2 # jump comment
    variables = __parse_line(lines[line_index])
    # print(">>>>>>>>>> variables nbatom", variables)
    out_dict["nbatom"] = int(variables[0])

    line_index += 2 # jump comment
    variables = __parse_line(lines[line_index])
    # print(">>>>>>>>>> variables atnum", variables)
    out_dict["atnum"] = []
    for variable in variables:
        out_dict["atnum"].append(float(variable))


    line_index += 2 # jump comment
    variables = __parse_line(lines[line_index])
    # print(">>>>>>>>>> variables fraction", variables)
    out_dict["fraction"] = []
    for variable in variables:
        out_dict["fraction"].append(float(variable))

    line_index += 2  # jump comment
    variables = __parse_line(lines[line_index])
    # print(">>>>>>>>>> temper atnum", variables)
    out_dict["temper"] = []
    for variable in variables:
        out_dict["temper"].append(float(variable))

    line_index += 2  # jump comment
    variables = __parse_line(lines[line_index])
    # print(">>>>>>>>>> variables G_0", variables)
    out_dict["G_0"] = []
    for variable in variables:
        out_dict["G_0"].append(int(variable))

    line_index += 1
    out_dict["G"] = []
    out_dict["G_BAR"] = []
    for i in range(out_dict["nbatom"]):
        line_index += 1
        variables_G    = __parse_line(lines[line_index], remove=["(",")",","])
        # print(">>>>>>>>>> variables G", variables_G)
        out_dict["G"].append(float(variables_G[0]) + 1j * float(variables_G[1]))
        # print(">>> G: ", out_dict["G"][-1])

        line_index += 1
        variables_GBAR = __parse_line(lines[line_index], remove=["(", ")", ","])
        # print(">>>>>>>>>> variables G_BAR", variables_GBAR)
        out_dict["G_BAR"].append(float(variables_GBAR[0]) + 1j * float(variables_GBAR[1]))
        # print(">>> G_BAR: ", out_dict["G_BAR"][-1])


    line_index += 1  # jump comment
    out_dict["f0coeff"] = []
    for i in range(out_dict["nbatom"]):
        f0_ilist = []
        line_index += 1
        variables = __parse_line(lines[line_index])
        # print(">>>>>>>>>> variables f0", variables)
        for i in range(int(variables[0])):
            f0_ilist.append(float(variables[i+1]))
        # print(">>>> f0coeff[i]", f0_ilist)
        out_dict["f0coeff"].append((f0_ilist))

    line_index += 2
    variables = __parse_line(lines[line_index])
    # print(">>>>>>>>>> variables npoint", variables)
    npoint = int(variables[0])
    out_dict["npoint"] = npoint


    line_index += 1
    Energy = numpy.zeros(npoint)
    F1a = numpy.zeros(npoint)
    F2a = numpy.zeros(npoint)
    F1b = numpy.zeros(npoint)
    F2b = numpy.zeros(npoint)

    # for j, jj in enumerate(indices_prototypical):
    #     f1a = xraylib.Fi(list_Zatom[jj], energy * 1e-3)
    #     f2a = -xraylib.Fii(list_Zatom[jj], energy * 1e-3)  # TODO: check the sign!!
    #     txt += (" %20.11e %20.11e 1.000 \n") % (f1a, f2a)
    Energy = numpy.zeros(npoint)
    out_f1       = numpy.zeros( (out_dict["nbatom"], npoint) )
    out_f2       = numpy.zeros( (out_dict["nbatom"], npoint) )
    out_fcompton = numpy.zeros( (out_dict["nbatom"], npoint) )

    for i in range(npoint):
        line_index += 1
        variables = __parse_line(lines[line_index])
        # print(">>>>>>>>>> variables Energy[i]", variables)
        Energy[i] = float(variables[0])
        for j in range(out_dict["nbatom"]):
            line_index += 1
            variables = __parse_line(lines[line_index])
            out_f1[j,i]        = float(variables[0])
            out_f2[j,i]        = float(variables[1])
            out_fcompton[j,i]  = float(variables[2])


    out_dict["energy"] = Energy
    out_dict["f1"] = out_f1
    out_dict["f2"] = out_f2
    out_dict["fcompton"] = out_fcompton

    return out_dict


def __parse_line(line, remove=[]):
    if len(remove) > 0:
        for str1 in remove:
            line = line.replace(str1, " ")
    line = " ".join(line.split())
    variables = line.split(" ")
    return variables



