import time

try: import xraylib
except: pass

import scipy.constants as codata

import numpy
from dabax.dabax_xraylib import DabaxXraylib

from xoppylib.scattering_functions.cross_calc import cross_calc_nist
from xoppylib.scattering_functions.f1f2_calc import f1f2_calc_nist



toangstroms = codata.h * codata.c / codata.e * 1e10

#=======================================================================================================================
#  GENERIC UTILITIES
#=======================================================================================================================
def write_spec_file(file_out,data,titles,scan_title=""):
    """
    Writes a spec-formatted file
    :param file_out: Name of output file
    :param data: the numpy.array with data (ncolumns, npoints)
    :param titles: a list with labels of the columns
    :param scan_title: a scan title
    :return:
    """
    #
    # write SPEC file for data[number_of_columns, number_of_points]
    #
    if len(titles) != data.shape[0]:
        raise Exception("Unmatched dimensions: titles: %d, data: (%d,%d)"%(len(titles),data.shape[0],data.shape[1]))

    f = open(file_out,"w")
    f.write("#F %s\n"%file_out)
    f.write("\n#S 1 %s\n"%scan_title)
    f.write("#N %d\n"%(data.shape[0]))
    f.write("#L")
    for ititle in titles:
        f.write("  %s"%(ititle))
    f.write("\n")
    for i in range(data.shape[1]):
        f.write(('%g '*(data.shape[0])+'\n')%( tuple(data[:,i]) ))
    f.close()
    print("File written to disk: %s"%file_out)

def nist_compound_list(material_constants_library=None):
    if material_constants_library is None:
        try:    material_constants_library = xraylib
        except: material_constants_library = DabaxXraylib()
    return material_constants_library.GetCompoundDataNISTList()

def density(descriptor, material_constants_library=None):
    if material_constants_library is None:
        try:    material_constants_library = xraylib
        except: material_constants_library = DabaxXraylib()

    kind = descriptor_kind_index(descriptor, material_constants_library=material_constants_library)
    if kind == 0:
        return density_element(descriptor, material_constants_library=material_constants_library)
    elif kind == 1:
        raise Exception("cannot retrieve density for a compound (%s): it must be defined by user" % descriptor)
    elif kind == 2:
        return density_nist(descriptor, material_constants_library=material_constants_library)
    else:
        raise Exception("Unknown descriptor: %s" % descriptor)

def density_element(DESCRIPTOR, verbose=False, material_constants_library=None):
    if material_constants_library is None:
        try:    material_constants_library = xraylib
        except: material_constants_library = DabaxXraylib()
    if isinstance(DESCRIPTOR, str):
        Z = material_constants_library.SymbolToAtomicNumber(DESCRIPTOR)
    elif isinstance(DESCRIPTOR, int):
        Z = DESCRIPTOR
    elif isinstance(input, float):
        Z = int(DESCRIPTOR)
    else:
        raise Exception("Bad input.")
    density = material_constants_library.ElementDensity(Z)
    if verbose:
        print("Density for Z=%d: %6.3f "%(Z, density))
    return density

def density_nist(DESCRIPTOR, verbose=False, material_constants_library=None):
    if material_constants_library is None:
        try:    material_constants_library = xraylib
        except: material_constants_library = DabaxXraylib()
    if isinstance(DESCRIPTOR, str):
        Zarray = material_constants_library.GetCompoundDataNISTByName(DESCRIPTOR)
        density = Zarray["density"]
        if verbose:
            print("Density for %s: %6.3f "%(DESCRIPTOR, density))
    elif isinstance(DESCRIPTOR, int):
        Zarray = material_constants_library.GetCompoundDataNISTByIndex(DESCRIPTOR)
        density = Zarray["density"]
        if verbose:
            print("Density for NIST compound with index %d: %6.3f " % (DESCRIPTOR, density))
    return density

def descriptor_kind_index(descriptor, material_constants_library=None):
    if material_constants_library is None:
        try:    material_constants_library = xraylib
        except: material_constants_library = DabaxXraylib()
    if not isinstance(descriptor, str):
        raise Exception("descriptor must be a string!")
    out = -1
    if len(descriptor) <=2:
        Z = material_constants_library.SymbolToAtomicNumber(descriptor)
        if Z > 0:
            return 0
    if descriptor in nist_compound_list():
        return 2
    try:
        material_constants_library.CompoundParser(descriptor)
        return 1
    except:
        pass
    return out


def parse_formula(formula, material_constants_library=None): # included now in xraylib, so not used but kept for other possible uses
    """

    :param formula: a formule (e.g. H2O)
    :return: a dictionary with tags: "Symbols","Elements","n","atomicWeight","massFractions","molecularWeight"

    """
    import re
    if material_constants_library is None:
        try:    material_constants_library = xraylib
        except: material_constants_library = DabaxXraylib()
    tmp = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    elements = []
    fatomic = []
    atomic_weight = []
    zetas = []
    massFractions = []
    for element,str_number in tmp:
        if str_number == '':
            number = 1
        else:
            number = int(str_number)

        elements.append(element)
        fatomic.append(number)
        zetas.append(material_constants_library.SymbolToAtomicNumber(element))
        atomic_weight.append(material_constants_library.AtomicWeight(material_constants_library.SymbolToAtomicNumber(element)))
        massFractions.append(number*material_constants_library.AtomicWeight(material_constants_library.SymbolToAtomicNumber(element)))

    mweight = 0.0
    for i in range(len(fatomic)):
        mweight += atomic_weight[i] * fatomic[i]
    print("Molecular weight: ",mweight)

    for i in range(len(massFractions)):
        massFractions[i] /= mweight

    return {"Symbols":elements,"Elements":zetas,"n":fatomic,"atomicWeight":atomic_weight,"massFractions":massFractions,"molecularWeight":mweight}

#
# bind some xraylib functions to allow NIST compound as inputs
#
def Refractive_Index_Re_Extended_NIST(descriptor, energy_in_keV, density, material_constants_library=None):
    if material_constants_library is None:
        try:    material_constants_library = xraylib
        except: material_constants_library = DabaxXraylib()
    if descriptor_kind_index(descriptor, material_constants_library=material_constants_library) == 2:
        return 1.0 - f1f2_calc_nist(descriptor, energy_in_keV * 1e3, F=3, density=density, verbose=False,
                                    material_constants_library=material_constants_library)#[0]
    else:
        return material_constants_library.Refractive_Index_Re(descriptor, energy_in_keV, density)

def Refractive_Index_Im_Extended_NIST(descriptor, energy_in_keV, density, material_constants_library=None):
    if material_constants_library is None:
        try:    material_constants_library = xraylib
        except: material_constants_library = DabaxXraylib()
    if descriptor_kind_index(descriptor, material_constants_library=material_constants_library) == 2:
        cs = cross_calc_nist(descriptor, energy_in_keV * 1e3, calculate=0, unit=3, verbose=False,
                             material_constants_library=material_constants_library)#[0]
        wavelength = codata.h * codata.c / codata.e / (energy_in_keV * 1e3)
        return (wavelength * 1e2) * cs / (4 * numpy.pi)
    else:
        return material_constants_library.Refractive_Index_Im(descriptor, energy_in_keV, density)


#=======================================================================================================================
#  CRYSTAL UTILITIES
#=======================================================================================================================

#TODO: in future we will use xraylib, by now we use this.
def f0_xop(Z):
    """
    Returns the f0 value as stired in the f0_xop DABAX file:

    Parametrization of f0 (the non-dispersive part of the atomic scttering factor) vs sin(theta/lambda).
    This file contains a Waasmaier&Kirfel-like parametrization
    for the f0 data evaluated by Kissel using modified relativistic form
    factor [1]. For the fitting, model and the data error as in [2] are used.
    The parametrization was calculating by fitting the
    original data in the RTAB database [1] (also in the DABAX file
    f0_mf_Kissel.dat) using the MPFIT IDL procedure of Markwardt
    (http://cow.physics.wisc.edu/~craigm/idl/idl.html) usind the
    function [2]:
       f0[k] = c + [SUM a_i*EXP(-b_i*(k^2)) ]
                   i=1,5
     where k = sin(theta) / lambda and c, a_i and b_i
     are the coefficients tabulated in this file (in columns:
     a1  a2  a3  a4  a5  c  b1  b2  b3  b4  b5

    References
    [1] L. Kissel, Radiation physics and chemistry 59 (2000) 185-200
        available at:
        http://www-phys.llnl.gov/Research/scattering/RTAB.html
    [2] D. Waasmaier & A. Kirfel (Acta Cryst. (1995). A51, 416-413)


     Remarks:
     1) This file contains data for only neutral atoms. For ionic states
        other DABAX files (like f0_WaasKirf.dat) could be used.
     2) The coefficients in this file are prepared for xop2.1 and are
        unpublished.
     3) We created this file for being used as a default for XOP2.1. The
        reasons were two:
        i) It is more practical to use a parametrization rather than to use
           the direct data in file f0_mf.dat because of spped and storage
           reasons.
        ii) The previous defaulty file for XOP 2.0 was f0_WaasKirf.dat, which
           contained a published parametrization [2], but the original data
           is from the International Tables for X-ray Crystallography, vol C,
           (1995) Edited by AJC Wilson, Kluwer Academic Press, table 6.1.1.3
           which contains data from multiple origins (and references). We
           prefer to use evaluated data in the same conditions for all elements
           to keep some consistency in their origin.
           Additionally, Kissel data for f1f2 and CrossSec are also used in XOP.

    Column description: a1  a2  a3  a4  a5  c  b1  b2  b3 b4  b5

    :param Z: the atomic number
    :return: the 11 coefficients for buiding f0
    """
    tmp = [
    [ 0.30426303,     0.45440815,     0.07977026,     0.15268426,     0.00871624,     0.00006937,     9.61768510,    23.52152246,     3.33237545,    53.49579285,     0.80301704],
    [ 0.79971010,     0.63926998,     0.25291139,     0.27260951,     0.03475295,     0.00044196,    10.35354478,     3.93677741,     1.38554540,    25.50380187,     0.34363759],
    [ 0.93188799,     0.17034671,     0.71602201,     0.35839485,     0.82020547,     0.00260460,     4.17494209,     0.33818181,    85.52950781,   175.66836887,     1.37984909],
    [ 1.35769775,     0.71952631,     0.76849159,     0.14970777,     0.99985829,     0.00347440,    37.89541686,     0.66643295,    92.11060309,     0.17800503,     1.90321664],
    [ 2.11021585,     0.94826030,     1.03175074,     0.17991800,     0.72039282,     0.00538888,    21.36228681,     1.17425000,    65.42872639,     0.12888999,     0.44259026],
    [ 2.75491071,     1.06083859,     1.39000885,   -61.38969569,     0.72824752,    61.44921510,    13.55853062,     0.76518509,    43.40152845,    -0.00002249,     0.23756415],
    [ 0.59731711,     2.28313328,     0.63580984,     2.26859990,     1.18920099,     0.02241857,     0.13665445,    17.86124963,    47.32019642,     7.67047709,     0.49262938],
    [ 2.85010001,     2.54410664,     0.64919585,     0.79765894,     1.12558170,     0.02853916,    12.89518279,     5.45538842,     0.11296751,    35.32893278,     0.38721225],
    [ 3.40542650,     2.84112029,     0.71156027,     0.94857450,     1.05115146,     0.03584797,     9.84065316,     4.07801707,     0.09701872,    27.57669889,     0.31753486],
    [ 3.93418627,     3.17292749,     0.78988890,     1.08878497,     0.96122888,     0.04476065,     7.82568355,     3.17536738,     0.08620371,    22.24803408,     0.27139624],
    [ 4.67888021,     3.31246434,     1.20302749,     1.11030781,     0.60080617,     0.07903453,     2.97160025,     8.01401365,     0.09917076,   119.87596938,     0.36875620],
    [ 4.85834741,     1.50402245,     1.53555848,     0.95416075,     3.00404753,     0.13060425,     4.35307904,    95.26343316,     0.11044872,    31.99246407,     1.75670443],
    [ 4.68669359,     2.38879059,     1.52287056,     1.07903978,     3.16354871,     0.14361992,     3.40015051,    36.83643520,     0.09557493,   114.27524168,     1.47492947],
    [ 4.98816795,     3.35710271,     1.50292204,     1.22172882,     2.76143663,     0.15142442,     2.53600438,    29.97580504,     0.08254945,    88.73513838,     1.16712390],
    [ 2.10509260,     4.36716206,     1.48028761,     1.33409382,     5.53780604,     0.15604602,     0.89460753,    24.41881968,     0.07145192,    69.85860169,     1.90671824],
    [ 6.20239600,     5.39538819,     1.45887725,     1.42806579,     1.33894393,     0.15420384,     1.46090299,    20.14653917,     0.06164271,    56.73801104,     0.64755460],
    [ 1.42704046,     6.77857375,     6.41364632,     1.52938606,     0.68790919,     0.13835865,     0.05185139,     1.14744676,    16.83234374,    47.08267330,     0.39585031],
    [ 7.12496467,     6.96362577,     0.40535492,     1.69961663,     1.50025574,     0.27745601,     0.91908882,    14.15475105,    14.15474574,    39.07709506,     0.06417165],
    [ 8.37197122,     7.07989232,     1.01347196,     0.77223727,     1.46667316,     0.26287737,    11.89949237,     0.78056461,   198.44072308,    41.56210501,     0.05445260],
    [ 8.64392994,     1.46357896,     1.56066861,     1.05545005,     7.04009562,     0.19999674,     9.69070226,     0.04305143,    72.04776233,   166.55421240,     0.66621783],
    [ 1.68383652,     1.47480433,     1.37047425,     9.24659290,     7.02207943,     0.16099853,    49.92976638,     0.03653814,   134.31889317,     8.45519757,     0.58503649],
    [ 9.88746586,     1.50747326,     1.90635535,     1.54310395,     7.00376164,     0.10558296,     7.44889667,     0.03084452,    38.16657035,   115.68708856,     0.51848158],
    [10.51693874,     1.57393530,     2.16699802,     1.69193955,     6.98396214,     0.01430187,     6.58802517,     0.02534021,    30.05889048,   101.44514004,     0.46207981],
    [11.07627441,     1.58619669,     2.93288168,     1.37491086,     6.96664643,     0.00459082,     5.94437112,     0.02328655,    20.59115302,    96.88722975,     0.41908215],
    [11.73712145,     1.82995875,     2.81141301,     1.90065220,     6.94066322,    -0.28355091,     5.23367741,     0.01636240,    20.53719746,    82.01631881,     0.37401385],
    [12.30434401,     2.32437385,     3.18807287,     1.99400470,     6.92612145,    -0.80737786,     4.69095987,     0.01074597,    17.39159075,    74.61601310,     0.33856083],
    [12.87035966,     4.43881362,     3.58294665,     2.06698965,     6.91586723,    -2.95253425,     4.22745007,     0.00461408,    15.08330800,    68.44820515,     0.30788472],
    [13.40782101,     6.90042218,     3.99539727,     2.14745733,   283.90205482,  -282.43818444,     3.82465894,     0.28165040,    13.17017325,    62.81493664,     0.00006082],
    [13.75399998,   297.69542224,     5.21908965,     1.58299921,     6.87528047,  -296.22007511,     3.49108135,     0.00005604,    11.26181897,    62.07734534,     0.26153431],
    [14.43242183,     6.83363443,     4.86041106,     2.31023153,   518.74776544,  -517.28462315,     3.16729215,     0.24008925,    10.34604629,    53.61746648,     0.00002943],
    [15.57306569,     6.80777698,     4.48158313,     2.55416620,   237.18610925,  -235.71998143,     2.94104939,     0.22275367,    11.54501025,    63.56972773,     0.00006123],
    [16.29213242,   271.89501093,     4.17901049,     3.17804713,     6.76929988,  -270.43761631,     2.69207459,     0.00004947,    12.10034542,    55.59088173,     0.20581071],
    [16.79861850,   336.43256975,     4.18039017,     3.72954296,     6.72535523,  -334.99650264,     2.44950901,     0.00003586,    12.83743210,    47.67503824,     0.18952317],
    [17.16712180,   336.96736021,     4.71747652,     3.89487231,     6.68380463,  -335.56546318,     2.22463287,     0.00003080,    14.03902313,    42.86632703,     0.17412989],
    [17.40231392,     5.93891344,   354.54400524,     3.52445295,     6.64817985,  -353.19765239,     2.01532010,    15.16569556,     0.00002303,    40.69766838,     0.15923958],
    [17.53267157,     7.44816522,   267.89934293,     2.98742575,     6.61999042,  -266.63403399,     1.82191497,    15.41761348,     0.00002029,    39.30642110,     0.14476941],
    [ 8.87288109,     1.56926290,     6.66073618,     1.08275119,    17.57340057,     1.08665457,    14.34390518,    35.97531308,     0.12903646,   209.51615220,     1.64279492],
    [17.59670810,     9.94862630,     6.04462400,     2.61652635,     0.54472165,     1.07329335,     1.49577834,    13.12002382,     0.11972149,   119.44032856,     0.11972149],
    [17.65536751,    10.43931521,     5.65885161,     3.13610892,     0.86203672,     1.06017183,     1.36826402,    12.12449612,     0.11133571,    95.69676914,     0.11133576],
    [17.70896284,    11.09168541,     5.76424243,     3.49865793,     0.68990337,     1.04561396,     1.25477542,    11.37544137,     0.10372418,    80.89514696,     0.10372414],
    [17.82410279,    12.34652807,     4.93750714,     3.19055176,     1.46016577,     1.02630932,     1.15717637,    11.35742646,     0.09662928,    69.39554321,     0.09662930],
    [ 6.17787753,    17.84044504,    13.24473559,     3.35857771,     0.15154195,     0.99960433,     0.08963202,     1.06333880,    10.67451656,    61.58100361,     0.08963181],
    [17.74278282,     3.79497777,     0.79808174,    13.24774635,     6.24221603,     0.95794137,     0.97113299,    42.86390771,   132.10191226,     9.26979654,     0.08236923],
    [ 6.18155275,    17.80169136,    14.63083035,     3.59439161,     0.64428903,     0.91877951,     0.07624173,     0.89717697,     9.12737711,    36.30062507,   130.83719947],
    [ 6.12780040,    17.79870560,     3.70551025,     0.61262679,    15.64710027,     0.86745289,     0.07028489,     0.82885382,    33.92215621,   126.42588650,     8.60698534],
    [ 6.09085570,     4.68870690,    17.15641530,     3.84300712,    13.15123080,     0.80254284,     0.06457257,     0.76863619,     8.38997512,    34.11521928,     0.76863617],
    [ 6.05724103,    17.76734177,     3.85910747,     0.54976811,    17.79132902,     0.70865971,     0.05857511,     7.66424959,    30.20483695,   121.19639558,     0.70979037],
    [ 6.04036427,    18.38160827,     4.20316493,     0.75471621,    17.74206099,     0.59841397,     0.05297111,     7.01093020,    30.72014689,   106.80764634,     0.65631719],
    [ 6.06154651,    19.00060821,     4.45915941,     1.04270193,    17.69494384,     0.44547768,     0.04732772,     6.43011640,    31.38897338,   129.84310238,     0.60802939],
    [19.44406366,     6.14105459,     5.04240100,     1.19460095,    17.63952092,     0.22942056,     5.87081521,     0.04153040,    32.37864163,   114.94916342,     0.56396218],
    [ 6.01179028,     6.25268840,    19.69743565,     1.16859113,    17.56401079,    -0.01748380,    32.17182048,     0.03634623,     5.33340196,   103.27699871,     0.52334570],
    [ 7.27011998,     6.55126218,    19.86580814,     0.95229359,    17.52010619,    -0.49670130,    31.27680875,     0.02990172,     4.84670590,   100.88905997,     0.48406725],
    [19.93807829,     7.09598483,     8.55254548,     0.81322939,    17.46003963,    -1.21189020,     4.40199492,     0.02378768,    29.37558678,    96.37033204,     0.44869824],
    [19.95850428,     8.36420076,     9.90402346,     0.65245953,    17.40931309,    -2.65593783,     4.00466253,     0.01705338,    27.35376466,    96.62465241,     0.41685323],
    [17.37569453,    12.13586009,    10.55273124,     1.29778936,    19.87680486,    -6.62800954,     0.38664179,     0.00966981,    23.44864968,   220.56960899,     3.64363491],
    [19.82277247,    17.37211521,    10.64912854,     2.42340529,    51.54683902,   -46.21974320,     3.34809910,     0.36080812,    20.04789088,   161.05858343,     0.00185019],
    [19.96684466,  2616.77020357,    11.08605976,     2.97061929,    17.30250838, -2611.52103692,     3.11105960,     0.00003320,    18.89301922,   126.40106401,     0.33961309],
    [17.29595781,  4345.36340933,    20.94755728,     2.49690482,    11.57180940, -4340.11668462,     0.32551631,     0.00001928,     3.04729063,   146.75357391,    17.76655273],
    [21.56386679,    17.23892858,    11.96705606,     2.55058606,  4645.78355064, -4640.56446888,     2.91250166,     0.30985101,    16.75073376,   139.14334527,     0.00001725],
    [17.18849734,  5256.12632558,    12.33127623,     2.58243269,    22.21683582, -5250.92543157,     0.29580553,     0.00001461,    15.88877581,   133.41685838,     2.79253798],
    [17.14790054,  5467.09133967,    12.67590249,     2.61445255,    22.89314617, -5461.92264966,     0.28225651,     0.00001335,    15.09888908,   128.17517281,     2.67966474],
    [23.57776892,  4336.95334631,    13.01186004,     2.65112019,    17.09246464, -4331.80653530,     2.57205344,     0.00001606,    14.35914195,   123.21157749,     0.26967441],
    [17.07135563,  5126.49069186,    13.33466886,     2.65426567,    24.27237487, -5121.36316596,     0.25889588,     0.00001297,    13.73081795,   120.24251547,     2.47764087],
    [24.64625088,    16.99747848,    13.36466048,     3.34514581,  4543.57338365, -4538.49039355,     2.34262325,     0.24679775,    12.90194424,    95.39774603,     0.00001379],
    [25.66612063,  4192.75632504,    13.95736847,     2.72815093,    17.00044486, -4187.69180909,     2.29102865,     0.00001424,    12.50615875,   111.05763849,     0.23787894],
    [26.38323252,  4398.24494424,    14.26736529,     2.76532287,    16.97289395, -4393.23947589,     2.20308248,     0.00001260,    11.95434176,   106.89618428,     0.22734974],
    [27.09405383,    16.93749917,    14.58634079,     2.80145206,  3698.14032223, -3693.18846082,     2.11867358,     0.21749694,    11.43043354,   102.92181598,     0.00001396],
    [27.81053249,  4104.64353871,    14.88325935,     2.83483528,    16.89331430, -4099.71721767,     2.04181579,     0.00001196,    10.95422544,    99.42542607,     0.20889623],
    [28.52850085,  2445.57145486,    15.22240776,     2.86116529,    16.92020688, -2440.77861007,     1.96383816,     0.00001716,    10.48926663,    96.46598501,     0.19884908],
    [29.24441972,  2212.27369164,    15.49510957,     2.92480251,    16.89099012, -2207.52991878,     1.89336193,     0.00001759,    10.05627636,    91.69307079,     0.19084871],
    [29.69788948,    15.47177518,  2097.86718749,     3.60694158,    16.82362572, -2093.19263162,     1.80307602,     9.47929223,     0.00001698,    77.64749351,     0.18222067],
    [30.11062139,    15.40407022,  2274.03155954,     4.38097094,    16.78436412, -2269.46340922,     1.71565360,     8.96253008,     0.00001361,    63.95107044,     0.17352112],
    [30.57052118,    15.57376402,  1353.88450796,     4.87958160,    16.74532612, -1349.42935153,     1.63430749,     8.68104102,     0.00001949,    56.83069217,     0.16513675],
    [30.98784842,    15.84365824,  1085.37020439,     5.34491722,    16.75914099, -1081.10872457,     1.55377182,     8.43294944,     0.00001745,    50.66121806,     0.15616127],
    [31.36764074,    16.20662904,  1140.89802522,     5.70714498,    16.65069139, -1136.66620413,     1.48081504,     8.26474305,     0.00001603,    45.63908603,     0.14966164],
    [31.75193454,    16.78095382,   410.94969397,     5.94974979,    16.75723964,  -407.05071307,     1.40570152,     8.12076847,     0.00001747,    42.41784435,     0.13977093],
    [31.76739790,     1.41742145,    16.74659192,    16.21969187,     6.34032022,     3.65295339,     1.32218506,    81.15672344,     0.13119994,     7.35439190,    26.57488749],
    [30.98992203,    18.15501717,    16.67920729,     6.30866056,     1.30977395,     3.63949590,     1.27222945,     7.83558995,     0.12654931,    36.86734925,     1.27222941],
    [16.88254232,    19.37581635,    32.62146048,     6.01941144,   432.23434409,  -429.07568590,     0.11690541,     7.87648778,     1.20474478,    32.97062267,    -0.00003099],
    [16.43979396,    19.91863890,    27.76463255,     6.41022928,     4.90481781,     3.58984976,     0.11673200,     7.55331808,     1.15426441,    33.30365931,     1.15426437],
    [16.27949672,    19.61124539,    32.52808287,     1.21893192,     6.82864702,     3.56321128,     0.11193237,     6.83979837,     1.09072759,   118.61695437,    25.09178834],
    [16.15573594,    32.56471040,     6.74799686,     1.62318006,    20.37867502,     3.53108830,     0.10731472,     1.03828673,    25.84002437,   104.03280242,     6.51727715],
    [16.05513439,    32.57219633,     7.10079270,     1.82264739,    20.92330655,     3.49908832,     0.10310727,     0.99041508,    26.79675111,    92.39516651,     6.16859916],
    [15.92966393,    32.53463621,    21.26597071,     1.85348130,     7.90481807,     3.45521535,     0.09871917,     0.94233879,     5.77485734,    84.25758041,    27.14259799],
    [15.82334418,    32.46748196,     9.01250570,     1.72434540,    21.46356994,     3.42272786,     0.09486320,     0.89843768,    27.00385753,    79.63992570,     5.39445822],
    [15.71714768,    32.36554794,    21.55254018,     1.54577949,    10.31957180,     3.38282129,     0.09111052,     0.85619893,     5.01523482,    76.65511543,    26.25222808],
    [15.61356578,    32.29005100,    21.58808798,     1.52079712,    11.49239752,     3.33918799,     0.08744016,     0.81690795,     4.68491604,   187.13075688,    24.42129865],
    [32.26137613,    21.47411433,    11.54875240,     2.70070705,    15.53353668,     3.29342205,     0.78242755,     4.39338862,    21.26815521,   145.78453108,     0.08389021],
    [15.44129352,    32.19461688,    21.67018557,    11.62375491,     3.60839956,     3.23867141,     0.08037991,     0.74781467,     4.14033299,    19.84028498,   113.68244239],
    [15.35041360,    32.15264867,    21.98489988,     4.20304734,    11.87984118,     3.17150887,     0.07685215,     0.71518502,     3.92354148,    96.69516744,    19.25785705],
    [32.41734383,    22.05085313,    13.10871855,     3.72910194,    15.28123528,     3.11865061,     0.68922416,     3.86872813,    17.74443730,   101.90118122,     0.07385957],
    [15.20314929,    32.53208188,    13.81186567,     3.76642798,    22.31042147,     3.04559348,     0.07066435,     0.66171756,    16.79213936,    97.72588956,     3.75457984],
    [32.64327679,    22.60430101,    14.43752068,     3.83052598,    15.12931302,     2.98794434,     0.63635924,     3.65636186,    15.88803466,    93.50959958,     0.06783444],
    [32.89114822,    23.09839499,    15.57640146,     3.06849154,    15.06334425,     2.89811885,     0.61290312,     3.63716663,    15.44356030,   103.61518316,     0.06473308],
    [33.02310917,    23.59414755,    16.08710719,     3.05160429,    15.00738866,     2.79583742,     0.58946182,     3.56398355,    14.80962925,   101.26355106,     0.06167575],
    [14.92952838,    33.03254055,    24.00228529,     3.79349958,    16.07933216,     2.68191187,     0.05858286,     0.56470268,     3.42619535,    86.81878918,    13.93081918],
    [14.87484815,    33.25429616,    24.75369621,     3.05890997,    16.98364667,     2.55564102,     0.05561473,     0.54346670,     3.41941648,    95.53275618,    13.62900417],
    [33.35395184,    25.38419399,    17.37894160,     3.08843097,    14.82268135,     2.41071058,     0.52171734,     3.34704782,    13.07655187,    91.79465535,     0.05263332],
    ]
    tmp = numpy.array(tmp)
    return tmp[Z-1].copy()


if __name__ == "__main__":
    pass
    #
    # parsers (not used, we use xraylib instead)
    #

    # aw1 = parse_formula("B4C")
    # print(aw1)

    # tmp = write_spec_file("tmp.spec",numpy.zeros((4,100)),titles=['x','y','z','more z'])



    #
    # f1f2_calc*
    #
    # for i in range(12):
    #     tmp = f1f2_calc(26,[10000.0,22000],F=i)
    #     print(">>>>>>>>>>>>>>F=%d, Z=26,f1f2_calc="%i,tmp,tmp.shape )
    #
    # for i in range(12):
    #     # print(">>>>>>>>>>>>>>F=%d, f1f2_calc_mix="%i, f1f2_calc_mix("SiC",[10000.0],F=i,density=3.21))
    # for i in range(12):
    #     tmp1 = f1f2_calc_mix("H2O",[10000.0],F=i,density=1.0)
    #     tmp2 = f1f2_calc_nist("Water, Liquid", [10000.0], F=i)
    #     print(">>>>>>>>>>>>>>F=%d, f1f2_calc_mix for H2O. f1f2_calc_nist for Water: ** %g, %g **" % (i, tmp1[0], tmp2[0]) )


    # assert( numpy.abs( tmp1[0] - 8.7654751491065230e-07) < 1e-9  )



    # #
    # # cross_calc*
    # #
    #
    # for unit in range(4):
    #     for calculate in range(5):
    #         print(">>>>>>>>>>>>>>unit=%d, calculate=%d, Z=26,result="%(unit,calculate), cross_calc(26,10000.0,unit=unit,calculate=calculate))
    #
    # for i in range(4):
    #     print(">>>>>>>>>>>>>>unit=%d, H2O,result="%i, cross_calc_mix("H2O",10000.0,unit=i,density=1.0,parse_or_nist=0))
    #
    # for i in range(4):
    #     print(">>>>>>>>>>>>>>unit=%d, Acetone,result="%i, cross_calc_mix("Acetone",10000.0,unit=i,density=None,parse_or_nist=1))
    #
    # #
    # # XPOWER
    # #
    #
    # energies = numpy.linspace(1000.0,50000.0,100)
    # source = numpy.ones_like(energies)
    # fileOut = xpower_calc(energies,source,substance=["Be","Rh"],flags=[0,1],dens=["?","?"],thick=[0.5,0.0],angle=[0.0,3.0],roughness=[0.0,0.0],output_file="xpower.spec")
    #
    #
    #
    #
    # #
    # # crystal tools
    # #
    #
    # print("f0 coefficisnts for Z=30",f0_xop(30))
    #
    # print("Bragg metric tensor for Si: ")
    # print("   d-spacing for 111: ",bragg_metrictensor(5.43,5.43,5.43,90.0,90.0,90.0,HKL=[1,1,1],RETURN_REAL_SPACE=0,RETURN_VOLUME=0))
    # print("   Volume of unit cell: ",bragg_metrictensor(5.43,5.43,5.43,90.0,90.0,90.0,RETURN_VOLUME=1))
    # print("   metric tensor in real spc: ",bragg_metrictensor(5.43,5.43,5.43,90.0,90.0,90.0,RETURN_REAL_SPACE=1))
    # print("   metric tensor in reciprocal spc: ",bragg_metrictensor(5.43,5.43,5.43,90.0,90.0,90.0))
    #
    # print("LORENTZ(5.039)",lorentz(10))
    #
    #
    # dic1 = bragg_calc(descriptor="Si",hh=1,kk=1,ll=1,temper=1.0,emin=7900.0,emax=8100.0,estep=5.0,fileout="bragg-tmp.dat")
    # print("KEYS: ",dic1.keys())
    # print(dic1)
    #
    # #
    # dic2 = crystal_fh(dic1,8000.0)
    #
    # print(dic2["info"])
    # print("KEYS: ",dic2.keys())
    #
    #
    # #
    # # MARE
    # #
    # # # descriptor,H,K,L,HMAX,KMAX,LMAX,FHEDGE,DISPLAY,lambda1,deltalambda,PHI,DELTAPHI,verbose=1)
    # list_of_scripts = mare_calc("Si2",2,2,2,3,3,3,2e-8,3,1.54,0.01,-20.0,0.1)
    # for script in list_of_scripts:
    #     exec(script)


    # a = f0_calc(1,"H2O",0,6,100)
    # b = f0_calc(2, "Water, Liquid", 0, 6, 100)
    # from srxraylib.plot.gol import plot
    # plot(a["data"][0,:],a["data"][1,:],
    #      b["data"][0,:],b["data"][1,:])

    # print(density_element("Fe", verbose=True), density_element(26, verbose=True))
    # print(density_nist("Water, Liquid", verbose=True),density_nist(177, verbose=True))

    # a = cross_calc_mix("H2O",1000.0,0,density=1.0)
    # b = cross_calc_nist("Water, Liquid", 1000.0, 0)
    # print("\n\n\nCross section :",a, b)

    # descriptors = ["Be","BB","CCC","BeO","Water, Liquid", "jdfjys","C12H22O6",'k',"K","KK","Water, Liquidd"]
    # out = []
    # for descriptor in descriptors:
    #     out.append(descriptor_kind_index(descriptor))
    # for i in range(len(descriptors)):
    #     print(descriptors[i], out[i])

    # for descriptor in ["Be","B","K","Water, Liquid","KK"]:
    #     print("%s density: %f" % (descriptor, density(descriptor)))

    # list1 = nist_compound_list()
    # for el in list1:
    #     print(el)

    # print(Refractive_Index_Re("H2O", 12.4 , 1.0), Refractive_Index_Re("Water, Liquid", 12.4, 1.0))
    # print(Refractive_Index_Im("H2O", 12.4 , 1.0), Refractive_Index_Im("Water, Liquid", 12.4, 1.0))

    # dic2a = bragg_calc(descriptor="Muscovite", hh=1, kk=1, ll=1, temper=1.0, emin=7900.0, emax=8100.0, estep=5.0)
    # dic2b = crystal_fh(dic2a, 8000.0)
    # assert ( numpy.abs(dic2b["F_0"].real - 801.722) < 0.1)
    # assert (numpy.abs(dic2b["F_0"].imag - 13.08712) < 0.011)

