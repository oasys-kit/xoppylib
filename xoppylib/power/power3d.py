"""
3D power density calculations for synchrotron radiation sources.
"""
#
# tools for power3d app
#
import os
import numpy
import scipy.constants as codata
import h5py

from dabax.dabax_xraylib import DabaxXraylib
try: import xraylib
except: pass

from xoppylib.scattering_functions.fresnel import reflectivity_fresnel
from xoppylib.xoppy_xraylib_util import nist_compound_list, density
from xoppylib.mlayer import MLayer
from xoppylib.power.power1d_calc_monochromators import get_multilayer_instance

from scipy.interpolate import interp2d, RectBivariateSpline
from srxraylib.util.h5_simple_writer import H5SimpleWriter
#
# calculation
#
def integral_2d(data2D,h=None,v=None, method=0):
    if h is None:
        h = numpy.arange(data2D.shape[0])
    if v is None:
        v = numpy.arange(data2D.shape[1])

    if method == 0:
        try:    totPower2 = numpy.trapezoid(data2D, v, axis=1)
        except: totPower2 = numpy.trapz(data2D, v, axis=1)
        try:    totPower2 = numpy.trapezoid(totPower2, h, axis=0)
        except: totPower2 = numpy.trapz(totPower2, h, axis=0)
    else:
        totPower2 = data2D.sum() * (h[1] - h[0]) * (v[1] - v[0])

    return totPower2


def integral_3d(data3D, e=None, h=None, v=None, method=0):
    if e is None:
        e = numpy.arange(data3D.shape[0])
    if h is None:
        h = numpy.arange(data3D.shape[1])
    if v is None:
        v = numpy.arange(data3D.shape[2])

    if len(e.shape) == 1: # e is 1D: usual case
        if method == 0:
            try:
                totPower2 = numpy.trapezoid(data3D, v, axis=2)
                totPower2 = numpy.trapezoid(totPower2, h, axis=1)
                totPower2 = numpy.trapezoid(totPower2, e, axis=0)
            except:
                totPower2 = numpy.trapz(data3D, v, axis=2)
                totPower2 = numpy.trapz(totPower2, h, axis=1)
                totPower2 = numpy.trapz(totPower2, e, axis=0)
        else:
            totPower2 = data3D.sum() * (e[1] - e[0]) * (h[1] - h[0]) * (v[1] - v[0])
    else: # e has the same dimension as data3D, calculation vs harmonics
        if method == 0:

            totPower2 = numpy.trapezoid(data3D, v, axis=2)
            totPower2 = numpy.trapezoid(totPower2, h, axis=1)
            totPower2 = numpy.sum(totPower2)

        else:
            totPower2 = data3D.sum() * (h[1] - h[0]) * (v[1] - v[0])

    return totPower2


# def _evaluate_elementwise(func, x):
#     """
#     Apply a scalar-only function elementwise to x, preserving its shape.
#     x can be a scalar (0-d) or an array of any shape (e.g. an energy value
#     per (H,V) point).
#     """
#     x = numpy.asarray(x)
#     if x.ndim == 0:
#         return func(float(x))
#     return numpy.array([func(float(v)) for v in x.ravel()]).reshape(x.shape)
#

def calculate_component_absorbance_and_transmittance(
    e0, h0, v0,
    substance="Si",
    thick=0.5,
    angle=3.5,
    defection=1, # deflection 0=H 1=V
    dens="?",
    roughness=0.0,
    flags=0,  # 0=Filter 1=Mirror 2 = Aperture 3 magnifier, 4 screen rotation, 5 thin object filter, 6 multilayer, 7 External file
    hgap=1000.0,                             # used if flag in (0, 1, 2, 6, 7)
    vgap=1000.0,                             # used if flag in (0, 1, 2, 6, 7)
    hgapcenter=0.0,                          # used if flag in (0, 1, 2, 6, 7)
    vgapcenter=0.0,                          # used if flag in (0, 1, 2, 6, 7)
    gapshape=0,     # 0=rectangle, 1=circle  # used if flag in (0, 1, 2, 6, 7)
    hmag=1.0,
    vmag=1.0,
    hrot=0.0,       # used if flag in (0, 4, 7)
    vrot=0.0,       # used if flag in (0, 4, 7)
    thin_object_file='',
    thin_object_thickness_outside_file_area=0.0,
    thin_object_back_profile_flag=0,
    thin_object_back_profile_file='',
    multilayer_file='',
    external_reflectivity_file='',
    verbose=1,
    material_constants_library=None,
    ):

    if material_constants_library is None:
        try:    material_constants_library = xraylib
        except: material_constants_library = DabaxXraylib()
    #
    # important: the transmittance calculated here is referred on axes perp to the beam
    # therefore they do not include geometrical corrections for correct integral
    #

    e = e0.copy()
    h = h0.copy()
    v = v0.copy()

    transmittance = numpy.ones((e.shape[0], h.size, v.size))
    E = e.copy()
    H = h.copy()
    V = v.copy()

    # initialize results

    #
    # get undefined densities
    #
    if flags in [0,1,5]:
        try:  # apply written value
            rho = float(dens)
        except:  # in case of ?
            rho = density(substance, material_constants_library=material_constants_library)
            print("Density for %s: %g g/cm3" % (substance, rho))
        dens = rho

    txt = ""

    txt_rotation = ""
    txt_rotation += '      H rotation angle [deg]: %f \n' % (hrot)
    txt_rotation += '      V rotation angle [deg]: %f \n' % (vrot)

    txt_aperture = ""
    txt_aperture += '      H gap [mm]: %f \n' % (hgap)
    txt_aperture += '      V gap [mm]: %f \n' % (vgap)
    txt_aperture += '      H gap center [mm]: %f \n' % (hgapcenter)
    txt_aperture += '      V gap center [mm]: %f \n' % (vgapcenter)


    if flags == 0:
        txt += '      *****   oe  [Filter] *************\n'
        txt += '      Material: %s\n' % (substance)
        txt += '      Density [g/cm^3]: %f \n' % (dens)
        txt += '      thickness [mm] : %f \n' % (thick)
        txt += txt_rotation
        txt += txt_aperture
    elif flags == 1:
        txt += '      *****   oe  [Mirror] *************\n'
        txt += '      Material: %s\n' % (substance)
        txt += '      Density [g/cm^3]: %f \n' % (dens)
        txt += '      grazing angle [mrad]: %f \n' % (angle)
        txt += '      roughness [A]: %f \n' % (roughness)
        txt += txt_aperture
    elif flags == 2:
        txt += '      *****   oe  [Aperture] *************\n'
        txt += txt_aperture
    elif flags == 3:
        txt += '      *****   oe  [Magnifier] *************\n'
        txt += '      H magnification: %f \n' % (hmag)
        txt += '      V magnification: %f \n' % (vmag)
    elif flags == 4:
        txt += '      *****   oe  [Screen rotated] *************\n'
        txt += txt_rotation
    elif flags == 5:
        txt += '      *****   oe  [Thin object filter] *************\n'
        txt += '      File with thickness for front surface [h5, OASYS-format]: %s \n' % (thin_object_file)
        if thin_object_back_profile_flag == 0:
            txt += '      Back surface set to z(x,y)=0\n'
        elif thin_object_back_profile_flag == 1:
            txt += '      File with thickness for back surface [h5, OASYS-format]: %s \n' % (thin_object_back_profile_file)
    elif flags == 6:
        txt += '      *****   oe  [Multilayer] *************\n'
        txt += '      File with multilayer parameters from XOPPY/Multilayer [h5]: %s \n' % (multilayer_file)
        txt += '      grazing angle [mrad]: %f \n' % (angle)
        txt += txt_aperture
    elif flags == 7:
        txt += '      *****   oe  [External Reflectivity File] *************\n'
        txt += '      File with multilayer parameters and incident angle (energy scan) from XOPPY/Multilayer [h5]: %s \n' % (multilayer_file)
        txt += txt_rotation
        txt += txt_aperture

    if flags == 0:  # filter
        if len(e.shape) == 1: # original
            for j, energy in enumerate(e):
                tmp = material_constants_library.CS_Total_CP(substance, energy / 1000.0)
                transmittance[j, :, :] = numpy.exp(-tmp * dens * (thick / 10.0))
        else: # harmonics
            tmp = material_constants_library.CS_Total_CP(substance, e / 1000.0)
            transmittance = numpy.exp(-tmp * dens * (thick / 10.0))


        # rotation
        H = h / numpy.cos(vrot * numpy.pi / 180)
        V = v / numpy.cos(hrot * numpy.pi / 180)

        # aperture
        if gapshape == 0:
            h_indices_bad = numpy.where(numpy.abs(H - hgapcenter) > (0.5 * hgap))
            if len(h_indices_bad) > 0:
                transmittance[:, h_indices_bad, :] = 0.0
            v_indices_bad = numpy.where(numpy.abs(V - vgapcenter) > (0.5 * vgap))
            if len(v_indices_bad) > 0:
                transmittance[:, :, v_indices_bad] = 0.0
        elif gapshape == 1:
            for i in range(H.size):
                for j in range(V.size):
                    if (((H[i] - hgapcenter) / (hgap / 2)) ** 2 + ((V[j] - vgapcenter) / (vgap / 2)) ** 2) > 1:
                        transmittance[:, i, j] = 0.0

        absorbance = 1.0 - transmittance

    elif flags == 1:  # mirror
        if len(e.shape) == 1: # original

            tmp = numpy.zeros(e.size)

            for j, energy in enumerate(e):
                tmp[j] = material_constants_library.Refractive_Index_Re(substance, energy / 1000.0, dens)

            if tmp[0] == 0.0:
                raise Exception("Probably the substrance %s is wrong" % substance)

            delta = 1.0 - tmp
            beta = numpy.zeros(e.size)

            for j, energy in enumerate(e):
                beta[j] = material_constants_library.Refractive_Index_Im(substance, energy / 1000.0, dens)

        else:
            # tmp = material_constants_library.CS_Total_CP(substance, e / 1000.0)
            # transmittance = numpy.exp(-tmp * dens * (thick / 10.0))

            tmp = material_constants_library.Refractive_Index_Re(substance, e / 1000.0, dens)
            delta = 1.0 - tmp
            beta = material_constants_library.Refractive_Index_Im(substance, e / 1000.0, dens)

        try:
            (rs, rp, runp) = reflectivity_fresnel(refraction_index_beta=beta, refraction_index_delta=delta, \
                                                  grazing_angle_mrad=angle, roughness_rms_A=roughness, \
                                                  photon_energy_ev=e)
        except:
            raise Exception("Failed to run reflectivity_fresnel")

        for j, energy in enumerate(e):
            transmittance[j, :, :] = rs[j]

        # rotation
        if defection == 0:  # horizontally deflecting
            H = h / numpy.sin(angle * 1e-3)
        elif defection == 1:  # vertically deflecting
            V = v / numpy.sin(angle * 1e-3)

        # size
        absorbance = 1.0 - transmittance

        if gapshape == 0:
            h_indices_bad = numpy.where(numpy.abs(H - hgapcenter) > (0.5 * hgap))
            if len(h_indices_bad) > 0:
                transmittance[:, h_indices_bad, :] = 0.0
                absorbance[:, h_indices_bad, :] = 0.0
            v_indices_bad = numpy.where(numpy.abs(V - vgapcenter) > (0.5 * vgap))
            if len(v_indices_bad) > 0:
                transmittance[:, :, v_indices_bad] = 0.0
                absorbance[:, :, v_indices_bad] = 0.0
        elif gapshape == 1:  # circle
            for i in range(H.size):
                for j in range(V.size):
                    if (((H[i] - hgapcenter) / (hgap / 2)) ** 2 + ((V[j] - vgapcenter) / (vgap / 2)) ** 2) > 1:
                        transmittance[:, i, j] = 0.0
                        absorbance[:, i, j] = 0.0

    elif flags == 2:  # aperture
        if gapshape == 0:
            h_indices_bad = numpy.where(numpy.abs(H - hgapcenter) > (0.5 * hgap))
            if len(h_indices_bad) > 0:
                transmittance[:, h_indices_bad, :] = 0.0
            v_indices_bad = numpy.where(numpy.abs(V - vgapcenter) > (0.5 * vgap))
            if len(v_indices_bad) > 0:
                transmittance[:, :, v_indices_bad] = 0.0
        elif gapshape == 1:  # circle
            for i in range(H.size):
                for j in range(V.size):
                    if (((H[i] - hgapcenter) / (hgap / 2))**2 + ((V[j] - vgapcenter)/ (vgap / 2))**2) > 1:
                        transmittance[:, i, j] = 0.0

        absorbance = 1.0 - transmittance

    elif flags == 3:  # magnifier
        H = h * hmag
        V = v * vmag

        absorbance = 1.0 - transmittance

    elif flags == 4:  # rotation screen
        H = h / numpy.cos(vrot * numpy.pi / 180)
        V = v / numpy.cos(hrot * numpy.pi / 180)

        absorbance = 1.0 - transmittance

    elif flags == 5:  # thin object filter
        H = h
        V = v

        # FRONT surface
        xx, yy, zz = read_surface_file(file_name=thin_object_file)
        thick_pre = zz.T * 1e3 # im mm
        xx *= 1e3 # im mm
        yy *= 1e3 # im mm

        f = RectBivariateSpline(xx,
                                yy,
                                thick_pre)
        thick_interpolated = f(h, v)

        # outside definition (supposed centered)
        h_indices_bad = numpy.where(numpy.abs(H) > (xx.max()))
        if len(h_indices_bad) > 0:
            thick_interpolated[h_indices_bad, :] = thin_object_thickness_outside_file_area * 1e3 # in mm
        v_indices_bad = numpy.where(numpy.abs(V) > (yy.max()))
        if len(v_indices_bad) > 0:
            thick_interpolated[:, v_indices_bad] = thin_object_thickness_outside_file_area * 1e3 # in mm

        # BACK surface
        if thin_object_back_profile_flag == 0:
            thick_interpolated_back = 0.0
        elif thin_object_back_profile_flag == 1:
            xx, yy, zz = read_surface_file(file_name=thin_object_back_profile_file)
            thick_pre = zz.T * 1e3 # im mm
            xx *= 1e3 # im mm
            yy *= 1e3 # im mm

            f = RectBivariateSpline(xx,
                                    yy,
                                    thick_pre)
            thick_interpolated_back = f(h, v)

            # outside definition (supposed centered) is set to zero
            h_indices_bad = numpy.where(numpy.abs(H) > (xx.max()))
            if len(h_indices_bad) > 0:
                thick_interpolated_back[h_indices_bad, :] = 0.0
            v_indices_bad = numpy.where(numpy.abs(V) > (yy.max()))
            if len(v_indices_bad) > 0:
                thick_interpolated_back[:, v_indices_bad] = 0.0

        # total thickness is |front - back profiles|
        thick_interpolated = numpy.abs(thick_interpolated - thick_interpolated_back)

        if len(e.shape) == 1: # original
            for j, energy in enumerate(e):
                tmp = material_constants_library.CS_Total_CP(substance, energy / 1000.0)
                transmittance[j, :, :] = numpy.exp(-tmp * dens * (thick_interpolated[:, :] / 10.0))
        else: # vs harmonics
            tmp = material_constants_library.CS_Total_CP(substance, e / 1000.0)
            for j, energy in enumerate(e):
                transmittance[j, :, :] = numpy.exp(-tmp[j, :, :] * dens * (thick_interpolated[:, :] / 10.0))

        absorbance = 1.0 - transmittance

    elif flags == 6:  # multilayer
        out = get_multilayer_instance(multilayer_file, material_constants_library=None, verbose=0)
        rs, rp = out.scan_energy(e, numpy.degrees(angle * 1e-3))

        if defection == 0:  # horizontally deflecting
            if len(e.shape) == 1: # usual case: vs energy
                for j, energy in enumerate(e):
                    transmittance[j, :, :] = rp[j] ** 2
            else: # vs harmonics
                transmittance = rp ** 2
            H = h / numpy.sin(angle * 1e-3)
        elif defection == 1:  # vertically deflecting
            if len(e.shape) == 1:  # usual case: vs energy
                for j, energy in enumerate(e):
                    transmittance[j, :, :] = rs[j] ** 2
            else:
                transmittance = rs ** 2
            V = v / numpy.sin(angle * 1e-3)

        # size
        absorbance = 1.0 - transmittance

        if gapshape == 0:
            h_indices_bad = numpy.where(numpy.abs(H - hgapcenter) > (0.5 * hgap))
            if len(h_indices_bad) > 0:
                transmittance[:, h_indices_bad, :] = 0.0
                absorbance[:, h_indices_bad, :] = 0.0
            v_indices_bad = numpy.where(numpy.abs(V - vgapcenter) > (0.5 * vgap))
            if len(v_indices_bad) > 0:
                transmittance[:, :, v_indices_bad] = 0.0
                absorbance[:, :, v_indices_bad] = 0.0
        elif gapshape == 1:  # circle
            for i in range(H.size):
                for j in range(V.size):
                    if (((H[i] - hgapcenter) / (hgap / 2)) ** 2 + ((V[j] - vgapcenter) / (vgap / 2)) ** 2) > 1:
                        transmittance[:, i, j] = 0.0
                        absorbance[:, i, j] = 0.0

    elif flags == 7:  # external reflectivity file

        try:
            a = numpy.loadtxt(external_reflectivity_file)
            energy0 = a[:, 0]
            refl0 = a[:, -1]
            refl = numpy.interp(e, energy0, refl0, left=0, right=0)
        except:
            raise Exception("Error extracting reflectivity from file: %s" % external_reflectivity_file)

        for j, energy in enumerate(e):
            transmittance[j, :, :] = refl[j]

        # rotation
        H = h / numpy.cos(vrot * numpy.pi / 180)
        V = v / numpy.cos(hrot * numpy.pi / 180)

        # size
        absorbance = 1.0 - transmittance

        if gapshape == 0:
            h_indices_bad = numpy.where(numpy.abs(H - hgapcenter) > (0.5 * hgap))
            if len(h_indices_bad) > 0:
                transmittance[:, h_indices_bad, :] = 0.0
                absorbance[:, h_indices_bad, :] = 0.0
            v_indices_bad = numpy.where(numpy.abs(V - vgapcenter) > (0.5 * vgap))
            if len(v_indices_bad) > 0:
                transmittance[:, :, v_indices_bad] = 0.0
                absorbance[:, :, v_indices_bad] = 0.0
        elif gapshape == 1:  # circle
            for i in range(H.size):
                for j in range(V.size):
                    if (((H[i] - hgapcenter) / (hgap / 2))  ** 2 + ((V[j] - vgapcenter) / (vgap / 2)) ** 2) > 1:
                        transmittance[:, i, j] = 0.0
                        absorbance[:, i, j] = 0.0
    else:
        raise NotImplementedError("Not implemented option: flags=%d" % flags)

    return transmittance, absorbance, E, H, V, txt

def apply_transmittance_to_incident_beam(transmittance, p0, e0, h0, v0,
    flags=0,
    hgap=1000.0,
    vgap=1000.0,
    hgapcenter=0.0,
    vgapcenter=0.0,
    hmag=1.0,
    vmag=1.0,
    interpolation_flag=0,
    interpolation_factor_h=1.0,
    interpolation_factor_v=1.0,
    slit_crop=0,
    ):

    p = p0.copy()
    e = e0.copy()
    h = h0.copy()
    v = v0.copy()

    # coordinates to send: the same as incident beam (perpendicular to the optical axis)
    # except for the magnifier
    if flags == 3:  # magnifier  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        h *= hmag
        v *= vmag

    p_transmitted = p * transmittance / (h[0] / h0[0]) / (v[0] / v0[0])

    if (slit_crop == 1 and flags == 2):
        print("Cropping to %g mm x %g mm" % (hgap,vgap))
        h_d = numpy.where(numpy.abs(h - hgapcenter) <= (0.5 * hgap))[0]
        v_d = numpy.where(numpy.abs(v - vgapcenter) <= (0.5 * vgap))[0]

        p_transmitted = p_transmitted[:, h_d[0]:(h_d[-1]), v_d[0]:v_d[-1]].copy()
        e = e
        h = h[h_d[0]:h_d[-1]].copy()
        v = v[v_d[0]:v_d[-1]].copy()

    if interpolation_flag == 1:

        h_old = h
        v_old = v
        p_transmitted_old = p_transmitted

        nh = int(h_old.size * interpolation_factor_h)
        nv = int(v_old.size * interpolation_factor_v)
        p_transmitted = numpy.zeros((e.shape[0], nh, nv))

        print("Interpolating from ",p_transmitted_old.shape, " to ", p_transmitted.shape)
        h = numpy.linspace(h_old[0], h_old[-1], nh)
        v = numpy.linspace(v_old[0], v_old[-1], nv)


        for i in range(e.shape[0]):
            f = RectBivariateSpline(h_old,
                                    v_old,
                                    p_transmitted_old[i, :, :])
            p_transmitted_i = f(h, v)
            p_transmitted[i, :, :] = p_transmitted_i

    return p_transmitted, e, h, v


#
# file io
#

# copied from OASYS1
def read_surface_file(file_name, subgroup_name = "surface_file"):
    if not os.path.isfile(file_name): raise ValueError("File " + file_name + " not existing")

    file = h5py.File(file_name, 'r')
    xx = file[subgroup_name + "/X"][()]
    yy = file[subgroup_name + "/Y"][()]
    zz = file[subgroup_name + "/Z"][()]

    return xx, yy, zz

def load_radiation_from_h5file(file_h5, subtitle="XOPPY_RADIATION"):

    hf = h5py.File(file_h5, 'r')

    if not subtitle in hf:
        raise Exception("XOPPY_RADIATION not found in h5 file %s"%file_h5)

    try:
        p = hf[subtitle + "/Radiation/stack_data"][:]
        e = hf[subtitle + "/Radiation/axis0"][:]
        h = hf[subtitle + "/Radiation/axis1"][:]
        v = hf[subtitle + "/Radiation/axis2"][:]
    except:
        raise Exception("Error reading h5 file %s \n"%file_h5 + "\n" + str(e))

    code = "unknown"

    try:
        if hf[subtitle + "/parameters/METHOD"][()] == 0:
            code = 'US'
        elif hf[subtitle + "/parameters/METHOD"][()] == 1:
            code = 'URGENT'
        elif hf[subtitle + "/parameters/METHOD"][()] == 2:
            code = 'SRW'
        elif hf[subtitle + "/parameters/METHOD"][()] == 3:
            code = 'pySRU'
    except:
        pass

    hf.close()

    return e, h, v, p, code


def write_radiation_to_h5file(e,h,v,p,
                       creator="xoppy",
                       h5_file="wiggler_radiation.h5",
                       h5_entry_name="XOPPY_RADIATION",
                       h5_initialize=True,
                       h5_parameters=None,
                       ):

    if h5_initialize:
        h5w = H5SimpleWriter.initialize_file(h5_file,creator=creator)
    else:
        h5w = H5SimpleWriter(h5_file,None)
    h5w.create_entry(h5_entry_name,nx_default=None)
    h5w.add_stack(e,h,v,p,stack_name="Radiation",entry_name=h5_entry_name,
        title_0="Photon energy [eV]",
        title_1="X gap [mm]",
        title_2="Y gap [mm]")
    h5w.create_entry("parameters",root_entry=h5_entry_name,nx_default=None)
    if h5_parameters is not None:
        for key in h5_parameters.keys():
            h5w.add_key(key,h5_parameters[key], entry_name=h5_entry_name+"/parameters")
    print("File written to disk: %s"%h5_file)


def write_txt_file(calculated_data, input_beam_content, filename="tmp.txt", method="3columns"):

    p0, e0, h0, v0 = input_beam_content # .get_content("xoppy_data")
    transmittance, absorbance, E, H, V = calculated_data

    #
    #
    #
    p = p0.copy()
    p_spectral_power = p * codata.e * 1e3

    absorbed3d = p_spectral_power * absorbance / (H[0] / h0[0]) / (V[0] / v0[0])
    try:    absorbed2d = numpy.trapezoid(absorbed3d, E, axis=0)
    except: absorbed2d = numpy.trapz(absorbed3d, E, axis=0)

    f = open(filename, 'w')
    if method == "3columns":
        for i in range(H.size):
            for j in range(V.size):
                f.write("%g  %g  %g\n" % (H[i]*1e-3, V[i]*1e-3, absorbed2d[i,j]*1e6))
    elif method == "matrix":
        f.write("%10.5g" % 0)
        for i in range(H.size):
            f.write(", %10.5g" % (H[i] * 1e-3))
        f.write("\n")

        for j in range(V.size):
                f.write("%10.5g" % (V[j] * 1e-3))
                for i in range(H.size):
                    f.write(", %10.5g" % (absorbed2d[i,j] * 1e6))
                f.write("\n")
    else:
        raise Exception("File type not understood.")
    f.close()

    print("File written to disk: %s" % filename)


def write_h5_file(calculated_data, input_beam_content, filename="tmp.txt",EL1_FLAG=1,EL1_HMAG=1.0,EL1_VMAG=1.0):

    p0, e0, h0, v0 = input_beam_content #.get_content("xoppy_data")
    transmittance, absorbance, E, H, V = calculated_data

    #
    #
    #
    p = p0.copy()
    e = e0.copy()
    h = h0.copy()
    v = v0.copy()
    p_spectral_power = p * codata.e * 1e3

    try:
        h5w = H5SimpleWriter.initialize_file(filename, creator="power3Dcomponent.py")
        txt = "\n\n\n"
        txt += info_total_power(p, e, v, h, transmittance, absorbance, EL1_FLAG=EL1_FLAG)
        h5w.add_key("info", txt, entry_name=None)
    except:
        print("ERROR writing h5 file (info)")


    try:
        #
        # source
        #
        entry_name = "source"

        h5w.create_entry(entry_name, nx_default=None)

        h5w.add_stack(e, h, v, p, stack_name="Radiation stack", entry_name=entry_name,
                      title_0="Photon energy [eV]",
                      title_1="X [mm] (normal to beam)",
                      title_2="Y [mm] (normal to beam)")

        try:    h5w.add_image(numpy.trapezoid(p_spectral_power, E, axis=0) , H, V,
                      image_name="Power Density", entry_name=entry_name,
                      title_x="X [mm] (normal to beam)",
                      title_y="Y [mm] (normal to beam)")
        except: h5w.add_image(numpy.trapz(p_spectral_power, E, axis=0) , H, V,
                      image_name="Power Density", entry_name=entry_name,
                      title_x="X [mm] (normal to beam)",
                      title_y="Y [mm] (normal to beam)")

        try:    h5w.add_dataset(E, numpy.trapezoid(numpy.trapezoid(p_spectral_power, v, axis=2), h, axis=1),
                        entry_name=entry_name, dataset_name="Spectral power",
                        title_x="Photon Energy [eV]",
                        title_y="Spectral density [W/eV]")
        except: h5w.add_dataset(E, numpy.trapz(numpy.trapz(p_spectral_power, v, axis=2), h, axis=1),
                        entry_name=entry_name, dataset_name="Spectral power",
                        title_x="Photon Energy [eV]",
                        title_y="Spectral density [W/eV]")

    except:
        print("ERROR writing h5 file (source)")

    try:
        #
        # optical element
        #
        entry_name = "optical_element"

        h5w.create_entry(entry_name, nx_default=None)

        h5w.add_stack(E, H, V, transmittance, stack_name="Transmittance stack", entry_name=entry_name,
                      title_0="Photon energy [eV]",
                      title_1="X [mm] (o.e. coordinates)",
                      title_2="Y [mm] (o.e. coordinates)")

        absorbed = p_spectral_power * absorbance / (H[0] / h0[0]) / (V[0] / v0[0])
        try:    h5w.add_image(numpy.trapezoid(absorbed, E, axis=0), H, V,
                      image_name="Absorbed Power Density on Element", entry_name=entry_name,
                      title_x="X [mm] (o.e. coordinates)",
                      title_y="Y [mm] (o.e. coordinates)")
        except: h5w.add_image(numpy.trapz(absorbed, E, axis=0), H, V,
                      image_name="Absorbed Power Density on Element", entry_name=entry_name,
                      title_x="X [mm] (o.e. coordinates)",
                      title_y="Y [mm] (o.e. coordinates)")
        try:    h5w.add_dataset(E, numpy.trapezoid(numpy.trapezoid(absorbed, v, axis=2), h, axis=1),
                        entry_name=entry_name, dataset_name="Absorbed Spectral Power",
                        title_x="Photon Energy [eV]",
                        title_y="Spectral density [W/eV]")
        except: h5w.add_dataset(E, numpy.trapz(numpy.trapz(absorbed, v, axis=2), h, axis=1),
                        entry_name=entry_name, dataset_name="Absorbed Spectral Power",
                        title_x="Photon Energy [eV]",
                        title_y="Spectral density [W/eV]")
        #
        # transmitted
        #

        # coordinates to send: the same as incident beam (perpendicular to the optical axis)
        # except for the magnifier
        if EL1_FLAG == 3:  # magnifier <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            h *= EL1_HMAG
            v *= EL1_VMAG

        transmitted = p_spectral_power * transmittance / (h[0] / h0[0]) / (v[0] / v0[0])
        try:    h5w.add_image(numpy.trapezoid(transmitted, E, axis=0), h, v,
                      image_name="Transmitted Power Density on Element", entry_name=entry_name,
                      title_x="X [mm] (normal to beam)",
                      title_y="Y [mm] (normal to beam)")
        except: h5w.add_image(numpy.trapz(transmitted, E, axis=0), h, v,
                      image_name="Transmitted Power Density on Element", entry_name=entry_name,
                      title_x="X [mm] (normal to beam)",
                      title_y="Y [mm] (normal to beam)")

        try:    h5w.add_dataset(E, numpy.trapezoid(numpy.trapezoid(transmitted, v, axis=2), h, axis=1),
                        entry_name=entry_name, dataset_name="Transmitted Spectral Power",
                        title_x="Photon Energy [eV]",
                        title_y="Spectral density [W/eV]")
        except: h5w.add_dataset(E, numpy.trapz(numpy.trapz(transmitted, v, axis=2), h, axis=1),
                        entry_name=entry_name, dataset_name="Transmitted Spectral Power",
                        title_x="Photon Energy [eV]",
                        title_y="Spectral density [W/eV]")
    except:
        print("ERROR writing h5 file (optical element)")

    try:
        h5_entry_name = "XOPPY_RADIATION"

        h5w.create_entry(h5_entry_name,nx_default=None)
        h5w.add_stack(e, h, v, transmitted,stack_name="Radiation",entry_name=h5_entry_name,
            title_0="Photon energy [eV]",
            title_1="X gap [mm]",
            title_2="Y gap [mm]")
    except:
        print("ERROR writing h5 file (adding XOPPY_RADIATION)")

    print("File written to disk: %s" % filename)

#
# info
#
def info_total_power(p, e, v, h, transmittance, absorbance, EL1_FLAG=1,
                     method=2, # 0=trapezoid, 1=sum, 2=both
                     ):

    method_str = ['trapezoid','sum']

    if method == 2:
        methods = [0, 1]
    else:
        methods = [method]

    txt = ""
    for method in methods:
        txt += "\n\n\n"
        txt += "(info_total_power: method=%d (%s))\n" % (method, method_str[method])

        power_input = integral_3d(p, e, h, v, method=method) * codata.e * 1e3
        txt += '      Input beam power: %f W\n'%(power_input)

        power_transmitted = integral_3d(p * transmittance, e, h, v, method=method) * codata.e * 1e3
        power_absorbed = integral_3d(p * absorbance, e, h, v, method=method) * codata.e * 1e3

        power_lost = power_input - ( power_transmitted +  power_absorbed)
        if numpy.abs( power_lost ) > 1e-9:
            txt += '      Beam power not considered (removed by o.e. acceptance): %6.3f W (accepted: %6.3f W)\n' % \
                   (power_lost, power_input - power_lost)

        txt += '      Beam power absorbed by optical element: %6.3f W\n' % power_absorbed

        if EL1_FLAG == 1:
            txt += '      Beam power reflected after optical element: %6.3f W\n' % power_transmitted
        else:
            txt += '      Beam power transmitted after optical element: %6.3f W\n' % power_transmitted

    return txt

if __name__ == "__main__":
    from srxraylib.plot.gol import plot_image, plot

    if 0: # multilayer
        GAPH = 0.002
        GAPV = 0.002
        HSLITPOINTS = 601
        VSLITPOINTS = 401
        PHOTONENERGYMIN = 7000.0
        PHOTONENERGYMAX = 8000.0
        PHOTONENERGYPOINTS = 20

        e0 = numpy.linspace(PHOTONENERGYMIN, PHOTONENERGYMAX, PHOTONENERGYPOINTS)
        h0 = numpy.linspace(-0.5 * GAPH, 0.5 * GAPH, HSLITPOINTS)
        v0 = numpy.linspace(-0.5 * GAPV, 0.5 * GAPV, VSLITPOINTS)



        transmittance, absorbance, E, H, V, txt  = calculate_component_absorbance_and_transmittance(
                        e0, # energy in eV
                        h0, # h in mm
                        v0, # v in mm
                        substance='Be',
                        thick=0.5,
                        angle=3.0,
                        defection=1,
                        dens='?',
                        roughness=0.0,
                        flags=6, # 0 = Filter, 2=Aperture, 5 = Thin object filter, 6=Multilayer, 7 External File
                        hgap=0.2e-3,
                        vgap=0.1e-3,
                        hgapcenter=0.1e-3,
                        vgapcenter=0.1e-3,
                        gapshape=1, # 0=rect
                        hmag=1.0,
                        vmag=1.0,
                        hrot=0.0,
                        vrot=0.0,
                        thin_object_file='/home/srio/Oasys/lens.h5',
                        thin_object_thickness_outside_file_area=163e-6,
                        thin_object_back_profile_flag=0, # 0= z=0, 1=Back profile from file z(x,y)
                        thin_object_back_profile_file='/home/srio/Oasys/lens.h5',
                        multilayer_file='/home/srio/Oasys/multilayerTiC.h5',
                        )



        plot(e0, transmittance[:,0,0],
             e0, absorbance[:,0,0],
             xtitle="Photon energy [eV]", legend=["Transmittance","Absorbance"], show=0)
        plot_image(transmittance[0,:,:], h0, v0,title="Transmittance at E=%g eV" % (e0[0]),
                   xtitle="H [m]",ytitle="V [m]",aspect='auto')
        if 0:
            from srxraylib.plot.gol import plot_image
            import scipy.constants as codata
            from xoppylib.power.power3d import integral_2d

            # transmitted/reflected beam

            spectral_power_transmitted = f_transmitted * codata.e * 1e3
            plot_image(spectral_power_transmitted[0, :, :], h, v,
                       title="Transmitted Spectral Power Density [W/eV/mm2] at E=%g eV" % (100.0), xtitle="H [mm]",
                       ytitle="V [mm]", aspect='auto')

            power_density_transmitted = numpy.trapezoid(spectral_power_transmitted, e, axis=0)
            power_density_integral = integral_2d(power_density_transmitted, h, v)
            plot_image(power_density_transmitted, h, v,
                       xtitle='H [mm] (normal to beam)',
                       ytitle='V [mm] (normal to beam)',
                       title='Power Density [W/mm^2]. Integral: %6.3f W' % power_density_integral, aspect='auto')

            # local absorption

            spectral_power_density_absorbed = f_absorbed * codata.e * 1e3

            plot_image(spectral_power_density_absorbed[0, :, :], H, V,
                       title="Absorbed Spectral Power Density [W/eV/mm2] at E=%g eV" % (100.0), xtitle="H [mm]",
                       ytitle="V [mm]", aspect='auto')

            power_density_absorbed = numpy.trapezoid(spectral_power_density_absorbed, E, axis=0)
            power_density_integral = integral_2d(power_density_absorbed, H, V)
            plot_image(power_density_absorbed, H, V,
                       xtitle='H [mm] (o.e. coordinates)',
                       ytitle='V [mm] (o.e. coordinates)',
                       title='Absorbed Power Density [W/mm^2]. Integral: %6.3f W' % power_density_integral,
                       aspect='auto')

    if 1: # test harmonic calculation
        #
        # script to make the calculations (created by XOPPY:undulator_spectrum)
        #
        from xoppylib.sources.xoppy_undulators import xoppy_calc_undulator_power_density_from_harmonics

        # define inputs here to be written in h5 file
        h5_parameters = dict()
        h5_parameters["ELECTRONENERGY"] = 6.0
        h5_parameters["ELECTRONENERGYSPREAD"] = 0.001
        h5_parameters["ELECTRONCURRENT"] = 0.2
        h5_parameters["ELECTRONBEAMSIZEH"] = 3.34281e-05
        h5_parameters["ELECTRONBEAMSIZEV"] = 7.28139e-06
        h5_parameters["ELECTRONBEAMDIVERGENCEH"] = 4.51097e-06
        h5_parameters["ELECTRONBEAMDIVERGENCEV"] = 1.94034e-06
        h5_parameters["PERIODID"] = 0.018
        h5_parameters["NPERIODS"] = 111
        h5_parameters["KV"] = 1.6563
        h5_parameters["KH"] = 0.0
        h5_parameters["KPHASE"] = 0.0
        h5_parameters["DISTANCE"] = 23.0
        h5_parameters["GAPH"] = 0.01
        h5_parameters["GAPV"] = 0.01
        h5_parameters["HSLITPOINTS"] = 51
        h5_parameters["VSLITPOINTS"] = 51
        h5_parameters["METHOD"] = 1  # 0=urgent (fortran), 1=urgentpy (python)
        h5_parameters["USEEMITTANCES"] = 1
        h5_parameters["MASK_FLAG"] = 0
        h5_parameters["MASK_ROT_H_DEG"] = 0.0
        h5_parameters["MASK_ROT_V_DEG"] = 0.0
        h5_parameters["MASK_H_MIN"] = -1000.0
        h5_parameters["MASK_H_MAX"] = 1000.0
        h5_parameters["MASK_V_MIN"] = -1000.0
        h5_parameters["MASK_V_MAX"] = 1000.0
        h5_parameters["harmonic_max"] = 5  # maximum harmonic to calculate
        h5_parameters["photon_energy_bin"] = 300.0  # in eV, for calculating Spectral Power

        horizontal, vertical, power_density, code, power_density_harmonics, energy, spectral_power, spectral_power_energy, flux3D = xoppy_calc_undulator_power_density_from_harmonics(
            ELECTRONENERGY=h5_parameters["ELECTRONENERGY"],
            ELECTRONENERGYSPREAD=h5_parameters["ELECTRONENERGYSPREAD"],
            ELECTRONCURRENT=h5_parameters["ELECTRONCURRENT"],
            ELECTRONBEAMSIZEH=h5_parameters["ELECTRONBEAMSIZEH"],
            ELECTRONBEAMSIZEV=h5_parameters["ELECTRONBEAMSIZEV"],
            ELECTRONBEAMDIVERGENCEH=h5_parameters["ELECTRONBEAMDIVERGENCEH"],
            ELECTRONBEAMDIVERGENCEV=h5_parameters["ELECTRONBEAMDIVERGENCEV"],
            PERIODID=h5_parameters["PERIODID"],
            NPERIODS=h5_parameters["NPERIODS"],
            KV=h5_parameters["KV"],
            KH=h5_parameters["KH"],
            KPHASE=h5_parameters["KPHASE"],
            DISTANCE=h5_parameters["DISTANCE"],
            GAPH=h5_parameters["GAPH"],
            GAPV=h5_parameters["GAPV"],
            HSLITPOINTS=h5_parameters["HSLITPOINTS"],
            VSLITPOINTS=h5_parameters["VSLITPOINTS"],
            METHOD=h5_parameters["METHOD"],
            harmonic_max=h5_parameters["harmonic_max"],
            USEEMITTANCES=h5_parameters["USEEMITTANCES"],
            MASK_FLAG=h5_parameters["MASK_FLAG"],
            MASK_ROT_H_DEG=h5_parameters["MASK_ROT_H_DEG"],
            MASK_ROT_V_DEG=h5_parameters["MASK_ROT_V_DEG"],
            MASK_H_MIN=h5_parameters["MASK_H_MIN"],
            MASK_H_MAX=h5_parameters["MASK_H_MAX"],
            MASK_V_MIN=h5_parameters["MASK_V_MIN"],
            MASK_V_MAX=h5_parameters["MASK_V_MAX"],
            h5_file="undulator_power_density_from_harmonics.h5",
            h5_entry_name="XOPPY_POWERDENSITY_FROM_HARMONICS",
            h5_initialize=True,
            h5_parameters=h5_parameters,
            photon_energy_bin=h5_parameters["photon_energy_bin"],
        )

        #
        # script to make the calculations (created by XOPPY:power3Dcomponent)
        #

        import numpy

        try:
            import xraylib
        except:
            print("xraylib not available")
        from dabax.dabax_xraylib import DabaxXraylib
        from xoppylib.power.power3d import calculate_component_absorbance_and_transmittance
        from xoppylib.power.power3d import apply_transmittance_to_incident_beam

        # compute local transmittance and absorbance
        e0, h0, v0, f0 = energy, horizontal, vertical, flux3D
        transmittance, absorbance, E, H, V, txt = calculate_component_absorbance_and_transmittance(
            e0,  # energy in eV
            h0,  # h in mm
            v0,  # v in mm
            substance='Rh',
            thick=0.5,
            angle=3.0,
            defection=1,
            dens='?',
            roughness=0.0,
            flags=1,
            # 0=Filter 1=Mirror 2=Aperture 3=magnifier, 4=Screen rotation  5=Thin object  6=Multilayer 7=External file
            hgap=1000.0,
            vgap=1000.0,
            hgapcenter=0.0,
            vgapcenter=0.0,
            hmag=1.0,
            vmag=1.0,
            hrot=0.0,
            vrot=0.0,
            thin_object_file='',
            thin_object_thickness_outside_file_area=0.0,
            thin_object_back_profile_flag=0,
            thin_object_back_profile_file='',
            multilayer_file='',
            external_reflectivity_file='',
            material_constants_library=DabaxXraylib(file_f1f2="f1f2_Windt.dat",
                                                    file_CrossSec="CrossSec_EPDL97.dat"),
        )


        # from srxraylib.plot.gol import plot

        # plot(e.flatten(), transmittance.flatten(), linestyle="", marker=".")
        plot(E.flatten(), transmittance.flatten(), marker='.', linestyle="", xrange=[5000, 35000])
        plot(E.flatten(), transmittance.flatten(), marker='.', linestyle="")
        plot(E.flatten(), absorbance.flatten(), marker='.', linestyle="", xrange=[5000, 35000])
        plot(E.flatten(), absorbance.flatten(), marker='.', linestyle="")
        # # apply transmittance to incident beam
        # f_transmitted, e, h, v = apply_transmittance_to_incident_beam(transmittance, f0, e0, h0, v0,
        #                                                               flags=0,
        #                                                               hgap=1000.0,
        #                                                               vgap=1000.0,
        #                                                               hgapcenter=0.0,
        #                                                               vgapcenter=0.0,
        #                                                               hmag=1.0,
        #                                                               vmag=1.0,
        #                                                               interpolation_flag=0,
        #                                                               interpolation_factor_h=1.0,
        #                                                               interpolation_factor_v=1.0,
        #                                                               slit_crop=0,
        #                                                               )
        #
        # f_absorbed = f0 * absorbance / (H[0] / h0[0]) / (V[0] / v0[0])
        #
        # # data to pass
        # energy, horizontal, vertical, flux3D = e, h, v, f_transmitted
        #
        # #
        # # example plots
        # #
        # if True:
        #     from srxraylib.plot.gol import plot_image
        #     import scipy.constants as codata
        #     from xoppylib.power.power3d import integral_2d
        #
        #     # transmitted/reflected beam
        #
        #     spectral_power_transmitted = f_transmitted * codata.e * 1e3
        #     plot_image(spectral_power_transmitted[0, :, :], h, v,
        #                title="Transmitted Spectral Power Density [W/eV/mm2] at lower energy", xtitle="H [mm]",
        #                ytitle="V [mm]", aspect='auto')
        #
        #     power_density_transmitted = numpy.trapezoid(spectral_power_transmitted, e, axis=0)
        #     power_density_integral = integral_2d(power_density_transmitted, h, v)
        #     plot_image(power_density_transmitted, h, v,
        #                xtitle='H [mm] (normal to beam)',
        #                ytitle='V [mm] (normal to beam)',
        #                title='Power Density [W/mm^2]. Integral: %6.3f W' % power_density_integral, aspect='auto')
        #
        #     # local absorption
        #
        #     spectral_power_density_absorbed = f_absorbed * codata.e * 1e3
        #
        #     plot_image(spectral_power_density_absorbed[0, :, :], H, V,
        #                title="Absorbed Spectral Power Density [W/eV/mm2] at lower energy", xtitle="H [mm]",
        #                ytitle="V [mm]", aspect='auto')
        #
        #     power_density_absorbed = numpy.trapezoid(spectral_power_density_absorbed, E, axis=0)
        #     power_density_integral = integral_2d(power_density_absorbed, H, V)
        #     plot_image(power_density_absorbed, H, V,
        #                xtitle='H [mm] (o.e. coordinates)',
        #                ytitle='V [mm] (o.e. coordinates)',
        #                title='Absorbed Power Density [W/mm^2]. Integral: %6.3f W' % power_density_integral,
        #                aspect='auto')
        #
        # #
        # # end script
        # #

