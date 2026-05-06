.. currentmodule:: xoppylib

===========
Package API
===========

This page lists the main modules in this package.


crystals
--------
``xoppylib.crystals`` Crystal diffraction tools and Bragg preprocessor file utilities

.. autosummary::
   :toctree: generated/

   crystals.tools
   crystals.bragg_preprocessor_file_io
   crystals.create_bragg_preprocessor_file_v1
   crystals.create_bragg_preprocessor_file_v2
   crystals.mare_calc


decorators
----------
``xoppylib.decorators`` Material constants libraries decorated with xoppy calculation functions

.. autosummary::
   :toctree: generated/

   decorators.xoppy_decorator
   decorators.dabax_decorated
   decorators.xraylib_decorated


scattering functions
--------------------
``xoppylib.scattering_functions`` X-ray scattering functions (f0, f1/f2, cross sections, Fresnel)

.. autosummary::
   :toctree: generated/

   scattering_functions.fresnel
   scattering_functions.f0_calc
   scattering_functions.f1f2_calc
   scattering_functions.cross_calc
   scattering_functions.xoppy_calc_f0
   scattering_functions.xoppy_calc_f1f2
   scattering_functions.xoppy_calc_crosssec


power
-----
``xoppylib.power`` Power and flux calculations for optical elements and monochromators

.. autosummary::
   :toctree: generated/

   power.power1d_calc
   power.power1d_calc_monochromators
   power.power3d
   power.xoppy_calc_power
   power.xoppy_calc_power_monochromator


sources
-------
``xoppylib.sources`` Synchrotron radiation source calculators

.. autosummary::
   :toctree: generated/

   sources.xoppy_undulators
   sources.xoppy_bm_wiggler
   sources.xoppy_calc_black_body
   sources.srundplug
   sources.urgentpy_spectrum
   sources.urgentpy_power_density_from_harmonics
   sources.parse_urgent


srcalc
------
``xoppylib.srcalc`` Mirror and grating ray-optics calculations

.. autosummary::
   :toctree: generated/

   srcalc.srcalc
   srcalc.beam
   srcalc.conic
   srcalc.toroid


utilities
---------
``xoppylib`` Top-level utility modules

.. autosummary::
   :toctree: generated/

   mlayer
   xoppy_xraylib_util
   xoppy_util
   xoppy_run_binaries
   fit_gaussian2d
