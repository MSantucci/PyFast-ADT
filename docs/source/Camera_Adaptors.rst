Camera Adaptors 
===========================
A camera adaptor is a class inherited from ``Cam_base`` and it MUST contains an implementation of the necessary
methods to use properly ``pyFast-ADT`` methods and acquire 3DED data with it.
Usually, thanks to the abstraction layer it contain only binding methods to translate the Camera API interface to
the one of ``pyFast-ADT``, but this can be different if a specific API require special methods/workarounds to run.

A Camera adaptor usually is specific for a certain camera model. ofc if the API used is able to interface multiple cameras can do it.
To give an example, the adaptor_timepix1 module is able to work only with the specific ASI timepix1 512*512 camera.
For most recent detector models such as medipix3 and timepix3 a different adaptor is required (currently adaptor_medipix3 in development).
A proper testing of the functionality is anyway required to ensure that the adaptor is working properly for your setup.

If a camera model is not compatible with an already existing adaptors, a new adaptor should be created to fit the experimental setup.
A procedure on how to implement new camera and microscope adaptors is described in the section:
:doc:`How to implement a custom adaptor <how_to_implement_new_adaptors>`

modules
----------

- **adaptor\_cam**, contain the ``Cam_base`` class and the main methods to handle a generic camera API.
- **adaptor\_timepix1**, specific adaptor built to work with ASI Timepix1 4-quad detector.
- **adaptor\_us2000**, specific adaptor built to work in FEI/Thermo Fisher microscopes with Gatan US2000 camera through temscript package. the camera can be interfaced only if the camera can be handled by TEM Imaging & Analysis (TIA) software.
- **adaptor\_us4000**, specific adaptor built to work in FEI/Thermo Fisher microscopes with Gatan US4000 camera through temscript package. the camera can be interfaced only if the camera can be handled by TEM Imaging & Analysis (TIA) software.
- **adaptor\_xf416r\_GPU**, specific adaptor built to work with a TVIPS XF416R 4k camera GPU accelerated in combination with EMMenu5 software.
- **adaptor\_ceta**, specific adaptor built to work with Thermo Fisher CETA 16M 4k camera through TEM Imaging & Analysis (TIA) software.
- **adaptor\_haadf**, adaptor built to work with a HAADF detector in STEM mode to collect crystal tracking images.

.. currentmodule:: adaptor.camera.adaptor_cam

The structure of camera adaptor
--------------------------------

a camera adaptor should contain all the methods defined from the ``Cam_base`` class, and if necessary specific methods
to handle the specific camera API properly. the main purpose of this adaptor is to handle the basic methods to collect
an image or a series of images, how to save them, setup parameters necessary for the data acquisition, such as:

- connect / disconnect the camera,
- set exposure / binning / processing parameters,
- start / stop live mode,
- acquire an image / series of images,
- prepare / acquire / save continuous rotation data.

in any case just follow an already made camera_adaptor to understand how to implement the methods properly.

Interface classes
^^^^^^^^^^^^^^^^^

:class:`Cam_base` - The entry point...
----------------------------------------

.. function:: connect()

    Creates a new instance of the :class:`Instrument` class. If your computer
    is not the microscope's PC or you don't have the *Scripting* option installed on
    your microscope, this method will raise an exception (most likely of the :exc:`OSError`
    type).

.. class:: Cam_base

    Top level object representing the microscope. Use the :func:`GetInstrument`
    function to create an instance of this class.

    .. attribute:: name

        name of the camera 


    .. method:: connect()

        Initialize the camera



