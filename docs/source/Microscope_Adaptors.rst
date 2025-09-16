Microscope Adaptors 
===========================
A microscope adaptor is a class inherited from ``Tem_base`` and it MUST contains an implementation of the necessary
methods to use properly ``pyFast-ADT`` methods and acquire 3DED data with it.
Usually, thanks to the abstraction layer it contain only binding methods to translate the Microscope API interface to
the one of ``pyFast-ADT``, but this can be different if a specific API require special methods/workarounds to run.

A Microscope adaptor can work properly for multiple microscope models if the API used is able to do that.
To give an example, the adaptor_fei module is able to work with both FEI and Thermo Fisher Scientific microscopes, because
the API used (i.e. python package `Temscript <https://github.com/niermann/temscript>`_) is compatible with most of the
models, due to the fact that it is just a wrapper of the TEM COM interface. A proper testing of the functionality is anyway required
to ensure that the adaptor is working properly.

If a microscope model is not compatible with an already existing adaptors, a new adaptor should be created to fit the experimental setup.
A procedure on how to implement new camera and microscope adaptors is described in the section:
:doc:`How to implement a custom adaptor <how_to_implement_new_adaptors>`


modules
----------

- **adaptor\_tem**, contain the ``Tem_base`` class and the main methods to handle a generic microscope API.
- **adaptor\_fei**, specific adaptor built to work with FEI microscopes relatively recent (i.e. 15 years old) and brand new Thermo Fisher microscopes.
- **adaptor\_fei\_temspy**, specific adaptor built to work with old FEI microscopes (i.e. 25 years old) using the temspy to allow the continuous rotation of the goniometer, not allow by scripting for this generation.
- **adaptor\_jeol**, specific adaptor built to work with old generation JEOL microscopes, using TemExt.dll to handle the microscope API. Maybe not completely compatible with new generation JEOL microscopes using pyJEM. Testing required.
- **adaptor\_gatan\_fei**, placeholder to implement the adaptor_fei using python built in from DigitalMicrograph, useful for the Gatan camera.
- **adaptor\_gatan\_jeol**, placeholder to implement the adaptor_jeol using python built in from DigitalMicrograph, useful for the Gatan camera.
- **temspy\_socket**, support module implemented to work togheter with adaptor_fei_temspy to handle the continuous rotation of the goniometer through temspy.

.. currentmodule:: adaptor.microscope.adaptor_tem

The structure of microscope adaptor
--------------------------------

a microscope adaptor should contain all the methods defined from the ``Tem_base`` class, and if necessary specific methods
to handle the specific microscope API properly. the main purpose of this adaptor is to handle the movements of the goniometer in x,y,z,alpha mainly,
handle the required electron optical settings such as:

- connect / disconnect the microscope,
- beam shift (usually handle by Condenser Lens 2 deflectors (C2 shift)),
- beam blanker,
- electron probe beam size (usually handled by C2 % intensity in TEM mode and Objective Lens (OL) defocus in STEM mode for FEI/Thermo Fisher microscopes),
- get/set imaging / diffraction mode,
- get/set magnification value (also camera length (kl) in diffraction),
- handle the threaded stage and beam movement during the 3DED acquisition,
- get the acceleration voltage (to calculate electron wavelength),
- get TEM / STEM mode.

in any case just follow an already made microscope_adaptor to understand how to implement the methods properly.

Interface classes
^^^^^^^^^^^^^^^^^

:class:`Tem_base` - The entry point...
----------------------------------------

.. function:: connect()

    Creates a new instance of the :class:`Instrument` class. If your computer
    is not the microscope's PC or you don't have the *Scripting* option installed on
    your microscope, this method will raise an exception (most likely of the :exc:`OSError`
    type).



.. class:: Tem_base



    Top level object representing the microscope. Use the :func:`GetInstrument`
    function to create an instance of this class.

    .. attribute:: name

        name of the camera 


    .. method:: connect()

        Initialize the camera



