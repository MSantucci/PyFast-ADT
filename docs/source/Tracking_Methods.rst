Tracking methods
===========================

modules
----------

Detector 

KalmanFilter

cross\_correlation\_kalman\_filtered

objTracking
   
   
.. currentmodule:: adaptor.camera.adaptor_cam

The structure of tracking methods 
-----------------------------------

bla bla blaaaa

Interface classes  
^^^^^^^^^^^^^^^^^
this line with the power symbol is to generate substructures in the webpage !!!

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



