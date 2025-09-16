Introduction
############

``PyFast-ADT`` is a python package to automate 3D Electron Diffraction (3DED) data acquisition for custom Electron Microscope setups.

It's development is based on it's previous version `Fast-ADT <https://github.com/sergiPlana/TEMEDtools>`_
developed by Dr. S.Plana-Ruiz.

``PyFast-ADT`` is able to collect 3DED data in continuous (i.e. continuous Rotation Electron Diffraction - cRED)
and in stepwise (i.e. Automated Electron Diffraction Tomography - ADT) modes, both available for parallel and
quasi parallel illumination (TEM, STEM mode).

On top of that, the software is well known in the community to be able to perform **"a priori" crystal tracking** during
the data acquisition, using the technique of beam shift. In this way, challenging nanocrystals can be measured using a
relatively small electron probe size.

Motivation
**********
``PyFast-ADT`` is meant to overcome previous hardware and software limitations and be the base for further development
in the automation of 3DED data acquisition.
The environment is no more based on Digital Micrograph, but purely Python, which allows for a more flexible and
modular development for further methods implementation.
Moreover, additional hardware to control Scanning-TEM (STEM) is now deprecated, and the software is now able to
interface microscopes and cameras directly through their APIs.

The idea around ``PyFast-ADT`` is to provide a software that can be easily installed and used by the community to
collect data, but also can be used as base for further development and new feature implementation. Please feel free to
fork the repository and contribute to the development of the software. We are happy to add new collaborators in the
project.

Supported Setup
***************
``PyFast-ADT`` is builded to be highly flexible and modular with regards to the hardware setup.
Currently, the software is able to interface with the following Microscopes:
- FEI Tecnai F30 (TEM/STEM)
- FEI Tecnai SPIRIT G2 (TEM/STEM)

and the following Cameras:
- Gatan cameras from TecnaiUserInterface (TUI)
- TVIPS XF416R
- ASI Timepix1

if your setup is not in this list and you would like to use ``PyFast-ADT``, please contact us at:
`marco.santu93@gmail.com` and/or open an issue ticket in the `github repository <https://github.com/MSantucci/PyFast-ADT/tree/dev>`_.

Thanks to it's structure the software can be easily extended to support new hardware, by defining a new
**microscope_adaptor** or **camera_adaptor**. this is a relatively small piece of script that define how the hardware
API connect top the defined functions in ``PyFast-ADT``.

References
**********
If you use ``PyFast-ADT`` in your research, please cite us using the following papers:

- The current project - PyFast-ADT: "currently in the writing process"
- The original project - FAST-ADT: "Fast-ADT: A fast and automated electron diffraction tomography setup for structure determination and refinement" - DOI: 10.1016/j.ultramic.2020.112951 - https://www.sciencedirect.com/science/article/abs/pii/S0304399119303663
- PEDT paper: Mugnaioli, E.; Gorelik, T.; Kolb, U. “Ab initio” structure solution from electron diffraction data obtained by a combination of automated diffraction tomography and precession technique. Ultramicroscopy 2009, 109, 758– 765,  DOI: 10.1016/j.ultramic.2009.01.011

Authors
*******

- Main Author PyFast-ADT: Marco Santucci - `orcid <https://orcid.org/0000-0002-7367-4343>`_
- Main Author Fast-ADT: Dr. Sergi Plana-Ruiz - `orcid <https://orcid.org/0000-0002-4047-8362>`_

Foundings
*********
This project is originally founded by the European Union’s H2020 ITN project NanED, grant agreement No. 956099 (M.S.).


Contributors
============
- Laura Gemmrich-Hernández - Script for pts2 file generation, user Guides, and software testing.


