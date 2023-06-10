# Optimised Path Space Regularisation (OPSR)

Source Code for the paper "Optimised Path Space Regularisation" from EGSR 2021

![alt text](https://github.com/WeiPhil/portfolio/blob/master/src/data/opsr/beach_thumbnail.png)

## Overview

The source code is divided in three main folders : 
- `./mitsuba2_opsr` : The Mitsuba 2 implementation of OPSR. Be sure to check out the README in the folder for more details.

- `./opsr_optimisation` : Everything related to the optimisation process. The folder is itself divided in three subfolders:
    - `./opsr_optimisation/optimisation_script` : This is where the optimisation script is located and where the results of the optimisation will be stored after running the script. Be sure to check out the README in the folder for more details. To run the optimisation script you will need __python3__ at least, the __numpy__ package and the mitsuba python library generated at compilation and accessible after sourcing `./mitsuba2_opsr/setpath.sh`.

    - `./opsr_optimisation/optimisation_scenes` : This folder contains the scene description and references used for the optimisation process. If you want to add other scenes, there are a few requirements : all the scenes need to have the same film size, you will need to compute a clean reference of the scene and you need to wrap the main integrator in a `momentrgb` integrator to access the necessary second order moment estimates in an AOV (arbitrary output variable) during the optimisation. There is no other requirements and you can simply add the scenes to the list of processed scenes in the optimisation script (See `./opsr_optimisation/optimisation_script/opsr_path_interp_optim.py`)

    - `./opsr_optimisation/opsr_interp_data_final` : Contains the pre-optimised attenuation factors for the interpolated version of OPSR also used to generate all the renders in the paper.

    - `./opsr_optimisation/attenuation_lookup_utilities.py` : A python script that includes some helper functions to lookup or debug the optimised attenuation factors.

- `./opsr_paper_scenes_and_results` : Two of the open-source scenes used in the paper to reproduce some of our results. To run the scene files that use the Specular Manifold Sampling integrator from Zeltner et. al (2020) you will need their implementation available [here](https://github.com/tizian/specular-manifold-sampling)
