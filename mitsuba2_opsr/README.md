# Mitsuba 2 implementation for Optimised Path Space Regularisation (OPSR)

## Compilation and Supported Mitsuba 2 Versions

We only provide the additional/changed files needed to run our technique. The base source code for Mitsuba 2 can be found [here](https://github.com/mitsuba-renderer/mitsuba2). You can then replaces/add the files we provide in this folder and compile the Mitsuba system as usual.

We tested our implementation with **Mitsuba 2.1.1** in the following modes : _gpu_autodiff_rgb_ (used for the optimisation process), _scalar_rgb_, _scalar_spectral_ and _packet_rgb_. Other versions/modes might work with minimal changes.

For more details on how to compile and run Mitsuba 2 and a list of all plugins we refer to the [official documenation](https://mitsuba2.readthedocs.io/en/latest/src/getting_started/intro.html).

## Overview

We implement three main integrator plugins : 
- `./src/integrators/path_opsr_interp.cpp` is only suited for the optimisation process as it only contains placeholders for the attenuation factors that can be optimised. 

- `./src/integrators/path_opsr_interp_learnt.cpp` and `./src/integrators/path_opsr_learnt.cpp` use pre-optimised attenuation factors that can be selected via the `roughening_mode` option to achieve different bias-variance trade-offs. The _interp_ keyword indicates that the attenuation factors lookup are interpolated during runtime and prevents any possible visual artifacts due to discontinuities in the discretised roughness. In practice we never witnessed any problems using the non-interpolated attenuation factors. On the other hand, the optimisation process's robustness is greatly improved using the interpolation method and we strongly advise to use this variant even if you want to use the non-interpolated variant later for rendering.

All integrators rely on a simple data-structure to keep track of the roughnesses encountered along a path and compute the accumulated roughness and lookup attenuation factors. This is implemented in
- `./include/render/roughening/opsr_path.h`

The learnt attenuation factors used in our papers can also be found in 

- `./include/mitsuba/render/roughening/opsr_data.h`

Some important changes have been made to the material interface defined in `./include/mitsuba/bsdf.h` to allow the sampling and evaluation of materials with an updated roughness. To use OPSR every material needs to expose a roughness parameter through the `get_roughness` function as well as implement the `*_rough` variant of `pdf`,`eval` and `sample` which take as an additional parameter the accumulated roughness in the path.