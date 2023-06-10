# Optimisation Process for Optimised Path Space Regularisation (OPSR)

## Optimising the attenuation factors

First of all, make sure you compiled Mitsuba 2 in the `gpu_autodiff_rgb` mode and setup the path by running `source setpath.sh` in the mitsuba 2 root folder. This allows us to use Mitsuba and Enoki as python libraries and run the renderer directly from the python script.

You will also need a recent version of _python3_ with the **numpy** and **lxml** package.

Then, from this directory, simply run
```
python3 opsr_path_interp_optim.py
```
this will perform a stochastic gradient descent and at every iteration : compute our variance-aware loss, backpropagate the gradients to the input parameters and update the optimised parameters.

The learned attenuation factors are stored as numpy arrays under `./numpy_data/` at every iteration. `./output/` and `./output_gathered/` contain the rendered image at every iteration and allow us to visualise the optimisation progress. At the root folder, the last gathered output (each crop's last render gathered in a single image) for every scene in the dataset is stored and gives an approximate render of what a render would look like if using the parameters of the last iteration. 

For more information on how to change the optimisation parameter, how the optimised data is saved and loaded from disk, how to use checkpoints and other implementation details, please checkout the comments in the script directly.