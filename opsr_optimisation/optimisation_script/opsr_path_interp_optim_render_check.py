# Non mitsuba imports
import numpy as np
import os

# Mitsuba imports (make sure mitsuba is in the path by sourcing
# the setpath.sh file in the mitsuba folder after compilation)
import mitsuba as mi
import drjit as dr

mi.set_variant('scalar_rgb')

render_size = 512

# render settings

spp = 128

# optim settings
# the number of samples per pixel to use at every iteration. Changing this and the depth
# both have big impact on gpu memory usage. Using 1024 spp with a max depth of 10 will require about !14GB!
# of gpu memory.
optim_spp = 512
# The hyperparameter used to get different variance/bias tradeoffs (see Equation 16 in the paper)
beta = 0.001
# using a maximal depth of 10 is necessary to avoid extreme GPU memory usage and still
# allows to generalise well
depth = 10

# the Mitsuba2 integrator plugin used for the optimisation process (which implements opsr)
integrator = "path_opsr_interp"
# the number of bins used to classify roughness (you shouldn't have to change this)
roughness_res = 4

main_scene_folder = '../optimisation_scenes/'
work_folder = './'

# Create any needed output directories
if not os.path.exists(work_folder + 'render_check'):
    os.makedirs(work_folder + 'render_check')

# Setting up which scenes are used for the optimisation
scene_folders = ['cbox_caustic_double_glass', 'cbox_caustic', 'rings']
scene_paths = ['cbox_caustic/cbox_caustic',
               'cbox_caustic/cbox_caustic_roughness_0_05',
               'cbox_caustic/cbox_caustic_roughness_0_15',
               #    'cbox_caustic/cbox_caustic_roughness_0_4',
               #    'cbox_caustic/cbox_caustic_roughness_0_6',
               'cbox_caustic_double_glass/cbox_caustic_double_glass',
               'cbox_caustic_double_glass/cbox_caustic_double_glass_roughness_0_05',
               'cbox_caustic_double_glass/cbox_caustic_double_glass_roughness_0_15',
               # 'cbox_caustic_double_glass/cbox_caustic_double_glass_roughness_0_4',
               # 'cbox_caustic_double_glass/cbox_caustic_double_glass_roughness_0_6',
               'rings/rings']

scene_ref_paths = ['cbox_caustic/references/cbox_caustic',
                   'cbox_caustic/references/cbox_caustic_roughness_0_05',
                   'cbox_caustic/references/cbox_caustic_roughness_0_15',
                   # 'cbox_caustic/references/cbox_caustic_roughness_0_4',
                   # 'cbox_caustic/references/cbox_caustic_roughness_0_6',
                   #    'cbox_caustic_double_glass/references/cbox_caustic_double_glass',
                   'cbox_caustic_double_glass/references/cbox_caustic_double_glass_roughness_0_05',
                   #    'cbox_caustic_double_glass/references/cbox_caustic_double_glass_roughness_0_15',
                   # 'cbox_caustic_double_glass/references/cbox_caustic_double_glass_roughness_0_4',
                   # 'cbox_caustic_double_glass/references/cbox_caustic_double_glass_roughness_0_6',
                   'rings/references/rings']

scenes_xmls = []

# append scene folders to file file_resolver
for scene_folder in scene_folders:
    mi.Thread.thread().file_resolver().append(main_scene_folder + scene_folder)

scenes = []
for scene_path in scene_paths:
    scene = mi.load_file(main_scene_folder + scene_path + '.xml',
                         spp=spp, size=render_size, depth=depth, integrator=integrator)
    scenes.append(scene)

num_scenes = len(scenes)

# Traverse scene and set differentiable parameters (assuming same parameters through the different scenes)
att_factors_5d_str = 'MomentRGBIntegrator.integrator_0.att_factors_5d'
att_factors_4d_str = 'MomentRGBIntegrator.integrator_0.att_factors_4d'
att_factors_3d_str = 'MomentRGBIntegrator.integrator_0.att_factors_3d'
att_factors_2d_str = 'MomentRGBIntegrator.integrator_0.att_factors_2d'

print("Using optimised data from last iteration")
# we recover the attenuation factors from the last iterations
att_factors_5d = np.load(
    'numpy_data/opsr_path_interp-att_factors_5d-beta_{}_spp_{}.npy'.format(beta, optim_spp))[-1]
att_factors_4d = np.load(
    'numpy_data/opsr_path_interp-att_factors_4d-beta_{}_spp_{}.npy'.format(beta, optim_spp))[-1]
att_factors_3d = np.load(
    'numpy_data/opsr_path_interp-att_factors_3d-beta_{}_spp_{}.npy'.format(beta, optim_spp))[-1]
att_factors_2d = np.load(
    'numpy_data/opsr_path_interp-att_factors_2d-beta_{}_spp_{}.npy'.format(beta, optim_spp))[-1]

params_scenes = []
for scene in scenes:
    params = mi.traverse(scene)
    params.keep([att_factors_2d_str, att_factors_3d_str,
                att_factors_4d_str, att_factors_5d_str])
    params[att_factors_5d_str] = att_factors_5d
    params[att_factors_4d_str] = att_factors_4d
    params[att_factors_3d_str] = att_factors_3d
    params[att_factors_2d_str] = att_factors_2d
    params.update()
    params_scenes.append(params)

for i in range(num_scenes):
    print("Rendering " + scene_paths[i] + "\n")
    image = mi.render(scenes[i], params=params_scenes[i], spp=spp)
    mi.Bitmap(image[:, :, 0:3]).write(work_folder +
                                      'render_check/' + 'render_check_scene_{}.exr'.format(i))
