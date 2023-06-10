# Non mitsuba imports
import drjit.llvm.ad as dr_ad
import drjit.cuda.ad as dr_ad
import numpy as np
import os

# Mitsuba imports (make sure mitsuba is in the path by sourcing
# the setpath.sh file in the mitsuba folder after compilation)
import mitsuba as mi
import drjit as dr

# Make sure you have compiled mitsuba for the required 'cuda_ad_rgb' or 'llvm_ad_rgb' mode
mi.set_variant('cuda_ad_rgb')

mi.set_variant('llvm_ad_rgb')

""" Divides a main crop (usually just the whole film) into subcrops of equal size and return a string of the new xml """


def create_multi_sensor_xml(filename, original_size: np.array, subcrop_size: np.array, main_crop_size: np.array, main_crop_offset: np.array, filter='gaussian', pixel_format='rgb'):
    from lxml import etree as ET
    import copy

    num_sensors_xy = main_crop_size/subcrop_size
    assert np.all(num_sensors_xy == num_sensors_xy.astype(
        np.int32)), "Cannot tile the main_crop with that subcrop size"
    num_sensors_xy = num_sensors_xy.astype(np.int32)

    main_crop_size = main_crop_size.astype(np.int32)
    main_crop_offset = main_crop_offset.astype(np.int32)
    subcrop_size = subcrop_size.astype(np.int32)

    # define a 1d list of crops out of the 2d offset positions
    crop_offsets = [[main_crop_offset[0]+subcrop_size[0]*num_sensors_x, main_crop_offset[1]+subcrop_size[1]*num_sensors_y]
                    for num_sensors_y in range(num_sensors_xy[1])
                    for num_sensors_x in range(num_sensors_xy[0])]

    tree = ET.parse(filename)
    root = tree.getroot()
    parent_map = {c: p for p in tree.iter() for c in p}

    sensors = list(root.iterfind('sensor'))
    assert len(sensors) == 1, "Please specify only one sensor in the scene file."

    first_sensor = sensors[0]
    sensor_idx = list(parent_map).index(first_sensor)
    old_film = first_sensor.find('film')
    first_sensor.remove(old_film)
    template_sensor_no_film = copy.deepcopy(first_sensor)
    root.remove(first_sensor)

    for i, (subcrop_offset_x, subcrop_offset_y) in enumerate(crop_offsets):
        new_sensor = copy.deepcopy(template_sensor_no_film)
        film = ET.Element("film", type="hdrfilm")
        film.tail = '\n\t\t\t\t'
        ET.SubElement(film, "integer", name="width", value=str(
            original_size[0])).tail = '\n\t\t\t\t'
        ET.SubElement(film, "integer", name="height", value=str(
            original_size[1])).tail = '\n\t\t\t\t'
        ET.SubElement(film, "integer", name="crop_offset_x",
                      value=str(subcrop_offset_x)).tail = '\n\t\t\t\t'
        ET.SubElement(film, "integer", name="crop_offset_y",
                      value=str(subcrop_offset_y)).tail = '\n\t\t\t\t'
        ET.SubElement(film, "integer", name="crop_width",
                      value=str(subcrop_size[0])).tail = '\n\t\t\t\t'
        ET.SubElement(film, "integer", name="crop_height",
                      value=str(subcrop_size[1])).tail = '\n\t\t\t\t'
        ET.SubElement(film, "string", name="pixel_format",
                      value=pixel_format).tail = '\n\t\t\t\t'
        ET.SubElement(film, "rfilter", type=filter).tail = '\n\t\t\t'
        new_sensor.append(film)
        root.insert(sensor_idx+i, new_sensor)

    # uncomment to see output scene debug
    # tree.write('multisensor.xml')
    string = ET.tostring(tree).decode()
    return string


""" Helper to extract a crop from the entire film that corresponds to a specific sensor idx (as generated via the 'create_multi_sensor_xml' function) """


def cropped_ref_image(bitmap, sensor_idx):
    np_bmp = np.array(bitmap).astype(np.float32)
    x, y = sensor_idx % num_sensors_xy[0], sensor_idx // num_sensors_xy[0]
    start_offset = main_crop_offset[0]+subcrop_size[0] * \
        x, main_crop_offset[1]+subcrop_size[1]*y
    image_ref_crop = dr_ad.TensorXf(
        np_bmp[start_offset[1]:start_offset[1]+subcrop_size[1], start_offset[0]:start_offset[0]+subcrop_size[0], :3])
    return image_ref_crop


# sensor settings (corresponds to the size of the reference image)
original_size = np.array([256, 256])
main_crop_size = np.array([256, 256])
main_crop_offset = np.array([0, 0])
subcrop_size = np.array([32, 32])

# How many optimisation iterations should be performed (10'000 is usually enough to converge)
iterations = 10000

# If for some reason the optimisation process is stopped one can start from a checkpoint.
# At every iteration the learned attenuation factors are written to disk and can therefore
# be used to continue the optimisation without having to start again from 0.
# Start iter indicates the logical iteration and will name the outputs accordingly
# furthemore, setting it to anything but 0 will simply use the last saved checkpoint
# and continue the optimisation process from there
start_iter = 0

# optimization settings
# the number of samples per pixel to use at every iteration. Changing this and the depth
# both have big impact on gpu memory usage. Using 1024 spp with a max depth of 10 will require about !14GB!
# of gpu memory.
spp = 512
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
if not os.path.exists(work_folder + 'numpy_data'):
    os.makedirs(work_folder + 'numpy_data')
if not os.path.exists(work_folder + 'output'):
    os.makedirs(work_folder + 'output')
if not os.path.exists(work_folder + 'output_gathered'):
    os.makedirs(work_folder + 'output_gathered')

# Setting up which scenes are used for the optimisation
scene_folders = ['cbox_caustic_double_glass', 'cbox_caustic', 'rings']
scene_paths = ['cbox_caustic/cbox_caustic',
               #    'cbox_caustic/cbox_caustic_roughness_0_05',
               #    'cbox_caustic/cbox_caustic_roughness_0_15',
               # 'cbox_caustic/cbox_caustic_roughness_0_4',
               # 'cbox_caustic/cbox_caustic_roughness_0_6',
               #    'cbox_caustic_double_glass/cbox_caustic_double_glass',
               #    'cbox_caustic_double_glass/cbox_caustic_double_glass_roughness_0_05',
               #    'cbox_caustic_double_glass/cbox_caustic_double_glass_roughness_0_15',
               # 'cbox_caustic_double_glass/cbox_caustic_double_glass_roughness_0_4',
               # 'cbox_caustic_double_glass/cbox_caustic_double_glass_roughness_0_6',
               'rings/rings']

scene_ref_paths = ['cbox_caustic/references/cbox_caustic',
                   #    'cbox_caustic/references/cbox_caustic_roughness_0_05',
                   #    'cbox_caustic/references/cbox_caustic_roughness_0_15',
                   # 'cbox_caustic/references/cbox_caustic_roughness_0_4',
                   # 'cbox_caustic/references/cbox_caustic_roughness_0_6',
                   #    'cbox_caustic_double_glass/references/cbox_caustic_double_glass',
                   #    'cbox_caustic_double_glass/references/cbox_caustic_double_glass_roughness_0_05',
                   #    'cbox_caustic_double_glass/references/cbox_caustic_double_glass_roughness_0_15',
                   # 'cbox_caustic_double_glass/references/cbox_caustic_double_glass_roughness_0_4',
                   # 'cbox_caustic_double_glass/references/cbox_caustic_double_glass_roughness_0_6',
                   'rings/references/rings']

scenes_multi_sensor_xmls = []

# append scene folders to file file_resolver
for scene_folder in scene_folders:
    mi.Thread.thread().file_resolver().append(main_scene_folder + scene_folder)

# create multi_sensor scenes (for crops)
for scene_path in scene_paths:
    multi_sensor_scene_xml = create_multi_sensor_xml(main_scene_folder + scene_path + '.xml',
                                                     original_size, subcrop_size, main_crop_size, main_crop_offset, filter='gaussian', pixel_format='rgb')
    scenes_multi_sensor_xmls.append(multi_sensor_scene_xml)

scenes = []
for multi_sensor_scene_xml in scenes_multi_sensor_xmls:
    scene = mi.load_string(multi_sensor_scene_xml, spp=spp,
                           size=original_size[0], depth=depth, integrator=integrator)
    scenes.append(scene)

# assuming multi_sensor was successfully created
num_sensors_xy = (main_crop_size/subcrop_size).astype(np.int32)

num_sensors = dr.prod(num_sensors_xy)
num_scenes = len(scenes)

# Traverse scene and set differentiable parameters (assuming same parameters through the different scenes)
att_factors_5d_str = 'MomentRGBIntegrator.integrator_0.att_factors_5d'
att_factors_4d_str = 'MomentRGBIntegrator.integrator_0.att_factors_4d'
att_factors_3d_str = 'MomentRGBIntegrator.integrator_0.att_factors_3d'
att_factors_2d_str = 'MomentRGBIntegrator.integrator_0.att_factors_2d'

if start_iter == 0:
    print("Starting a new optimisation process.")
    params_scenes = []
    for scene in scenes:
        params = mi.traverse(scene)
        params.keep([att_factors_2d_str, att_factors_3d_str,
                    att_factors_4d_str, att_factors_5d_str])
        att_factors_5d = np.ones(
            (roughness_res, roughness_res, roughness_res, roughness_res, roughness_res)).flatten() - 0.6
        att_factors_4d = np.ones(
            (roughness_res, roughness_res, roughness_res, roughness_res)).flatten() - 0.6
        att_factors_3d = np.ones(
            (roughness_res, roughness_res, roughness_res)).flatten() - 0.6
        att_factors_2d = np.ones(
            (roughness_res, roughness_res)).flatten() - 0.6

        # last vertex diffuse, don't add any bias, easy connection
        att_factors_2d.reshape([roughness_res, roughness_res])[-1, :] = 0
        att_factors_3d.reshape(
            [roughness_res, roughness_res, roughness_res])[-1, :, :] = 0
        att_factors_4d.reshape(
            [roughness_res, roughness_res, roughness_res, roughness_res])[-1:, :, :] = 0
        att_factors_5d.reshape([roughness_res, roughness_res, roughness_res,
                               roughness_res, roughness_res])[-1:, :, :, :, :] = 0

        params[att_factors_5d_str] = att_factors_5d.flatten()
        params[att_factors_4d_str] = att_factors_4d.flatten()
        params[att_factors_3d_str] = att_factors_3d.flatten()
        params[att_factors_2d_str] = att_factors_2d.flatten()
        params.update()
        params_scenes.append(params)
else:
    print("Recovering from checkpoint!")
    # we recover the attenuation factors from the last iterations
    att_factors_5d = np.load(
        'numpy_data/opsr_path_interp-att_factors_5d-beta_{}_spp_{}.npy'.format(beta, spp))[-1]
    att_factors_4d = np.load(
        'numpy_data/opsr_path_interp-att_factors_4d-beta_{}_spp_{}.npy'.format(beta, spp))[-1]
    att_factors_3d = np.load(
        'numpy_data/opsr_path_interp-att_factors_3d-beta_{}_spp_{}.npy'.format(beta, spp))[-1]
    att_factors_2d = np.load(
        'numpy_data/opsr_path_interp-att_factors_2d-beta_{}_spp_{}.npy'.format(beta, spp))[-1]

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

# Load the reference images
ref_bitmaps = []
for scene_ref_path in scene_ref_paths:
    ref_bitmap = mi.Bitmap(main_scene_folder +
                           scene_ref_path + '_{}.exr'.format(original_size[0]))
    ref_bitmaps.append(ref_bitmap)

# Helper to add the result of one iteration (one crop) to the full output image
# This is not necessary but handy to see the optimisation progress.


def add_to_gathered(gathered, dr_image, sensor_idx):
    x, y = sensor_idx % num_sensors_xy[0], sensor_idx // num_sensors_xy[0]
    start_offset = main_crop_offset[0]+subcrop_size[0] * \
        x, main_crop_offset[1]+subcrop_size[1]*y
    gathered[start_offset[1]:start_offset[1]+subcrop_size[1], start_offset[0]:start_offset[0]+subcrop_size[0], :3] = np.array(dr_image).reshape((*subcrop_size, 3))


# We write some info to a log file at every iteration.
with open('optimisation_log.txt', 'w') as log_file:
    log_file.write("Params:\n")
    log_file.write("--------------------------\n")
    log_file.write('integrator :' + str(integrator)+"\n")
    log_file.write('original_size :' + str(original_size)+"\n")
    log_file.write('main_crop_size :' + str(main_crop_size)+"\n")
    log_file.write('main_crop_offset :' + str(main_crop_offset)+"\n")
    log_file.write('subcrop_size :' + str(subcrop_size)+"\n")
    log_file.write('iterations :' + str(iterations)+"\n")
    log_file.write('spp :' + str(spp)+"\n")
    log_file.write('beta :' + str(beta)+"\n")
    log_file.write('depth :' + str(depth)+"\n")
    log_file.write('roughness_res :' + str(roughness_res)+"\n")
    log_file.write("---------------------------\n")


def run_optimisation(start_iter=0, end_iter=iterations, opt=None):
    # Assume we can load the old data to continue from a checkpoint
    if start_iter != 0:
        param_data_5d = list(np.load(
            'numpy_data/opsr_path_interp-att_factors_5d-beta_{}_spp_{}.npy'.format(beta, spp)))
        param_data_4d = list(np.load(
            'numpy_data/opsr_path_interp-att_factors_4d-beta_{}_spp_{}.npy'.format(beta, spp)))
        param_data_3d = list(np.load(
            'numpy_data/opsr_path_interp-att_factors_3d-beta_{}_spp_{}.npy'.format(beta, spp)))
        param_data_2d = list(np.load(
            'numpy_data/opsr_path_interp-att_factors_2d-beta_{}_spp_{}.npy'.format(beta, spp)))
        loss_data_sensors, image_loss_data_sensors, var_data_sensors = np.load(
            'numpy_data/opsr_path_interp-losses-beta_{}_spp_{}.npy'.format(beta, spp))
    else:
        param_data_5d = []
        param_data_4d = []
        param_data_3d = []
        param_data_2d = []
        loss_data_sensors = - \
            np.ones((len(scenes), end_iter - start_iter, num_sensors))
        image_loss_data_sensors = - \
            np.ones((len(scenes), end_iter - start_iter, num_sensors))
        var_data_sensors = - \
            np.ones((len(scenes), end_iter - start_iter, num_sensors))

    # Inititialise the gathered images where we will store the latest iteration result (for visualisation only)
    gathered_rgb_images = []
    gathered_rgb_m2_images = []
    for scene in scenes:
        gathered_rgb = np.zeros((main_crop_size[1], main_crop_size[0], 3))
        gathered_rgb_m2 = np.zeros((main_crop_size[1], main_crop_size[0], 3))
        gathered_rgb_images.append(gathered_rgb)
        gathered_rgb_m2_images.append(gathered_rgb_m2)

    output_folder = 'output/'
    gathered_output_folder = 'output_gathered/'

    prev_scene_idx = 0

    # initialise scene and sensor indexes using permutations
    sensor_indexes = []
    for num_scene in range(num_scenes):
        sensor_indexes.append(
            list(np.random.permutation(np.arange(num_sensors))))

    scene_indexes = list(np.random.permutation((np.arange(num_scenes))))

    for i in range(0, end_iter - start_iter):
        print("Optimisation Progress : {} from {}".format(
            start_iter + i, end_iter), end="\r")

        with open('optimisation_log.txt', 'a') as log_file:
            log_file.write("***************\n")

        # Get next scene and sensor index
        if not scene_indexes:
            scene_indexes = list(
                np.random.permutation((np.arange(num_scenes))))

        scene_idx = scene_indexes.pop()
        if not sensor_indexes[scene_idx]:
            sensor_indexes[scene_idx] = list(
                np.random.permutation((np.arange(num_sensors))))

        sensor_idx = sensor_indexes[scene_idx].pop()

        # initialise optimiser for the first time
        if (opt == None):
            opt = mi.ad.Adam(lr=0.0005)
            opt[att_factors_5d_str] = params_scenes[prev_scene_idx][att_factors_5d_str]
            opt[att_factors_4d_str] = params_scenes[prev_scene_idx][att_factors_4d_str]
            opt[att_factors_3d_str] = params_scenes[prev_scene_idx][att_factors_3d_str]
            opt[att_factors_2d_str] = params_scenes[prev_scene_idx][att_factors_2d_str]
            params_scenes[scene_idx].update(opt)
        else:
            params_scenes[scene_idx][att_factors_5d_str] = params_scenes[prev_scene_idx][att_factors_5d_str]
            params_scenes[scene_idx][att_factors_4d_str] = params_scenes[prev_scene_idx][att_factors_4d_str]
            params_scenes[scene_idx][att_factors_3d_str] = params_scenes[prev_scene_idx][att_factors_3d_str]
            params_scenes[scene_idx][att_factors_2d_str] = params_scenes[prev_scene_idx][att_factors_2d_str]
            params_scenes[scene_idx].update(opt)

        crop_size = scenes[scene_idx].sensors()[sensor_idx].film().crop_size()

        image = mi.render(
            scenes[scene_idx], params=params_scenes[scene_idx], spp=spp, sensor=int(sensor_idx))

        # recompose the rgb image from the corresponding channels.
        aov_rgb = dr.zeros(dr_ad.TensorXf, (*crop_size, 3))
        aov_rgb[:, :, 0] = image[:, :, 0::9].array
        aov_rgb[:, :, 1] = image[:, :, 1::9].array
        aov_rgb[:, :, 2] = image[:, :, 2::9].array

        # recompose the 2nd moment rgb image from the corresponding channels
        aov_rgb_m2 = dr.zeros(dr_ad.TensorXf, (*crop_size, 3))
        aov_rgb_m2[:, :, 0] = image[:, :, 6::9].array
        aov_rgb_m2[:, :, 1] = image[:, :, 7::9].array
        aov_rgb_m2[:, :, 2] = image[:, :, 8::9].array

        # Add each crop to the gathered image for visualisation
        add_to_gathered(gathered_rgb_images[scene_idx], aov_rgb, sensor_idx)
        add_to_gathered(
            gathered_rgb_m2_images[scene_idx], aov_rgb_m2, sensor_idx)

        # We compute our variance-aware loss as described in Equation 16 of the paper
        image_ref = cropped_ref_image(ref_bitmaps[scene_idx], sensor_idx)
        variance = aov_rgb_m2 - dr.sqr(aov_rgb)
        image_loss = dr.sum(dr.abs(image_ref - aov_rgb)) / len(image_ref)
        var_loss = dr.sum(variance)/len(variance)
        mean_energy_rcp = dr.rcp(dr.sum(image_ref) / len(image_ref))

        loss = image_loss*mean_energy_rcp + beta * var_loss

        # Write all the intermediary results to disk (the 'output' folder contains the output of each progression while output_gathered)
        mi.Bitmap(array=aov_rgb).write(work_folder +
                                       output_folder + 'out_{}.exr'.format(i + start_iter))
        mi.Bitmap(array=variance).write(work_folder +
                                        output_folder + 'var_out_{}.exr'.format(i + start_iter))
        mi.Bitmap(array=dr_ad.TensorXf(gathered_rgb_images[scene_idx])).write(
            'last_gathered_rgb_{}.exr'.format(scene_idx))
        mi.Bitmap(array=dr_ad.TensorXf(gathered_rgb_images[scene_idx])).write(
            work_folder + gathered_output_folder + 'gathered_rgb_{}_{}.exr'.format(scene_idx, i+start_iter))
        mi.Bitmap(array=dr_ad.TensorXf(gathered_rgb_m2_images[scene_idx])).write(
            work_folder + gathered_output_folder + 'gathered_rgb_m2_{}.exr'.format(scene_idx))

        with open('optimisation_log.txt', 'a') as log_file:
            log_file.write("Scene index : %i\n" % scene_idx)
            log_file.write("Sensor index : %i\n" % sensor_idx)
            log_file.write("Loss at iteration %i : %f\n" % (i, loss[0]))
            log_file.write("Image error at iteration %i : %f\n" %
                           (i, image_loss[0]))
            log_file.write("Variance error at iteration %i : %f\n" %
                           (i, var_loss[0]))

        loss_data_sensors[scene_idx, i, sensor_idx] = loss[0]
        image_loss_data_sensors[scene_idx, i, sensor_idx] = image_loss[0]
        var_data_sensors[scene_idx, i, sensor_idx] = var_loss[0]

        # Back-propagate errors to input parameters
        dr.backward(loss)

        with open('optimisation_log.txt', 'a') as log_file:
            for key, value in params_scenes[scene_idx].items():
                log_file.write('Gradient for ' + key + ' is ' +
                               str(dr.grad(params_scenes[scene_idx][key]))+'\n')

        # optimiser: take a gradient step
        opt.step()

        with open('optimisation_log.txt', 'a') as log_file:
            for key, value in params_scenes[scene_idx].items():
                if (dr.any(dr.isnan(dr.grad(params_scenes[scene_idx][key]))) == True):
                    log_file.write(
                        str(dr.grad(params_scenes[scene_idx][key])) + '\n')
                    log_file.write('/!\\ Gradient for ' + key +
                                   ' is nan, setting it to zero\n' % i)
                    dr.set_grad(params_scenes[scene_idx][key], [0, 0, 0])
                if (dr.any(params_scenes[scene_idx][key] < 0.005)):
                    params_scenes[scene_idx][key] = dr.maximum(
                        params_scenes[scene_idx][key], 0.005)
                    log_file.write("/!\\ Clamped the value of " + key +
                                   ' to ' + str(params_scenes[scene_idx][key]) + '\n')
                    params_scenes[scene_idx].update()
                if (dr.any(params_scenes[scene_idx][key] > 0.999)):
                    params_scenes[scene_idx][key] = dr.minimum(
                        params_scenes[scene_idx][key], 0.999)
                    log_file.write("/!\\ Value higher than 1 " + key +
                                   ' to ' + str(params_scenes[scene_idx][key]) + '\n')
                    params_scenes[scene_idx].update()
                if (dr.any(dr.isnan(params_scenes[scene_idx][key]))):
                    log_file.write(
                        "/!\\ Encountered nan! Replaced by initial value \n")
                    params_scenes[scene_idx][key][dr.isnan(
                        params_scenes[scene_idx][key])] = 0.4
                    params_scenes[scene_idx].update()

        param_data_5d.append(
            np.copy(params_scenes[scene_idx][att_factors_5d_str]))
        param_data_4d.append(
            np.copy(params_scenes[scene_idx][att_factors_4d_str]))
        param_data_3d.append(
            np.copy(params_scenes[scene_idx][att_factors_3d_str]))
        param_data_2d.append(
            np.copy(params_scenes[scene_idx][att_factors_2d_str]))
        # Compare iterate against ground-truth value
        # err_ref = dr.hsum(dr.sqr(param_ref - params_scenes[scene_idx]['water.alpha.value']))
        with open('optimisation_log.txt', 'a') as log_file:
            for key, value in params_scenes[scene_idx].items():
                log_file.write('New params: ' + key +
                               ' : ' + str(value) + '\n')

        # save at each iteration to easilly continue or backup if needed
        np.save(
            'numpy_data/opsr_path_interp-att_factors_5d-beta_{}_spp_{}'.format(beta, spp), param_data_5d)
        np.save(
            'numpy_data/opsr_path_interp-att_factors_4d-beta_{}_spp_{}'.format(beta, spp), param_data_4d)
        np.save(
            'numpy_data/opsr_path_interp-att_factors_3d-beta_{}_spp_{}'.format(beta, spp), param_data_3d)
        np.save(
            'numpy_data/opsr_path_interp-att_factors_2d-beta_{}_spp_{}'.format(beta, spp), param_data_2d)

        np.save('numpy_data/opsr_path_interp-losses-beta_{}_spp_{}'.format(beta,
                spp), [loss_data_sensors, image_loss_data_sensors, var_data_sensors])
        # update old scene idx
        prev_scene_idx = scene_idx

    return opt, loss_data_sensors, image_loss_data_sensors, var_data_sensors, param_data_5d, param_data_4d, param_data_3d, param_data_2d


# Here is where we run the optimisation process
# We output the important quantities of the optimisation process for further processing, e.g for plotting or post-processing of the learned data
optimiser, loss_data_sensors, image_loss_data_sensors, var_data_sensors, param_data_5d, param_data_4d, param_data_3d, param_data_2d = run_optimisation(
    start_iter=start_iter, end_iter=start_iter + iterations)

print("Optimisation finished!")
