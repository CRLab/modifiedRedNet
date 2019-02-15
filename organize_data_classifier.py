"""
    utils.py - Some convenience functions for using data from
        Joint 2D-3D-Semantic Data for Indoor Scene Understanding
           I. Armeni*, A. Sax*, A. Zamir, S. Savarese
        Website: 3dsemantics.stanford.edu
        Paper: https://arxiv.org/pdf/1702.01105.pdf

    Code Author: Alexander Sax

    Usage: For import only. i.e. 'import utils.py'
      Dependencies include scipy, OpenEXR
"""

import os
import progressbar
import numpy as np
import scipy.misc
import tqdm
import random
import json
import argparse
import yaml
import pathos.multiprocessing

low = 80
high = 250

label_dictionary = {}
label_name_to_index = None

# for Matterport 3D dataset
table_labels = [15, 47, 73, 81, 89, 97, 177, 215, 216, 472, 474, 477, 661, 677, 678, 720, 732, 745, 756, 842]
object_labels = [338, 349]

""" Semantics """


def get_index(color):
    """ Parse a color as a base-256 number and returns the index
    Args:
        color: A 3-tuple in RGB-order where each element \in [0, 255]
    Returns:
        index: an int containing the indec specified in 'color'
    """
    return color[0] * 256 * 256 + color[1] * 256 + color[2]


def get_color(index):
    """ Parse a 24-bit integer as a RGB color. I.e. Convert to base 256
    Args:
        index: An int. The first 24 bits will be interpreted as a color.
            Negative values will not work properly.
    Returns:
        color: A color s.t. get_index( get_color( i ) ) = i
    """
    b = index % 256  # least significant byte
    g = (index >> 8) % 256
    r = (index >> 16) % 256  # most significant byte
    return r, g, b


""" Label functions """


def load_labels(dataset_name, label_materials_file):
    """ Convenience function for loading JSON labels """
    # Apparently we cannot use the labels provided at their repo: https://github.com/alexsax/2D-3D-Semantics
    # Instead you have to use the labels as they appear in the `semantic.mtl` file within the dataset folder
    # So for example, trial `space7` has a `semantic.mtl` file, you'll load in the material names, then using those
    # indices assign them to a specific label, which you create a map for.
    global label_dictionary, label_name_to_index

    # If we have already loaded the labels for this dataset, return
    if dataset_name in label_dictionary:
        return

    # Read the materials one by one and index them
    materials = []
    with open(label_materials_file, 'r') as sem:
        for line in sem:
            line_split = line.split(" ")
            if line_split[0] == "newmtl":
                material_name = line_split[1].strip("\n")
                material_category = material_name.split("_")[0]
                materials.append(material_category)

    # If we have not yet defined an index for the label names:
    if label_name_to_index is None:
        label_name_to_index = {}
        unique_material_categories = np.unique(materials)
        for i in range(len(unique_material_categories)):
            label_name_to_index[unique_material_categories[i]] = i

    # Map the indices onto the names in the list
    label_dictionary[dataset_name] = np.array(map(lambda label_name: label_name_to_index[label_name], materials))


def parse_label(label):
    """ Parses a label into a dict """
    res = {}
    clazz, instance_num, room_type, room_num, area_num = label.split("_")
    res['instance_class'] = clazz
    res['instance_num'] = int(instance_num)
    res['room_type'] = room_type
    res['room_num'] = int(room_num)
    res['area_num'] = int(area_num)
    return res


""" EXR Functions """


def normalize_array_for_matplotlib(arr_to_rescale):
    """ Rescales an array to be between [0, 1]
    Args:
        arr_to_rescale:
    Returns:
        An array in [0,1] with f(0) = 0.5
    """
    return (arr_to_rescale / np.abs(arr_to_rescale).max()) / 2 + 0.5


def concatenate_segmentation_labels(segmentation_npy_path, out_path, is_stanford, dataset_name):
    global label_dictionary, table_labels, table_values
    # https://github.com/niessner/Matterport/blob/master/metadata/category_mapping.tsv

    segmentation = np.load(segmentation_npy_path)
    if is_stanford:
        R = segmentation[:, :, 0]
        G = segmentation[:, :, 1]
        B = segmentation[:, :, 2]
        pixel_labels = R * 256 * 256 + G * 256 + B
        height, width = pixel_labels.shape[0], pixel_labels.shape[1]
        pixel_labels = pixel_labels.reshape((height * width))

        labeled_pixels = label_dictionary[dataset_name][pixel_labels]
        labeled_pixels = np.reshape(labeled_pixels, (height, width))
        # unique, counts = np.unique(labeled_pixels, return_counts=True)

        np.save(out_path, labeled_pixels)

    else:
        segmentation = scipy.misc.imresize(segmentation, (256, 256))
        segmentation = segmentation[:, :, 2]
        new_segmentation = np.zeros(segmentation.shape, dtype=np.uint8)

        table_values = np.isin(segmentation, table_labels)
        # object_values = np.isin(segmentation, object_labels)

        new_segmentation[table_values] = 1
        # new_segmentation[object_values] = 2

        np.save(out_path, new_segmentation)


def npy_to_depth(depth_npy_path, out_path):
    # generates a png and saves into predesignated location
    depth = np.load(depth_npy_path)
    depth = depth.reshape(depth.shape[1], depth.shape[0])

    scaled_depth = depth.copy()
    depth = depth * 25
    depth[depth > 255.] = 255.
    depth = np.reshape(depth, (256, 256))
    scipy.misc.imsave(out_path, depth)

    # if "house" in out_path:
    #     depth = scipy.misc.imresize(depth, (256, 256))
    # scipy.misc.imsave(out_path, depth)


def npy_to_rgb(rgb_npy_path, out_path):
    # generates a png and saves into predesignated location

    rgb = np.load(rgb_npy_path)
    if "house" in out_path:
        rgb = scipy.misc.imresize(rgb, (256, 256))
    rgb = rgb.reshape(rgb.shape[1], rgb.shape[0], -1)
    # save image
    rgb = rgb[:, :, :3]
    scipy.misc.imsave(out_path, rgb)


def action_txt_to_class(action_txt_path):
    action_dict = yaml.load(open(action_txt_path, 'r'))
    angular = action_dict['angular']
    linear = action_dict['linear']
    
    if linear[0] > 0:
        return 'forward'
    if linear[0] < 0:
        return 'backward'
    if angular[2] > 0:
        return 'right'
    if angular[2] < 0:
        return 'left'
    return 'stop'


def generate_metadata(root_dir, rgb_file_path, depth_file_path, labels_file_path, save_loc):
    """ Generate a json file containing the image dimensions, the number of
        classes and the med class frequency, and colours for every single class
        all saved in a json file in the desired save_location.
    Inputs:
        rgb_filepath (str): path to rgb images
        depth_file_path (str): path to depth images
        labels_file_path (str): path to labels
        save_loc (str): save location path
    """
    global label_name_to_index

    # calculate dimensions of images
    # To maximize efficiency, the function assumes that images are of the same
    # dimension
    with open(rgb_file_path, mode='r') as rgbs:
        image_path = os.path.join(root_dir, rgbs.readline().strip())
        trial_image = scipy.misc.imread(image_path)
        assert isinstance(trial_image, np.ndarray), \
            'image should be loaded as np array'
        dimensions = trial_image.shape

    # number of classes
    class_count = {}
    with open(labels_file_path, mode='r') as labels:
        for line in tqdm.tqdm(labels):
            # load the labels
            assert '.npy' in line, 'the labels should be stored in .npy'
            labels = np.load(os.path.join(root_dir, line.strip())).flatten()
            for element in labels:
                if element not in class_count:
                    class_count[element] = 1
                else:
                    class_count[element] += 1
    total_pixel_count = np.sum(list(class_count.values()))
    class_prob = {str(key): class_count[key] / float(total_pixel_count) \
                  for key in class_count}
    prob_median = np.median(list(class_prob.values()))
    med_freq = {key: prob_median / float(class_prob[key]) for key in class_prob}
    med_freq_list = [x for _, x in sorted(zip(list(med_freq.keys()), list(med_freq.values())))]
    # get num_classes
    num_classes = len(class_count)
    # generate random colours, add on the original
    colours = [(0, 0, 0)]
    for i in range(num_classes):
        colours.append(
            (random.randint(low, high), random.randint(low, high),
             random.randint(low, high)))
    assert len(colours) == num_classes + 1

    json_dict = {
         'height': dimensions[0],
         'width': dimensions[1],
         'colours': colours,
         'med_freq': med_freq_list,
         'num_classes': len(class_count),
         'class_prob': class_prob,
         'label_to_index_dict': label_name_to_index
    }

    with open(save_loc, 'w') as labels:
        json.dump(json_dict, labels)

    return json_dict


def handle_job(
        (source_rgb_path, source_depth_path, source_segmentation_path, out_rgb_path, out_depth_path, out_segmentation_path, is_stanford, dataset_name)):
    npy_to_rgb(source_rgb_path, out_rgb_path)
    npy_to_depth(source_depth_path, out_depth_path)
    # concatenate_segmentation_labels(source_segmentation_path, out_segmentation_path, is_stanford, dataset_name)


def migrate_examples(root_dir, dataset_name, gibson_asset_dataset_path):
    trial_names = os.listdir(root_dir)

    rgb_npy_path = []
    depth_npy_path = []
    label_npy_path = []

    if not os.path.isdir(os.path.join(root_dir, 'data', dataset_name)):
        os.makedirs(os.path.join(root_dir, 'data', dataset_name))
        os.makedirs(os.path.join(root_dir, 'data', dataset_name, 'rgb'))
        os.makedirs(os.path.join(root_dir, 'data', dataset_name, 'depth'))
        os.makedirs(os.path.join(root_dir, 'data', dataset_name, 'segmentation'))

    # Build list of jobs
    jobs = []
    index = 0
    for trial_name in trial_names:
        if trial_name == "data":
            continue

        if os.path.isdir(os.path.join(root_dir, trial_name)):
            is_stanford = not trial_name.startswith("house")
            trial_name_truncated = trial_name.split("_")[0]
            if is_stanford:
                material_path = os.path.join(gibson_asset_dataset_path, trial_name_truncated, "semantic.mtl")
                load_labels(trial_name_truncated, material_path)

            samples = os.listdir(os.path.join(root_dir, trial_name))
            samples = sorted(samples)

            samples = [sample for sample in samples if ".npy" in sample]
            num_samples = len(samples) / 3

            for i in range(2, num_samples):
                source_rgb_path = os.path.join(root_dir, trial_name, "{}_rgb.npy".format(i))
                source_depth_path = os.path.join(root_dir, trial_name, "{}_depth.npy".format(i))
                source_segmentation_path = os.path.join(root_dir, trial_name, "{}_segmentation.npy".format(i))
                source_action_txt_path = os.path.join(root_dir, trial_name, "{}_action.yaml".format(i))

                if os.path.isfile(source_rgb_path) and os.path.isfile(source_depth_path) and os.path.isfile(source_segmentation_path) and os.path.isfile(source_action_txt_path):
                    class_name = action_txt_to_class(source_action_txt_path)

                    if not os.path.isdir(os.path.join(root_dir, 'data', dataset_name, "rgb", class_name)):
                        os.makedirs(os.path.join(root_dir, 'data', dataset_name, "rgb", class_name))
                        os.makedirs(os.path.join(root_dir, 'data', dataset_name, "depth", class_name))

                    rgb_root_path = os.path.join('data', dataset_name, "rgb", class_name, "{}_rgb.png".format(index))
                    depth_root_path = os.path.join('data', dataset_name, "depth", class_name, "{}_depth.png".format(index))
                    segmentation_root_path = os.path.join('data', dataset_name, "segmentation", class_name, "{}_segmentation.npy".format(index))

                    out_rgb_path = os.path.join(root_dir, rgb_root_path)
                    out_depth_path = os.path.join(root_dir, depth_root_path)
                    out_segmentation_path = os.path.join(root_dir, segmentation_root_path)

                    jobs.append((source_rgb_path, source_depth_path, source_segmentation_path, out_rgb_path,
                                 out_depth_path, out_segmentation_path, is_stanford, trial_name_truncated))

                    rgb_npy_path.append(rgb_root_path)
                    depth_npy_path.append(depth_root_path)
                    label_npy_path.append(segmentation_root_path)

                    index += 1

    # Process jobs
    bar = progressbar.ProgressBar(max_value=len(jobs))
    bar.update(0)
    pool = pathos.multiprocessing.Pool(processes=32)
    # for job in jobs:
    #     handle_job(job)
    #     bar += 1
    for _ in pool.imap_unordered(handle_job, jobs):
        bar += 1

    return rgb_npy_path, depth_npy_path, label_npy_path


def parse_args():
    parser = argparse.ArgumentParser(description='Compute completions for a given depth and tactile cloud dataset')
    parser.add_argument("root_dir", type=str, default=".", help="""Root directory of given trials of experiment""")
    parser.add_argument("train_ratio", type=float, default=0.8, help="""Ratio of training to test examples""")
    parser.add_argument("dataset_name", type=str, default="gibson_data", help="""Name of dataset""")
    parser.add_argument("gibson_asset_dataset_path", type=str,
                        default="/home/david/workspace/GibsonEnv/gibson/assets/dataset",
                        help="""path to Gibson environments""")

    args = parser.parse_args()

    return args


def save_paths(paths, indices, outfilepath):
    with open(outfilepath, 'w') as f:
        paths = paths[indices]
        f.writelines(["{}\n".format(item) for item in paths])


def main():
    args = parse_args()

    # adopting the way done in Rednet
    train_rgb_filepath = os.path.join(args.root_dir, 'data', args.dataset_name, 'img_dir_train.txt')
    train_depth_filepath = os.path.join(args.root_dir, 'data', args.dataset_name, 'depth_dir_train.txt')
    train_labels_filepath = os.path.join(args.root_dir, 'data', args.dataset_name, 'label_train.txt')

    test_rgb_filepath = os.path.join(args.root_dir, 'data', args.dataset_name, 'img_dir_test.txt')
    test_depth_filepath = os.path.join(args.root_dir, 'data', args.dataset_name, 'depth_dir_test.txt')
    test_labels_filepath = os.path.join(args.root_dir, 'data', args.dataset_name, 'label_test.txt')

    train_meta_filepath = os.path.join(args.root_dir, 'data', args.dataset_name, 'meta_train.json')
    test_meta_filepath = os.path.join(args.root_dir, 'data', args.dataset_name, 'meta_test.json')

    rgb_paths, depth_paths, label_paths = migrate_examples(args.root_dir, args.dataset_name, args.gibson_asset_dataset_path)
    rgb_paths = np.array(rgb_paths)
    depth_paths = np.array(depth_paths)
    # label_paths = np.array(label_paths)

    # create the training test split files
    train_idx = np.random.choice(len(depth_paths), int(args.train_ratio * len(depth_paths)), replace=False)
    test_idx = np.array(list(set(range(len(depth_paths))) - set(train_idx)))

    # writing into the training files
    save_paths(rgb_paths, train_idx, train_rgb_filepath)
    save_paths(depth_paths, train_idx, train_depth_filepath)
    # save_paths(label_paths, train_idx, train_labels_filepath)

    # writing into the testing files
    save_paths(rgb_paths, test_idx, test_rgb_filepath)
    save_paths(depth_paths, test_idx, test_depth_filepath)
    # save_paths(label_paths, test_idx, test_labels_filepath)

    # generate the metadata files
    # generate_metadata(args.root_dir, train_rgb_filepath, train_depth_filepath, train_labels_filepath,
    #                   train_meta_filepath)
    # generate_metadata(args.root_dir, test_rgb_filepath, test_depth_filepath, test_labels_filepath, test_meta_filepath)


if __name__ == '__main__':
    main()
