import os
from shutil import copyfile
import progressbar
import numpy as np
import scipy.misc
import tqdm

low = 80
high = 250

def concatenate_segmentation_labels(segmentation_npy_path, out_path):
    # https://github.com/niessner/Matterport/blob/master/metadata/category_mapping.tsv
    table_labels = [15, 47, 73, 81, 89, 97, 177, 215, 216, 472, 474, 477, 661, 677, 678, 720, 732, 745, 756, 842]
    object_labels = [338, 349]

    segmentation = np.load(segmentation_npy_path)
    segmentation = segmentation[:, :, 2]
    new_segmentation = np.zeros(segmentation.shape, dtype=np.uint32)

    table_values = np.isin(segmentation, table_labels)
    object_values = np.isin(segmentation, object_labels)

    new_segmentation[table_values] = 1
    new_segmentation[object_values] = 2

    np.save(out_path, new_segmentation)


def npy_to_depth(depth_npy_path, out_path):
    # generates a png and saves into predesignated location
    depth = np.load(depth_npy_path)
    depth = depth.reshape(depth.shape[1], depth.shape[0])
    scipy.misc.imsave(out_path, depth)


def npy_to_rgb(rgb_npy_path, out_path):
    # generates a png and saves into predesignated location

    rgb = np.load(rgb_npy_path)
    rgb = rgb.reshape(rgb.shape[1], rgb.shape[0], -1)
    # save image
    rgb = rgb[:,:,:3]
    scipy.misc.imsave(out_path, rgb)


def generate_metadata(rgb_file_path, depth_file_path, labels_file_path, save_loc):
    """ Generate a json file containing the image dimensions, the number of
        classes and the med class frequency, and colours for every single class
        all saved in a json file in the desired save_location.
    Inputs:
        rgb_filepath (str): path to rgb images
        depth_file_path (str): path to depth images
        labels_file_path (str): path to labels
        save_loc (str): save location path
    """
    # calculate dimensions of images
    # To maximize efficiency, the function assumes that images are of the same
    # dimension
    with open(rgb_file_path, mode='r') as f:
        image_path = f.readline().strip()
        trial_image = scipy.misc.imread(image_path)
        assert isinstance(trial_image, np.ndarray), \
            'image should be loaded as np array'
        dimensions = trial_image.shape

    # number of classes
    class_count = {}
    with open(labels_file_path, mode='r') as f:
        for line in tqdm.tqdm(f):
            # load the labels
            assert '.npy' in line, 'the labels should be stored in .npy'
            labels = np.load(line.strip()).flatten()
            for element in labels:
                if element not in class_count:
                    class_count[element] = 1
                else:
                    class_count[element] += 1
    total_pixel_count = np.sum(list(class_count.values()))
    class_prob = {key: class_count[key] / float(total_pixel_count) \
                  for key in class_count}
    prob_median = np.median(list(class_prod.values()))
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

    json_dict = {'height': dimensions[0], 'width': dimensions[1],
                 'colours': colours, 'med_freq': med_freq_list,
                 'num_classes': len(class_count), 'class_prob': class_prob}

    with open(save_loc, 'w') as f:
        json.dump(json_dict, f)

    return json_dict


def migrate_examples(root_dir):
    trial_names = os.listdir(root_dir)
    index = 0

    rgb_npy_path = []
    depth_npy_path = []
    label_npy_path = []

    if not os.path.isdir(os.path.join(root_dir, 'gibson_data')):
        os.makedirs(os.path.join(root_dir, 'gibson_data'))
        os.makedirs(os.path.join(root_dir, 'gibson_data/rgb'))
        os.makedirs(os.path.join(root_dir, 'gibson_data/depth'))
        os.makedirs(os.path.join(root_dir, 'gibson_data/segmentation'))

    for trial_name in trial_names:
        if os.path.isdir(os.path.join(root_dir, trial_name)):
            samples = os.listdir(os.path.join(root_dir, trial_name))
            samples = sorted(samples)

            samples = [sample for sample in samples if ".npy" in sample]
            num_samples = len(samples)/3

            for i in progressbar.progressbar(range(num_samples)):
                try:
                    rgb_path = os.path.join(root_dir, trial_name, "{}_rgb.npy".format(i))
                    depth_path = os.path.join(root_dir, trial_name, "{}_depth.npy".format(i))
                    segmentation_path = os.path.join(root_dir, trial_name, "{}_segmentation.npy".format(i))

                    if os.path.isfile(rgb_path) and os.path.isfile(depth_path) and os.path.isfile(segmentation_path):
                        out_rgb_path = os.path.join(root_dir, "gibson_data", "rgb", "{}_rgb.png".format(index))
                        out_depth_path = os.path.join(root_dir, "gibson_data", "depth", "{}_depth.png".format(index))
                        out_segmentation_path = os.path.join(root_dir, "gibson_data", "segmentation", "{}_segmentation.npy".format(index))

                        #npy_to_rgb(rgb_path, out_rgb_path)
                        #npy_to_depth(depth_path, out_depth_path)
                        #concatenate_segmentation_labels(segmentation_path, out_segmentation_path)

                        rgb_npy_path.append(out_rgb_path)
                        depth_npy_path.append(out_depth_path)
                        label_npy_path.append(out_segmentation_path)

                        index += 1

                except Exception as e:
                    print(e)
                    break

    return rgb_npy_path, depth_npy_path, label_npy_path

if __name__=='__main__':
    root_dir = "."
    train_ratio = 0.8

    # adopting the way done in Rednet
    train_rgb_filepath = './gibson_data/img_dir_train.txt'
    train_depth_filepath = './gibson_data/depth_dir_train.txt'
    train_labels_filepath = './gibson_data/label_train.txt'

    test_rgb_filepath = './gibson_data/img_dir_test.txt'
    test_depth_filepath = './gibson_data/depth_dir_test.txt'
    test_labels_filepath = './gibson_data/label_test.txt'

    rgb_paths, depth_paths, label_paths = migrate_examples(root_dir)
    print(len(rgb_paths))
    rgb_paths = np.array(rgb_paths)
    print(len(rgb_paths))
    print(len(depth_paths))
    depth_paths = np.array(depth_paths)
    print(len(depth_paths))
    print(len(label_paths))
    label_paths = np.array(label_paths)
    print(len(label_paths))

    #assert len(rgb_paths) == len(depth_paths) \
    #        and len(depth_paths) == len(label_paths), \
    #        'list lengths should be the same'

    # create the training test split files
    train_idx = np.random.choice(len(depth_paths), int(train_ratio*len(depth_paths)), replace=False)
    test_idx = np.array(list(set(range(len(depth_paths))) - set(train_idx)))

    # writing into the training files
    with open(train_rgb_filepath, 'w') as f:
        paths = rgb_paths[train_idx]
        f.writelines(["%s \n" % item for item in paths])
    with open(train_depth_filepath, 'w') as f:
        paths = depth_paths[train_idx]
        f.writelines(["%s \n" % item for item in paths])
    with open(train_labels_filepath, 'w') as f:
        paths = label_paths[train_idx]
        f.writelines(["%s \n" % item for item in paths])
    # writing into the testing files

    with open(test_rgb_filepath, 'w') as f:
        paths = rgb_paths[test_idx]
        f.writelines(["%s \n" % item for item in paths])
    with open(test_depth_filepath, 'w') as f:
        paths = depth_paths[test_idx]
        f.writelines(["%s \n" % item for item in paths])
    with open(test_labels_filepath, 'w') as f:
        paths = label_paths[test_idx]
        f.writelines(["%s \n" % item for item in paths])
