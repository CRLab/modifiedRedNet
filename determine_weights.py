import os
from scipy.misc import imread, imsave
import numpy as np
import random
import json
import matplotlib as plt

import argparse

parser = argparse.ArgumentParser(description='RedNet Indoor Sementic Segmentation')
parser.add_argument('--data-dir', default=None, metavar='DIR',
                    help='path to dataset')

args = parser.parse_args()

img_dir_train_file = 'img_dir_train.txt'
depth_dir_train_file = 'depth_dir_train.txt'
label_dir_train_file = 'label_train.txt'
img_dir_test_file = 'img_dir_test.txt'
depth_dir_test_file = 'depth_dir_test.txt'
label_dir_test_file = 'label_test.txt'


def generate_metadata(data_dir=None):
    # number of classes
    class_count = {}
    with open(data_dir + label_dir_train_file, mode='r') as f:
        for line in f.readlines():
            path = data_dir + line.replace("data/gibson_data/", "")
            # load the labels
            assert '.npy' in path, 'the labels should be stored in .npy'
            labels = np.load(path.strip()).flatten()
            for element in labels:
                if element not in class_count:
                    class_count[element] = 1
                else:
                    class_count[element] += 1
    total_pixel_count = np.sum(list(class_count.values()))
    class_prob = {key: class_count[key]/float(total_pixel_count) \
                    for key in class_count}
    prob_median = np.median(list(class_prob.values()))
    med_freq = {key: prob_median/float(class_prob[key]) for key in class_prob}
    #med_freq_list = [x for _,x in sorted(zip(list(med_freq.keys()), list(med_freq.values())))]
    print("med freqs: %s" % med_freq)    

    weights = {key: 1/float(class_prob[key]) for key in class_prob}
    print("skew weights: %s" % weights)

    colors = {}
    for i in range(len(weights)):
        colors[i] = plt.colors.hsv_to_rgb((np.random.randint(0, 255)/255., 150/255., 150/255. )) * 255

    print("colors: %s" % colors)

if __name__=='__main__':
    generate_metadata(args.data_dir)
