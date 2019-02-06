import argparse
import torch
import imageio
import skimage.transform
import torchvision
import numpy as np

import torch.optim
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
import torchvision.transforms as transforms
import RedNet_model
import RedNet_data
from utils import utils
from utils.utils import load_ckpt

parser = argparse.ArgumentParser(description='RedNet Indoor Sementic Segmentation')
parser.add_argument('--data-dir', default=None, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch-size', default=5, type=int,
                    metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-o', '--output', default=None, metavar='DIR',
                    help='path to output')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480
def inference():

    model = RedNet_model.RedNet(pretrained=True)
    load_ckpt(model, None, args.last_ckpt, device)
    model.eval()
    model.to(device)

    test_data = RedNet_data.GibsonDataset(transform=transforms.Compose([RedNet_data.scaleNorm(),
                                                                        RedNet_data.ToTensor(),
                                                                        RedNet_data.Normalize()]),
                                           phase_train=False,
                                           data_dir=args.data_dir)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=False)
    
    n_saved = 0

    for batch_idx, sample in enumerate(test_loader):
        if n_saved > 2500:
            exit()

        image = sample['image'].to(device)
        depth = sample['depth'].to(device)
        label = sample['label'].to(device)

#        pred = model(image, depth)

        for i in range(image.shape[0]):
            n_saved += 1
            print("%s: %s, [%s]" % (n_saved,  "has_table" if 1 in label[i].numpy() else "no_table", np.unique(label[i].numpy())))
 #           output = utils.color_label(torch.max(pred[i], 0)[1] + 1)
            gt_output = utils.color_label(label[i] + 1)
            imageio.imsave(args.output + "%s-%s.png" % (n_saved, "has_table" if 1 in label[i].numpy() else "no_table"), image[i].numpy().transpose(1, 2, 0))
            imageio.imsave(args.output + "%s-%s-gt.png" % (n_saved, "has_table" if 1 in label[i].numpy() else "no_table"), gt_output.numpy()[0].transpose((1, 2, 0)))
#            imageio.imsave(args.output + "%s-%s-seg.png" % (n_saved, i), output.cpu().numpy()[0].transpose((1, 2, 0)))



    # for batch_idx, sample in enumerate(test_loader):
    #     image = sample['image'].to(device)
    #     depth = sample['depth'].to(device)
    #     pred = model(image, depth)
        

if __name__ == '__main__':
    inference()
