import os
import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import fnmatch
import numpy as np
import cv2
import re
from model.networks_chord import Generator
from utils.tools import get_config, test_bbox, random_bbox, mask_image, is_image_file, default_loader, normalize, get_model_list


parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/test.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, default = '2019', help='manual seed')
parser.add_argument('--image', type=str)
parser.add_argument('--mask_dir', type=str, default='new_mask_512_680/')
parser.add_argument('--output_dir', type=str, default='output_new_mask/')
# parser.add_argument('--input_dir', type=str, default='input_seminar/')
parser.add_argument('--flow', type=str, default='')
parser.add_argument('--checkpoint_path', type=str, default='/home/nadim/Code_Ground/local_global_attention/checkpoints/Places/hole_benchmark')
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--test_root', type=str, default='/home/nadim/Dataset/Val_places')

def dataset_files(rootdir, pattern):
    """Returns a list of all image files in the given directory"""

    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    matches.sort(key=natural_keys)

    return matches

def sort(filename):
    num_array = sorted([str(y) for y in [x.split('.')[0] for x in filename]])
    return np.array([str(x)+'.jpg' for x in num_array])



def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]




def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    print("Arguments: {}".format(args))

    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    print("Configuration: {}".format(config))

    try:  # for unexpected error logging
        with torch.no_grad():   # enter no grad context
            file = dataset_files(args.test_root, "*.jpg")
            mask_file = dataset_files(args.mask_dir, "*.png")
            for j in range(len(mask_file)):
                for i in range(len(file)):
                    if is_image_file(file[i]):
                        if mask_file and is_image_file(mask_file[j]):
                            # Test a single masked image with a given mask
                            x = default_loader(file[i])
                            mask = default_loader(mask_file[j])
                            # x = cv2.cvtColor(cv2.imread(file[i]), cv2.COLOR_BGR2RGB)
                            # mask = cv2.cvtColor(cv2.imread(mask_file[j]), cv2.COLOR_BGR2RGB)
                            # x = cv2.resize(x, (config['image_shape'][0], config['image_shape'][1]))
                            # mask = cv2.resize(mask, (config['image_shape'][0], config['image_shape'][1]))
                            x = transforms.Resize(config['image_shape'][:-1])(x)
                            x = transforms.CenterCrop(config['image_shape'][:-1])(x)
                            # mask = transforms.Resize(config['image_shape'][:-1])(mask)
                            # mask = transforms.CenterCrop(config['image_shape'][:-1])(mask)
                            x = transforms.ToTensor()(x)
                            mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
                            x = normalize(x)
                            x = x * (1. - mask)
                            x = x.unsqueeze(dim=0)
                            # x_raw = x
                            mask = mask.unsqueeze(dim=0)
                        elif mask_file[j]:
                            raise TypeError("{} is not an image file.".format(mask_file[j]))
                        else:
                            # Test a single ground-truth image with a random mask
                            ground_truth = default_loader(file[i])
                            ground_truth = transforms.Resize(config['image_shape'][:-1])(ground_truth)
                            ground_truth = transforms.CenterCrop(config['image_shape'][:-1])(ground_truth)
                            ground_truth = transforms.ToTensor()(ground_truth)
                            ground_truth = normalize(ground_truth)
                            ground_truth = ground_truth.unsqueeze(dim=0)
                            bboxes = test_bbox(config, batch_size=ground_truth.size(0), t=50, l=50)
                            x, mask = mask_image(ground_truth, bboxes, config)

                        # Set checkpoint path
                        if not args.checkpoint_path:
                            checkpoint_path = os.path.join('checkpoints',
                                                           config['dataset_name'],
                                                           config['mask_type'] + '_' + config['expname'])
                        else:
                            checkpoint_path = args.checkpoint_path

                        # Define the trainer
                        netG = Generator(config['netG'], cuda, device_ids)
                        # Resume weight
                        g_checkpoint = torch.load(f'{checkpoint_path}/gen.pt')
                        netG.load_state_dict(g_checkpoint)
                        # model_iteration = int(last_model_name[-11:-3])
                        print("Model Resumed".format(checkpoint_path))

                        if cuda:
                            netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
                            x = x.cuda()
                            mask = mask.cuda()

                        # Inference
                        x1, x2 = netG(x, mask)
                        inpainted_result = x2 * mask + x * (1. - mask)
                        inpainted_result_cpu = torch.Tensor.cpu(inpainted_result).detach().permute(0, 2, 3, 1)
                        inpainted_result_cpu = np.asarray(inpainted_result_cpu[0])
                        inpainted_result_cpu = cv2.normalize(inpainted_result_cpu, inpainted_result_cpu, 0, 255,
                                                             cv2.NORM_MINMAX)

                        # cat_result = torch.cat([x, inpainted_result, ground_truth], dim=3).cuda()

                        vutils.save_image(inpainted_result, args.output_dir+ 'output_{}/'.format(j+1) + 'output_{}.png'.format(i), padding=0, normalize=True)
                        # cv2.imwrite(args.output_dir+ 'output_{}/'.format(j+1) + 'output_{}.png'.format(i), inpainted_result_cpu)
                        #             cv2.cvtColor(inpainted_result_cpu, cv2.COLOR_BGR2RGB))
                        print("{}th image saved".format(i))
                    else:
                        raise TypeError("{} is not an image file.".format)
            # exit no grad context
    except Exception as e:  # for unexpected error logging
        print("Error: {}".format(e))
        raise e


if __name__ == '__main__':
    main()
