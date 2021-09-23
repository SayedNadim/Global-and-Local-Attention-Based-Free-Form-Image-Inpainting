import os
import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model.networks_chord import Generator
from utils.tools import get_config, is_image_file, default_loader, normalize

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/test.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, default='2019', help='manual seed')
parser.add_argument('--mask_path', type=str, default='sample_mask/mask_236.png')
parser.add_argument('--image_path', type=str, default='sample_image/Places365_val_00000905.jpg')
parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--iter', type=int, default=0)


def main():
    args = parser.parse_args()
    config = get_config(args.config)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

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
        with torch.no_grad():  # enter no grad context
            if is_image_file(args.image_path):
                if args.mask_path and is_image_file(args.mask_path):
                    # Test a single masked image with a given mask
                    x = default_loader(args.image_path)
                    mask = default_loader(args.mask_path)
                    x = transforms.Resize(config['image_shape'][:-1])(x)
                    # x = transforms.CenterCrop(config['image_shape'][:-1])(x)
                    mask = transforms.Resize(config['image_shape'][:-1])(mask)
                    # mask = transforms.CenterCrop(config['image_shape'][:-1])(mask)
                    x = transforms.ToTensor()(x)
                    mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
                    x = normalize(x)
                    x = x * (1. - mask)
                    x = x.unsqueeze(dim=0)
                    x_raw = x
                    mask = mask.unsqueeze(dim=0)
                else:
                    raise TypeError("{} is not an image file.".format(args.mask_path))
                # Set checkpoint path
                checkpoint_path = config['checkpoint_dir']
                netG = Generator(config['netG'], cuda, device_ids)
                # Resume weight
                g_checkpoint = torch.load(f'{checkpoint_path}/gen.pt')
                netG.load_state_dict(g_checkpoint)
                print("Model Resumed".format(checkpoint_path))
                if cuda:
                    netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
                    x = x.cuda()
                    mask = mask.cuda()

                # Inference
                x1, x2 = netG(x, mask)
                inpainted_result = x2 * mask + x * (1. - mask)
                vutils.save_image(inpainted_result, args.output_dir+'/output_3.png', padding=0, normalize=True)
                print("Saved the inpainted result to {}".format(args.output_dir))
            else:
                raise TypeError("{} is not an image file.".format)
        # exit no grad context
    except Exception as e:  # for unexpected error logging
        print("Error: {}".format(e))
        raise e


if __name__ == '__main__':
    main()
