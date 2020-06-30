import os
import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model.network import Generator
from utils.tools import get_config, test_bbox, mask_image, is_image_file, default_loader, normalize


parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/test.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, default = '2019', help='manual seed')
parser.add_argument('--image', type=str, default='example/image/image.jpg')
parser.add_argument('--mask', type=str, default='example/mask/mask.png')
parser.add_argument('--output', type=str, default='output/output.png')
parser.add_argument('--flow', type=str, default='')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--iter', type=int, default=0)

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
            if is_image_file(args.image):
                if args.mask and is_image_file(args.mask):
                    # Test a single masked image with a given mask
                    x = default_loader(args.image)
                    mask = default_loader(args.mask)
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
                elif args.mask:
                    raise TypeError("{} is not an image file.".format(args.mask))
                else:
                    # Test a single ground-truth image with a random mask
                    ground_truth = default_loader(args.image)
                    ground_truth = transforms.ToTensor()(ground_truth)
                    ground_truth = normalize(ground_truth)
                    ground_truth = ground_truth.unsqueeze(dim=0)
                    bboxes = test_bbox(config, batch_size=ground_truth.size(0), t = 50, l = 60)
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
                netG.load_state_dict(g_checkpoint, strict=False)
                # model_iteration = int(last_model_name[-11:-3])
                print("Model Resumed".format(checkpoint_path))

                if cuda:
                    netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
                    x = x.cuda()
                    mask = mask.cuda()

                # Inference
                x1, x2 = netG(x, mask)
                inpainted_result = x2 * mask + x * (1. - mask)

                vutils.save_image(inpainted_result, args.output, padding=0, normalize=True)
                print("Saved the inpainted result to {}".format(args.output))
            else:
                raise TypeError("{} is not an image file.".format)
        # exit no grad context
    except Exception as e:  # for unexpected error logging
        print("Error: {}".format(e))
        raise e


if __name__ == '__main__':
    main()
