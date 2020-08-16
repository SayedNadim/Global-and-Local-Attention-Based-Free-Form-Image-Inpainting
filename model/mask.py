import numpy as np
import cv2
import random
import scipy.ndimage as ndimage
import scipy.misc as misc
import torch
import torch.nn.functional as F
import math

def random_mask(max_mask,
                height=256,
                width=256,
                max_vertex=15,
                max_brush_width=40,
                max_length=200):
    mask = np.zeros((height, width))
    mask_all = []
    max_angle = np.pi
    for mn in range(max_mask):
        num_vertex = int(np.random.uniform(max_vertex//3, max_vertex))
        start_x = int(np.random.uniform(0, height))
        start_y = int(np.random.uniform(0, width))

        for step in range(0, num_vertex):
            angle = np.random.uniform(0, max_angle)

            if step % 2 == 0:
                # Reverse mode
                angle = 2 * math.pi - angle

            length = np.random.uniform(10, max_length)
            brush_width = int(np.random.uniform(max_brush_width / 4, max_brush_width))

            end_x = int(start_x + length * math.sin(angle))
            end_y = int(start_y + length * math.cos(angle))

            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1, brush_width)

            start_x = end_x
            start_y = end_y

            # cv2.circle(mask, (447, 63), brush_width // 4, 255, -1)

        if np.random.uniform(0.0, 1.0) > 0.5:
            mask = np.fliplr(mask)

        if np.random.uniform(0.0, 1.0) > 0.5:
            mask = np.flipud(mask)
        # mask = mask // 255
        mask_all.append(mask)
    batched_mask = torch.tensor(mask_all, dtype=torch.float32).unsqueeze(1)
    return batched_mask

def mask_image(x, config):
    height, width, _ = config['image_shape']
    max_mask = x.shape[0]
    mask = random_mask(max_mask)
    if x.is_cuda:
        mask = mask.cuda()

    if config['mask_type'] == 'hole':
        result = x * (1. - mask)
    elif config['mask_type'] == 'mosaic':
        # TODO: Matching the mosaic patch size and the mask size
        mosaic_unit_size = config['mosaic_unit_size']
        downsampled_image = F.interpolate(x, scale_factor=1. / mosaic_unit_size, mode='nearest')
        upsampled_image = F.interpolate(downsampled_image, size=(height, width), mode='nearest')
        result = upsampled_image * mask + x * (1. - mask)
    else:
        raise NotImplementedError('Not implemented mask type.')

    return result, mask