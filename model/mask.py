import numpy as np
import cv2
import random
import scipy.ndimage as ndimage
import scipy.misc as misc
import torch
import torch.nn.functional as F
import math


def random_mask(height=256,
                width=256,
                max_vertex=15,
                max_stroke=5,
                max_brush_width=40,
                max_length=150):
    mask = np.zeros((height, width))
    max_angle = np.pi
    for mn in range(max_stroke):
        num_vertex = int(np.random.uniform(max_vertex // 3, max_vertex))
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
        # mask = mask // 255
    return mask


def mask_image(x, config):
    height, width, _ = config['image_shape']
    max_mask = x.shape[0]
    result = torch.ones_like(x)
    mask = torch.ones(size=[x.shape[0], 1, x.shape[2], x.shape[3]])
    for i in range(max_mask):
        mask_temp = random_mask(height=height, width=width)
        mask_temp = torch.tensor(mask_temp, dtype=torch.float32)
        if x.is_cuda:
            mask_temp.cuda()
        result[i, :, :, :] = x[i, :, :, :] * (1. - mask_temp)
        mask[i, :, :, :] = mask[i, :, :, :] * mask_temp
    return result, mask
