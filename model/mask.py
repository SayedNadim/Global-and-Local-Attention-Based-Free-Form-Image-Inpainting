import numpy as np
import cv2
import random
import scipy.ndimage as ndimage
import scipy.misc as misc
import torch
import torch.nn.functional as F


class Masks():

    @staticmethod
    def get_ff_mask(h, w, num_v=None):
        # Source: Generative Inpainting https://github.com/JiahuiYu/generative_inpainting

        mask = np.zeros((h, w))
        if num_v is None:
            num_v = 15 + np.random.randint(9)  # 5

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(4.0)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(60)  # 40
                brush_w = 10 + np.random.randint(15)  # 10
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y

        return (mask > 0).astype(np.float32)


def mask_image(x, config):
    height, width, _ = config['image_shape']
    mask_all = []
    for i in range(x.size(0)):
        mask = Masks.get_ff_mask(height, width)
        mask_all.append(mask)
    mask = torch.from_numpy(np.asarray(mask_all)).unsqueeze(1).float()
    ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
    mask = ones * mask
    if x.is_cuda:
        mask = mask.cuda()
    result = x * (1. - mask)
    return result, mask
