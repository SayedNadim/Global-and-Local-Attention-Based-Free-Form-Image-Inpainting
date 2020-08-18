import numpy as np
import cv2
import random
import scipy.ndimage as ndimage
import scipy.misc as misc
import torch
import torch.nn.functional as F
import math
from PIL import Image, ImageDraw


def random_mask(height=256,
                width=256,
                max_vertex=10,
                max_stroke=3,
                max_brush_width=40,
                max_length=100):
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
    return (mask > 0).astype(np.float32)

## From DEEPFILL2
class brush_stroke_mask(torch.nn.Module):
    def __init__(self):
        super(brush_stroke_mask, self).__init__()
        self.min_num_vertex = 4
        self.max_num_vertex = 12
        self.mean_angle = 2 * math.pi / 5
        self.angle_range = 2 * math.pi / 15
        self.min_width = 12
        self.max_width = 40

    def generate_mask(self,H, W):
        average_radius = math.sqrt(H * H + W * W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(self.min_num_vertex, self.max_num_vertex)
            angle_min = self.mean_angle - np.random.uniform(0, self.angle_range)
            angle_max = self.mean_angle + np.random.uniform(0, self.angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius // 2),
                    0, 2 * average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(self.min_width, self.max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        # mask = np.reshape(mask, (1, 1, H, W))
        return mask


def mask_image(x, config):
    height, width, _ = config['image_shape']
    max_mask = x.shape[0]
    result = torch.ones_like(x)
    mask = torch.ones(size=[x.shape[0], 1, x.shape[2], x.shape[3]])
    for i in range(max_mask):
        # mask_temp = random_mask(height=height, width=width)
        mask_temp = brush_stroke_mask().generate_mask(height, width)
        mask_temp_tensor = torch.tensor(mask_temp, dtype=torch.float32)
        if x.is_cuda:
            mask_temp_tensor.cuda()
        result[i, :, :, :] = x[i, :, :, :] * (1. - mask_temp_tensor)
        mask[i, :, :, :] = mask[i, :, :, :] * mask_temp_tensor
    return result, mask

