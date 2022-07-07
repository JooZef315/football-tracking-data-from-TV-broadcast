import numpy as np
import torch

# from .utils import utils


def normalize_single_image(image):
    assert len(image.shape) == 3
    assert isinstance(image, torch.Tensor)
    img_mean = torch.mean(image, dim=(1, 2)).view(-1, 1, 1)
    img_std = image.contiguous().view(image.size(0), -1).std(-1).view(-1, 1, 1)
    image = (image - img_mean) / img_std
    return image


def rgb_template_to_coord_conv_template(rgb_template):
    assert isinstance(rgb_template, np.ndarray)
    assert rgb_template.min() >= 0.0
    assert rgb_template.max() <= 1.0
    rgb_template = np.mean(rgb_template, 2)
    x_coord, y_coord = np.meshgrid(np.linspace(0, 1, num=rgb_template.shape[1]),
                                   np.linspace(0, 1, num=rgb_template.shape[0]))
    coord_conv_template = np.stack((rgb_template, x_coord, y_coord), axis=2)
    return coord_conv_template
