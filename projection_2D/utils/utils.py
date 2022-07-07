'''utils functions, variables
'''
import random

import numpy as np
import torch
try:
    import kornia
except ModuleNotFoundError:
    pass

from . import constant_var
# from utils import constant_var


def confirm(question='OK to continue?'):
    """
    Ask user to enter Y or N (case-insensitive).
    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(question + ' [y/n] ').lower()
    return answer == "y"


def print_notification(content_list, notifi_type='NOTIFICATION'):
    print(
        '---------------------- {0} ----------------------'.format(notifi_type))
    print()
    for content in content_list:
        print(content)
    print()
    print('-------------------------- END --------------------------')


def to_torch(np_array):
    if constant_var.USE_CUDA:
        tensor = torch.from_numpy(np_array).float().cuda()
    else:
        tensor = torch.from_numpy(np_array).float()
    return torch.autograd.Variable(tensor, requires_grad=False)


def set_tensor_device(torch_var):
    if constant_var.USE_CUDA:
        return torch_var.cuda()
    else:
        return torch_var


def set_model_device(model):
    if constant_var.USE_CUDA:
        return model.cuda()
    else:
        return model


def to_numpy(cuda_var):
    return cuda_var.data.cpu().numpy()


def isnan(x):
    return x != x


def hasnan(x):
    return isnan(x).any()


def torch_img_to_np_img(torch_img):
    '''convert a torch image to matplotlib-able numpy image
    torch use Channels x Height x Width
    numpy use Height x Width x Channels
    Arguments:
        torch_img {[type]} -- [description]
    '''
    assert isinstance(torch_img, torch.Tensor), 'cannot process data type: {0}'.format(type(torch_img))
    if len(torch_img.shape) == 4 and (torch_img.shape[1] == 3 or torch_img.shape[1] == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (0, 2, 3, 1))
    if len(torch_img.shape) == 3 and (torch_img.shape[0] == 3 or torch_img.shape[0] == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (1, 2, 0))
    elif len(torch_img.shape) == 2:
        return torch_img.detach().cpu().numpy()
    else:
        raise ValueError('cannot process this image')


def np_img_to_torch_img(np_img):
    '''convert a numpy image to torch image
    numpy use Height x Width x Channels
    torch use Channels x Height x Width

    Arguments:
        np_img {[type]} -- [description]
    '''
    assert isinstance(np_img, np.ndarray), 'cannot process data type: {0}'.format(type(np_img))
    if len(np_img.shape) == 4 and (np_img.shape[3] == 3 or np_img.shape[3] == 1):
        return to_torch(np.transpose(np_img, (0, 3, 1, 2)))
    if len(np_img.shape) == 3 and (np_img.shape[2] == 3 or np_img.shape[2] == 1):
        return to_torch(np.transpose(np_img, (2, 0, 1)))
    elif len(np_img.shape) == 2:
        return to_torch(np_img)
    else:
        raise ValueError('cannot process this image')


def fix_randomness():
    random.seed(542)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(719)
    np.random.seed(121)


# the homographies map to coordinates in the range [-0.5, 0.5] (the ones in GT datasets)
BASE_RANGE = 0.5


def FULL_CANON4PTS_NP():
    return np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=np.float32)


def LOWER_CANON4PTS_NP():
    return np.array([[-0.5, 0.1], [-0.5, 0.5], [0.5, 0.5], [0.5, 0.1]], dtype=np.float32)


def get_perspective_transform(src, dst):
    '''
    kornia: https://github.com/arraiyopensource/kornia
    license: https://github.com/arraiyopensource/kornia/blob/master/LICENSE
    '''
    try:
        return kornia.get_perspective_transform(src, dst)
    except:
        r"""Calculates a perspective transform from four pairs of the corresponding
        points.
        The function calculates the matrix of a perspective transform so that:
        .. math ::
            \begin{bmatrix}
            t_{i}x_{i}^{'} \\
            t_{i}y_{i}^{'} \\
            t_{i} \\
            \end{bmatrix}
            =
            \textbf{map_matrix} \cdot
            \begin{bmatrix}
            x_{i} \\
            y_{i} \\
            1 \\
            \end{bmatrix}
        where
        .. math ::
            dst(i) = (x_{i}^{'},y_{i}^{'}), src(i) = (x_{i}, y_{i}), i = 0,1,2,3
        Args:
            src (Tensor): coordinates of quadrangle vertices in the source image.
            dst (Tensor): coordinates of the corresponding quadrangle vertices in
                the destination image.
        Returns:
            Tensor: the perspective transformation.
        Shape:
            - Input: :math:`(B, 4, 2)` and :math:`(B, 4, 2)`
            - Output: :math:`(B, 3, 3)`
        """
        if not torch.is_tensor(src):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(src)))
        if not torch.is_tensor(dst):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(dst)))
        if not src.shape[-2:] == (4, 2):
            raise ValueError("Inputs must be a Bx4x2 tensor. Got {}"
                            .format(src.shape))
        if not src.shape == dst.shape:
            raise ValueError("Inputs must have the same shape. Got {}"
                            .format(dst.shape))
        if not (src.shape[0] == dst.shape[0]):
            raise ValueError("Inputs must have same batch size dimension. Expect {} but got {}"
                            .format(src.shape, dst.shape))

        def ax(p, q):
            ones = torch.ones_like(p)[..., 0:1]
            zeros = torch.zeros_like(p)[..., 0:1]
            return torch.cat(
                [p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros,
                -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1]
                ], dim=1)

        def ay(p, q):
            ones = torch.ones_like(p)[..., 0:1]
            zeros = torch.zeros_like(p)[..., 0:1]
            return torch.cat(
                [zeros, zeros, zeros, p[:, 0:1], p[:, 1:2], ones,
                -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2]], dim=1)
        # we build matrix A by using only 4 point correspondence. The linear
        # system is solved with the least square method, so here
        # we could even pass more correspondence
        p = []
        p.append(ax(src[:, 0], dst[:, 0]))
        p.append(ay(src[:, 0], dst[:, 0]))

        p.append(ax(src[:, 1], dst[:, 1]))
        p.append(ay(src[:, 1], dst[:, 1]))

        p.append(ax(src[:, 2], dst[:, 2]))
        p.append(ay(src[:, 2], dst[:, 2]))

        p.append(ax(src[:, 3], dst[:, 3]))
        p.append(ay(src[:, 3], dst[:, 3]))

        # A is Bx8x8
        A = torch.stack(p, dim=1)

        # b is a Bx8x1
        b = torch.stack([
            dst[:, 0:1, 0], dst[:, 0:1, 1],
            dst[:, 1:2, 0], dst[:, 1:2, 1],
            dst[:, 2:3, 0], dst[:, 2:3, 1],
            dst[:, 3:4, 0], dst[:, 3:4, 1],
        ], dim=1)

        # solve the system Ax = b
        X, LU = torch.solve(b, A)

        # create variable to return
        batch_size = src.shape[0]
        M = torch.ones(batch_size, 9, device=src.device, dtype=src.dtype)
        M[..., :8] = torch.squeeze(X, dim=-1)
        return M.view(-1, 3, 3)  # Bx3x3
