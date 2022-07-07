'''evaluation metrics
'''


import torch
import numpy as np

from .utils import utils, warp


class IOU(object):
    '''IOU metrics
    '''

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        assert hasattr(opt, 'iou_space'), 'missing iou_space in options'
        assert hasattr(opt, 'dataset_name'), 'missing dataset_name in options'
        self.epsilon = 1e-6
        if self.opt.dataset_name == 'world_cup':
            self.template_width = 115
            self.template_height = 74
            self.frame_width = 256
            self.frame_height = 256
        else:
            raise ValueError(
                'unknown dataset name: {0}'.format(self.opt.dataset_name))

    def __call__(self, output, target):
        return self.forward(output, target)

    def forward(self, output, target):
        '''calculate the IOU between two homography

        Arguments:
            output {[type]} -- a batch of inferred homography, torch tensor, shape: (B, 3, 3)
            target {[type]} -- corresponding GT homography, torch tensor, shape: (B, 3, 3)
        '''

        if self.opt.iou_space == 'model_part':
            model_part_iou_val = self.get_model_part_iou_rasterization(
                output, target)
            return model_part_iou_val
        elif self.opt.iou_space == 'model_whole':
            model_whole_iou_val = self.get_model_whole_iou_rasterization(
                output, target)
            return model_whole_iou_val
        elif self.opt.iou_space == 'part_and_whole':
            model_part_iou_val = self.get_model_part_iou_rasterization(
                output, target)
            model_whole_iou_val = self.get_model_whole_iou_rasterization(
                output, target)
            return model_part_iou_val, model_whole_iou_val
        else:
            raise ValueError('unknown space for iou calculation: {0}'.format(self.opt.iou_space))

    def get_model_part_iou_rasterization(self, output, target):
        if output.shape == (3, 3):
            output = output[None]
        if target.shape == (3, 3):
            target = target[None]
        assert output.shape == target.shape, 'output shape does not match target shape'
        batch_size = output.shape[0]
        fake_frame = torch.ones(
            [batch_size, 1, self.frame_height, self.frame_width], device=output.device)
        fake_template = torch.ones(
            [batch_size, 1, self.template_height, self.template_width], device=output.device)
        output_mask = warp.warp_image(
            fake_frame, output.inverse(), out_shape=fake_template.shape[-2:])
        target_mask = warp.warp_image(
            fake_frame, target.inverse(), out_shape=fake_template.shape[-2:])
        output_mask[output_mask > 0] = 1
        target_mask[target_mask > 0] = 1

        intersection_mask = output_mask * target_mask
        output = output_mask.sum(dim=[1, 2, 3])
        target = target_mask.sum(dim=[1, 2, 3])
        intersection = intersection_mask.sum(dim=[1, 2, 3])
        union = output + target - intersection
        iou = intersection / (union + self.epsilon)
        return utils.to_numpy(iou)

    def get_model_whole_iou_rasterization(self, output, target):
        '''calculate the model whole iou
        '''
        if output.shape == (3, 3):
            output = output[None]
        if target.shape == (3, 3):
            target = target[None]
        assert output.shape == target.shape, 'output shape does not match target shape'
        ZOOM_OUT_SCALE = 4
        batch_size = output.shape[0]
        fake_template = torch.ones([batch_size, 1, self.template_height * ZOOM_OUT_SCALE, self.template_width * ZOOM_OUT_SCALE], device=output.device)
        scaling_mat = torch.eye(3, device=output.device).repeat(batch_size, 1, 1)
        scaling_mat[:, 0, 0] = scaling_mat[:, 1, 1] = ZOOM_OUT_SCALE
        target_mask = warp.warp_image(fake_template, scaling_mat, out_shape=fake_template.shape[-2:])
        mapping_mat = torch.matmul(output, scaling_mat)
        mapping_mat = torch.matmul(target.inverse(), mapping_mat)
        output_mask = warp.warp_image(fake_template, mapping_mat, out_shape=fake_template.shape[-2:])

        output_mask = (output_mask >= 0.5).float()
        target_mask = (target_mask >= 0.5).float()
        intersection_mask = output_mask * target_mask
        output = output_mask.sum(dim=[1, 2, 3])
        target = target_mask.sum(dim=[1, 2, 3])
        intersection = intersection_mask.sum(dim=[1, 2, 3])
        union = output + target - intersection
        iou = intersection / (union + self.epsilon)
        return utils.to_numpy(iou)
