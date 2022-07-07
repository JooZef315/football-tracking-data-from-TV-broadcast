'''
the initial guesser + loss surface optimization
'''

import abc

import numpy as np
import torch
from tqdm import tqdm

from . import end_2_end_optimization_helper, loss_surface
from ..utils import utils, warp


class End2EndOptimFactory(object):
    @staticmethod
    def get_end_2_end_optimization_model(opt):
        if opt.optim_method == 'directh':
            model = End2EndOptimDirectH(opt)
        elif opt.optim_method == 'stn':
            model = End2EndOptimSTN(opt)
        else:
            raise ValueError('unknown optimization method:', opt.optim_method)
        return model


class End2EndOptim(abc.ABC):
    '''
    model for optimization
    '''

    def __init__(self, opt):
        self.opt = opt
        self.check_options()
        self.build_criterion()
        self.build_models()
        self.build_homography_inference()
        self.lambdas = None

    def check_options(self):
        valid_models = ['loss_surface']
        if self.opt.error_model not in valid_models:
            content_list = []
            content_list += [
                'End2EndOptim current only support {0} as optimization objective'.format(valid_models)]
            utils.print_notification(content_list, 'ERROR')
            exit(1)
        assert self.opt.optim_iters > 0, 'optimization iterations should be larger than 0'

    def build_criterion(self):
        if self.opt.optim_criterion == 'l1loss':
            self.criterion = torch.nn.L1Loss(reduction='sum')
        elif self.opt.optim_criterion == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='sum')
        else:
            raise ValueError('unknown optimization criterion: {0}'.format(
                self.opt.optim_criterion))

    def build_models(self):
        self.optim_net = loss_surface.ErrorModelFactory.get_error_model(
            self.opt)
        self.optim_net = utils.set_model_device(self.optim_net)
        self.optim_net.eval()
        if self.opt.error_target == 'iou_whole':
            self.target_dist = torch.ones((1, 1), requires_grad=False)
            self.target_dist = utils.set_tensor_device(self.target_dist)
        else:
            raise ValueError(
                'unknown error target: {0}'.format(self.opt.error_target))

    def build_homography_inference(self):
        self.homography_inference = end_2_end_optimization_helper.HomographyInferenceFactory.get_homography_inference(
            self.opt)

    def create_gd_optimizer(self, params):
        optim_list = [{"params": params, "lr": self.opt.lr_optim}]
        if self.opt.optim_type == 'adam':
            optim = torch.optim.Adam(optim_list)
        elif self.opt.optim_type == 'sgd':
            optim = torch.optim.SGD(optim_list)
        else:
            raise ValueError(
                'unknown optimization type: {0}'.format(self.opt.optim_type))
        return optim

    def first_order_main_optimization_loop(self, frame, template, optim_tools, get_corners_fun, corner_to_mat_fun):
        loss_hist = []
        corners_optim_list = []
        optimizer = optim_tools['optimizer']
        B = frame.shape[0]
        for i in tqdm(range(0, self.opt.optim_iters)):
            corners_optim = get_corners_fun()
            corners_optim_list.append(corners_optim)
            inferred_transformation_mat = corner_to_mat_fun(corners_optim)
            warped_tmp = warp.warp_image(
                template, inferred_transformation_mat, out_shape=frame.shape[-2:])
            inferred_dist = self.optim_net((frame, warped_tmp))
            optim_loss = self.get_loss(inferred_dist, self.target_dist.repeat(B, 1))
            loss_hist.append(optim_loss.clone().detach().cpu().numpy())
            if torch.isnan(optim_loss.data):
                assert 0, 'loss is nan during optimization'
            else:
                optimizer.zero_grad()
                optim_loss.backward()
                optimizer.step()
            if optim_loss.data < 0.000000:
                break
        loss_hist = np.array(loss_hist)
        return loss_hist, corners_optim_list

    def get_loss(self, output, target):
        optim_loss = self.criterion(output, target)
        return optim_loss

    def main_optimization_loop(self, frame, template, optim_tools, get_corners_fun, corner_to_mat_fun):
        if self.opt.optim_type == 'adam' or 'sgd':
            loss_hist, corners_optim_list = self.first_order_main_optimization_loop(
                frame, template, optim_tools, get_corners_fun, corner_to_mat_fun)
        else:
            raise ValueError(
                'unknown optimization type: {0}'.format(self.opt.optim_type))
        return loss_hist, corners_optim_list

    @abc.abstractmethod
    def optim(self, frame, template, refresh=True):
        pass


class End2EndOptimDirectH(End2EndOptim):
    def optim(self, frame, template, refresh=True):
        def get_corners_directh():
            return corners_optim

        def corner_to_mat_directh(corners):
            return end_2_end_optimization_helper.get_homography_between_corners_and_default_canon4pts(corners, self.opt.directh_part)

        self.homography_inference.refresh()
        assert self.homography_inference.get_training_status(
        ) is False, 'set model to eval mode at optimization stage'
        assert self.optim_net.training is False, 'set model to eval mode at optimization stage'
        B = frame.shape[0]

        template = template.repeat(B, 1, 1, 1)
        upstream_homography = self.homography_inference.infer_upstream_homography(frame)
        # canon4pts would be full or lower based on the options
        canon4pts = end_2_end_optimization_helper.get_default_canon4pts(B, canon4pts_type=self.opt.directh_part)

        corners_optim = warp.get_four_corners(upstream_homography, canon4pts=canon4pts[0])
        corners_optim = corners_optim.permute(0, 2, 1)
        corners_optim = corners_optim.clone().detach().requires_grad_(True)
        optim = self.create_gd_optimizer(params=corners_optim)
        optim_tools = {'optimizer': optim}
        loss_hist, corners_optim_list = self.main_optimization_loop(frame,
                                                                    template,
                                                                    optim_tools,
                                                                    get_corners_directh,
                                                                    corner_to_mat_directh)

        orig_homography = upstream_homography
        optim_homography = corner_to_mat_directh(corners_optim_list[loss_hist.argmin()])
        return orig_homography, optim_homography


class End2EndOptimSTN(End2EndOptim):
    def optim(self, frame, template, refresh=True):
        def get_corners_stn():
            return self.homography_inference.infer_upstream_corners(frame)

        def corner_to_mat_stn(corners):
            return end_2_end_optimization_helper.get_homography_between_corners_and_default_canon4pts(corners, 'lower')

        if refresh:
            self.homography_inference.refresh()
        assert self.homography_inference.get_training_status() is False, 'set model to eval mode at optimization stage'
        assert self.optim_net.training is False, 'set model to eval mode at optimization stage'
        B = frame.shape[0]
        assert B == 1, 'STN optimization only support one image at a time'

        upstream_homography = self.homography_inference.infer_upstream_homography(frame)
        optim = self.create_gd_optimizer(params=self.homography_inference.get_upstream_params())
        optim_tools = {'optimizer': optim}
        loss_hist, corners_optim_list = self.main_optimization_loop(frame,
                                                                    template,
                                                                    optim_tools,
                                                                    get_corners_stn,
                                                                    corner_to_mat_stn)
        orig_homography = upstream_homography
        optim_homography = end_2_end_optimization_helper.get_homography_between_corners_and_default_canon4pts(corners_optim_list[loss_hist.argmin()], 'lower')
        return orig_homography, optim_homography
