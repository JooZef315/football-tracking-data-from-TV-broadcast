'''
the model for learning the loss surface.
'''

import os
import abc
from argparse import Namespace

import torch
import torch.nn.functional as F

from . import base_model, resnet
from ..utils import utils


class ErrorModelFactory():
    '''this is the factory for the error model
    '''

    @staticmethod
    def get_error_model(opt):
        if opt.error_model == 'loss_surface':
            model = LossSurfaceRegressor(opt)
            model = utils.set_model_device(model)
        else:
            raise ValueError(
                'unknown loss surface model: {0}'.format(opt.loss_surface_name))
        return model


class BaseErrorModel(base_model.BaseModel, torch.nn.Module, abc.ABC):
    '''base model for all kinds of error models
    '''

    def __init__(self):
        super(BaseErrorModel, self).__init__()

    def check_options(self):
        if self.opt.error_model != self.name:
            content_list = []
            content_list += ['You are not using the correct class for training or eval']
            content_list += [
                'error_model in options: {0}, current error_model class: {1}'.format(self.opt.error_model, self.name)]
            utils.print_notification(content_list, 'ERROR')
            exit(1)

    def make_value_positive(self, x):
        if self.prevent_neg == 'sigmoid':
            x = torch.sigmoid(x)
        else:
            content_list = []
            content_list += [
                'Unknown prevent_neg method: {0}'.format(self.prevent_neg)]
            utils.print_notification(content_list, 'ERROR')
            exit(1)
        return x

    def load_pretrained_weights(self):
        '''load pretrained weights
        this function can load weights from another model.
        '''
        super().load_pretrained_weights()

    def _verify_checkpoint(self, checkpoint):
        if checkpoint['prevent_neg'] != self.opt.prevent_neg:
            content_list = []
            content_list += [
                'Prevent negative method are different between the checkpoint and user options']
            utils.print_notification(content_list, 'ERROR')
            exit(1)

    def _get_checkpoint_path(self):
        checkpoint_path = os.path.join(self.opt.out_dir, self.opt.load_weights_error_model, 'checkpoint.pth.tar')
        return checkpoint_path

    def create_resnet_config(self):
        need_spectral_norm = False
        pretrained = False
        group_norm = 0
        if hasattr(self.opt, 'need_spectral_norm') and self.opt.need_spectral_norm:
            need_spectral_norm = True
        elif hasattr(self.opt, 'need_spectral_norm_error_model') and self.opt.need_spectral_norm_error_model:
            need_spectral_norm = True
        if hasattr(self.opt, 'group_norm'):
            group_norm = self.opt.group_norm
        elif hasattr(self.opt, 'group_norm_error_model'):
            group_norm = self.opt.group_norm_error_model

        if hasattr(self.opt, 'imagenet_pretrain') and self.opt.imagenet_pretrain:
            pretrained = True
        resnet_config = Namespace(need_spectral_norm=need_spectral_norm,
                                  pretrained=pretrained,
                                  group_norm=group_norm,
                                  )
        self.print_resnet_config(resnet_config)
        return resnet_config


class LossSurfaceRegressor(BaseErrorModel):
    '''
    Model for learning the loss surface
    '''

    def __init__(self, opt):
        self.opt = opt
        self.name = 'loss_surface'
        self.check_options()
        super(LossSurfaceRegressor, self).__init__()
        self.create_model()

    def create_model(self):
        self.prevent_neg = self.opt.prevent_neg
        if self.opt.error_target in ['iou_whole']:
            self.out_dim = 1
        else:
            raise ValueError(
                'unknown error target: {0}'.format(self.opt.error_target))
        self.input_features = 3 * 2
        resnet_config = self.create_resnet_config()
        self.feature_extractor = resnet.resnet18(resnet_config, pretrained=False, num_classes=self.out_dim,
                                                 input_features=self.input_features)
        if (hasattr(self.opt, 'load_weights_error_model') and self.opt.load_weights_error_model):
            self.load_pretrained_weights()

    def create_resnet_config(self):
        if hasattr(self.opt, 'imagenet_pretrain') and self.opt.imagenet_pretrain:
            content_list = []
            content_list += [
                'LossSurfaceRegressor do not support imagenet pretrained weights loading']
            utils.print_notification(content_list, 'ERROR')
            exit(1)

        resnet_config = super().create_resnet_config()
        return resnet_config

    def forward(self, x):
        video, template = x
        image_stack = torch.cat((video, template), 1)  # stack along channel
        y = self.feature_extractor(image_stack)
        y = self.make_value_positive(y)
        return y
