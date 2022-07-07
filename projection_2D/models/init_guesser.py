'''
the model for learning the initial guess
'''

import os
from argparse import Namespace

import torch

from . import base_model, resnet
from ..utils import utils


class InitialGuesserFactory(object):
    @staticmethod
    def get_initial_guesser(opt):
        if opt.guess_model == 'init_guess':
            model = InitialGuesser(opt)
            model = utils.set_model_device(model)
        else:
            raise ValueError('unknown initial guess model:',
                             opt.loss_surface_name)
        return model


class InitialGuesser(base_model.BaseModel, torch.nn.Module):
    '''
    Model for learning the initial guess
    '''

    def __init__(self, opt):
        self.opt = opt
        self.name = 'init_guess'
        self.check_options()
        super(InitialGuesser, self).__init__()
        self.create_model()

    def check_options(self):
        if self.opt.guess_model != self.name:
            content_list = []
            content_list += ['You are not using the correct class for training or eval']
            utils.print_notification(content_list, 'ERROR')
            exit(1)

    def create_model(self):
        self.out_dim = 8
        self.input_features = 3
        resnet_config = self.create_resnet_config()
        self.feature_extractor = resnet.resnet18(resnet_config, pretrained=resnet_config.pretrained,
                                                 num_classes=self.out_dim, input_features=self.input_features)
        if (hasattr(self.opt, 'load_weights_upstream') and self.opt.load_weights_upstream):
            assert resnet_config.pretrained is False, 'pretrained weights or imagenet weights'
            self.load_pretrained_weights()

    def create_resnet_config(self):
        need_spectral_norm = False
        pretrained = False
        group_norm = 0
        if hasattr(self.opt, 'need_spectral_norm') and self.opt.need_spectral_norm:
            need_spectral_norm = self.opt.need_spectral_norm
        elif hasattr(self.opt, 'need_spectral_norm_upstream') and self.opt.need_spectral_norm_upstream:
            need_spectral_norm = self.opt.need_spectral_norm_error_model
        if hasattr(self.opt, 'group_norm'):
            group_norm = self.opt.group_norm
        elif hasattr(self.opt, 'group_norm_upstream'):
            group_norm = self.opt.group_norm_upstream
        if hasattr(self.opt, 'imagenet_pretrain') and self.opt.imagenet_pretrain:
            pretrained = True
        resnet_config = Namespace(need_spectral_norm=need_spectral_norm,
                                  pretrained=pretrained,
                                  group_norm=group_norm,
                                  )
        self.print_resnet_config(resnet_config)
        return resnet_config

    def forward(self, x):
        video = x
        y = self.feature_extractor(video)
        return y

    def load_pretrained_weights(self):
        '''load pretrained weights
        this function can load weights from another model.
        '''
        super().load_pretrained_weights()

    def _verify_checkpoint(self, check_options):
        pass

    def _get_checkpoint_path(self):
        checkpoint_path = os.path.join(self.opt.out_dir, self.opt.load_weights_upstream, 'checkpoint.pth.tar')
        return checkpoint_path
