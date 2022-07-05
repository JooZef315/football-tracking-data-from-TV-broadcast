'''option checking methods
after get all the options, we check whether this
combination is valid or not
'''

import os
import json

import torch

from utils import utils, constant_var


def check_pretrained_weights(opt):
    pretrained_weights_option_list = ['load_weights_upstream', 'load_weights_error_model']
    for pretrained_weights_option in pretrained_weights_option_list:
        if hasattr(opt, pretrained_weights_option) and getattr(opt, pretrained_weights_option):
            weights_path = os.path.join(opt.out_dir,
                                        getattr(
                                            opt, pretrained_weights_option),
                                        'checkpoint.pth.tar')
            if not os.path.exists(weights_path):
                content_list = []
                content_list += ['Cannot find pretrained weights for {0}, at {1}'.format(
                    pretrained_weights_option, weights_path)]
                utils.print_notification(content_list, 'ERROR')
                exit(1)
            if hasattr(opt, 'error_model'):
                check_prevent_neg(opt)


def check_warp_params(opt):
    opt.warp_dim = 8 if opt.warp_type == 'homography' else None


def check_prevent_neg(opt):
    if hasattr(opt, 'prevent_neg') and hasattr(opt, 'load_weights_error_model'):
        json_path = os.path.join(
            opt.out_dir, opt.load_weights_error_model, 'params.json')
        with open(json_path, 'r') as f:
            model_config = json.load(f)
        weights_prevent_neg = model_config['prevent_neg']
        if weights_prevent_neg != opt.prevent_neg:
            content_list = []
            content_list += [
                'Prevent negative method are different between the checkpoint and user options']
            utils.print_notification(content_list, 'ERROR')
            exit(1)


def check_cuda(opt):
    if opt.use_cuda:
        assert torch.cuda.is_available()
        constant_var.USE_CUDA = opt.use_cuda
