'''
testing options

jiang wei
uvic
start: 2018.11.28
'''

import sys
import argparse
import json
import os


from options.options_check import check_cuda, check_pretrained_weights, check_warp_params
from options.options_utils import str2bool, print_opt, confirm_opt


def read_global_config():
    try:
        __location__ = os.path.realpath(os.path.join(
            os.getcwd(), os.path.dirname(__file__)))
        with open(os.path.join(__location__, 'global_config.json'), 'r') as f:
            global_config = json.load(f)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        exit(1)
    return global_config


def set_general_arguments(parser):
    general_arg = parser.add_argument_group('General')
    general_arg.add_argument('--confirm', type=str2bool,
                             default=True, help='promote confirmation for user')
    general_arg.add_argument('--use_cuda', type=str2bool,
                             default=True, help='use cuda')


def set_data_arguments(parser):
    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--dataset_name', choices=['world_cup'],
                          type=str, default='world_cup', help='dataset name')
    data_arg.add_argument('--coord_conv_template', type=str2bool,
                          default=True, help='replace redundant channels tp XY grids')
    data_arg.add_argument('--template_path', type=str,
                          default='./projection_2D/data/pitch_template.png', help='playfield template image')
    data_arg.add_argument('--need_single_image_normalization', type=str2bool,
                          default=True, help='normalize a single image')


def set_warp_arguments(parser):
    warp_arg = parser.add_argument_group('Warp')
    warp_arg.add_argument('--warp_type', choices=['homography'], type=str, default='homography', help='how to warp images')
    warp_arg.add_argument('--homo_param_method', choices=['deep_homography'], type=str, default='deep_homography', help='how to parameterize homography')


def set_dataset_paths(opt, global_config):
    if opt.dataset_name == 'world_cup':
        opt.test_dataset_path = global_config[opt.dataset_name]['test_dataset']
    else:
        raise ValueError('unknown dataset_name: {0}'.format(opt.dataset_name))
    return opt


def set_end2end_optim_options():

    global_config = read_global_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str2bool,
                        default=True, help='test mode')
    args, _ = parser.parse_known_args()
    training = not args.test_mode
    set_general_arguments(parser)
    set_data_arguments(parser)
    set_warp_arguments(parser)

    parser.add_argument('error_model', choices=['loss_surface'], help='type of network')
    parser.add_argument('guess_model', choices=['init_guess'], help='type of initial guess network')
    parser.add_argument('--prevent_neg', choices=['sigmoid'],
                        type=str, default='sigmoid', help='how to prevent negative error estimation')
    parser.add_argument('--load_weights_upstream', type=str, default=None,
                        help='load a pretrained set of weights for upstream initial guesser, you need to provide the model name')
    parser.add_argument('--load_weights_error_model', type=str, default=None,
                        help='load a pretrained set of weights for error model, you need to provide the model name')
    parser.add_argument('--out_dir', type=str,
                        default=global_config['out'], help='out directory')
    parser.add_argument('--need_spectral_norm_upstream', type=str2bool, default=False,
                        help='apply spectral norm to conv layers, for upstream initial guesser')
    parser.add_argument('--need_spectral_norm_error_model', type=str2bool,
                        default=True, help='apply spectral norm to conv layers, for error model')
    parser.add_argument('--group_norm_upstream', type=int,
                        default=0, help='0 is batch norm, otherwise means number of groups')
    parser.add_argument('--group_norm_error_model', type=int,
                        default=0, help='0 is batch norm, otherwise means number of groups')
    parser.add_argument('--optim_method', default='directh', choices=[
                        'stn', 'directh'], help='optimization method')
    parser.add_argument('--optim_type', default='adam',
                        choices=['adam', 'sgd'], help='gradient descent optimizer type')
    parser.add_argument('--directh_part', default='lower', choices=[
                        'full', 'lower'], help='optimize the lower 4 corners, or full 4 corners')
    parser.add_argument('--optim_criterion', default='l1loss',
                        choices=['l1loss', 'mse'], help='criterion for optimization')
    parser.add_argument('--lr_optim', type=float,
                        default=1e-3, help='optimization learning rate')
    parser.add_argument('--optim_iters', type=int, default=400,
                        help='iterations for optimization')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for optimization')
    parser.add_argument('--iou_space', default='part_and_whole', choices=[
                        'model_whole', 'model_part', 'part_and_whole'], help='space for iou calculation')
    parser.add_argument('--error_target', choices=['iou_whole'], type=str, default='iou_whole', help='error target for loss surface')

    opt = parser.parse_args()
    opt.command = ' '.join(sys.argv)
    opt.training = bool(training)
    assert opt.training is False, 'end2end_optim only support test mode'

    opt = set_dataset_paths(opt, global_config)
    check_cuda(opt)
    check_warp_params(opt)
    check_pretrained_weights(opt)

    if opt.confirm:
        confirm_opt(opt)
    else:
        print_opt(opt)

    return opt
