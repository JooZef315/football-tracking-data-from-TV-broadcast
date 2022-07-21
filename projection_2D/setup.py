import torch
from .utils import utils,  constant_var
from .models.end_2_end_optimization import End2EndOptimFactory
from .options import fake_options

def initialize_model():
    # if want to run on CPU,  make it False
    print(f'cuda: {constant_var.HAS_CUDA}')
    constant_var.USE_CUDA = True
    utils.fix_randomness()

    # if GPU is RTX 20XX, disable cudnn
    torch.backends.cudnn.enabled = True

    # set some options
    opt = fake_options.FakeOptions()
    opt.batch_size = 1
    opt.coord_conv_template = True
    opt.error_model = 'loss_surface'
    opt.error_target = 'iou_whole'
    opt.goal_image_path = './Data/Capture.JPG'
    opt.guess_model = 'init_guess'
    opt.homo_param_method = 'deep_homography'
    opt.load_weights_error_model = './pretrained_loss_surface'
    opt.load_weights_upstream = './pretrained_init_guess'
    opt.lr_optim = 1e-5
    opt.need_single_image_normalization = True
    opt.need_spectral_norm_error_model = True
    opt.need_spectral_norm_upstream = False
    opt.optim_criterion = 'l1loss'
    opt.optim_iters = 200
    opt.optim_method = 'stn'
    opt.optim_type = 'adam'
    opt.out_dir = './projection_2D/pretrained_weights/out/'   
    opt.prevent_neg = 'sigmoid'
    opt.template_path = './projection_2D/data/pitch_template.png'
    opt.warp_dim = 8
    opt.warp_type = 'homography'

    e2e = End2EndOptimFactory.get_end_2_end_optimization_model(opt)
    return e2e