import abc

import numpy as np

from ..utils import utils
from . import init_guesser


def get_homography_between_corners_and_default_canon4pts(corners, canon4pts_type: str):
    batch_size = corners.shape[0]
    if canon4pts_type == 'lower':
        lower_canon4pts = get_default_canon4pts(batch_size, canon4pts_type)
        homography = utils.get_perspective_transform(lower_canon4pts, corners)
    elif canon4pts_type == 'full':
        full_canon4pts = get_default_canon4pts(batch_size, canon4pts_type)
        homography = utils.get_perspective_transform(full_canon4pts, corners)
    else:
        raise ValueError('unknown canon4pts type')
    return homography


def get_default_canon4pts(batch_size, canon4pts_type: str):
    if canon4pts_type == 'lower':
        lower_canon4pts = utils.LOWER_CANON4PTS_NP()
        lower_canon4pts = np.tile(lower_canon4pts, (batch_size, 1, 1))
        lower_canon4pts = utils.to_torch(lower_canon4pts)
        return lower_canon4pts
    elif canon4pts_type == 'full':
        full_canon4pts = utils.FULL_CANON4PTS_NP()
        full_canon4pts = np.tile(full_canon4pts, (batch_size, 1, 1))
        full_canon4pts = utils.to_torch(full_canon4pts)
        return full_canon4pts
    else:
        raise ValueError('unknown canon4pts type')


class HomographyInferenceFactory(object):
    @staticmethod
    def get_homography_inference(opt):
        if opt.homo_param_method == 'deep_homography':
            homography = HomographyInferenceDeepHomo(opt)
        else:
            raise ValueError('unknown homography parameterization: {0}'.format(opt.homo_param_method))
        return homography


class HomographyInference(abc.ABC):
    '''homography inference engine
    because we have different homography parameterization,
    so we want to make it a common interface
    '''

    def __init__(self, opt):
        self.opt = opt
        self.check_options()
        self.build_models()

    def check_options(self):
        if self.opt.guess_model != 'init_guess':
            content_list = []
            content_list += ['HomographyInference currently only support init_guess as upstream']
            utils.print_notification(content_list, 'ERROR')
            exit(1)

    def build_models(self):
        # self.upstream = init_guesser.InitialGuesser(self.opt)
        self.upstream = init_guesser.InitialGuesserFactory.get_initial_guesser(
            self.opt)
        self.upstream = utils.set_model_device(self.upstream)
        self.upstream.eval()

    def refresh(self):
        self.upstream.load_pretrained_weights()

    def get_upstream_params(self):
        assert self.opt.guess_model == 'init_guess'
        return self.upstream.parameters()

    def get_training_status(self) -> bool:
        return self.upstream.training

    @abc.abstractmethod
    def infer_upstream_homography(self, frame):
        pass

    @abc.abstractmethod
    def infer_upstream_corners(self, frame):
        pass


class HomographyInferenceDeepHomo(HomographyInference):

    def infer_upstream_corners(self, frame):
        self.upstream.eval()
        inferred_corners_orig = self.upstream(frame)
        inferred_corners_orig = inferred_corners_orig.reshape(-1, 2, 4)
        inferred_corners_orig = inferred_corners_orig.permute(0, 2, 1)
        return inferred_corners_orig

    def infer_upstream_homography(self, frame):
        batch_size = frame.shape[0]
        inferred_corners_orig = self.infer_upstream_corners(frame)
        lower_canon4pts = get_default_canon4pts(batch_size, canon4pts_type='lower')
        homography = utils.get_perspective_transform(lower_canon4pts, inferred_corners_orig)
        return homography
