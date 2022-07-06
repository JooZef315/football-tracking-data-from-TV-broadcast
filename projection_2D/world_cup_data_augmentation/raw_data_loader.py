'''load raw world cup dataset
'''

import sys
sys.path.append("..")
import os
import json

import numpy as np
import torch
import imageio
import scipy.io as sio
from PIL import Image

from utils import utils, warp

class RawDataloader():

    def __init__(self, data_type):
        self.get_path_from_json()
        self.data_type = data_type
        self.GRASS_INDEX = 'grass'
        self.load_template()
        self.frame_h = 720
        self.frame_w = 1280
        self.template_h = 74
        self.template_w = 115
        self.output_image_size = (256, 256)

    def get_path_from_json(self):
        with open('./config.json', 'r') as f:
            config = json.load(f)
            self.data_dir = config['world_cup_raw_dir']
            self.template_path = config['world_cup_template_path']
        assert os.path.isdir(self.data_dir)
        assert os.path.isfile(self.template_path)

    def load_template(self):
        self.template_np = imageio.imread(self.template_path, pilmode='RGB') / 255.0
        self.template_torch = utils.to_torch(imageio.imread(self.template_path, pilmode='RGB') / 255.0).permute(2, 0, 1)
        self.template_torch = torch.mean(self.template_torch, 0, keepdim=True)
        x_coord, y_coord = torch.meshgrid([torch.linspace(-1, 1, steps=self.template_torch.shape[-2]), torch.linspace(-1, 1, steps=self.template_torch.shape[-1])])
        x_coord = x_coord.to(self.template_torch.device)[None]
        y_coord = y_coord.to(self.template_torch.device)[None]
        self.template_torch = torch.cat([self.template_torch, x_coord, y_coord], dim=0)

    def get_image_path_by_id(self, image_id):
        image_path = os.path.join(self.data_dir, self.data_type, str(image_id)+'.jpg')
        return image_path

    def get_segmentation_path_by_id(self, seg_id):
        seg_path = os.path.join(self.data_dir, self.data_type, str(seg_id)+'_grass_gt.mat')
        return seg_path
    
    def get_homography_path_by_id(self, homo_id):
        homo_path = os.path.join(self.data_dir, self.data_type, str(homo_id)+'.homographyMatrix')
        return homo_path

    def _get_np_image_by_path(self, image_path):
        image = imageio.imread(image_path, pilmode='RGB') / 255.0
        # reverse channels order, because old dataset did this way
        image = image[..., [2, 1, 0]]
        return image

    def _get_np_seg_by_path(self, seg_path):
        seg_mat = sio.loadmat(seg_path)
        seg_image = seg_mat[self.GRASS_INDEX]
        return seg_image

    def get_homography_by_path(self, homo_path):
        def get_frame_space_scaling_homography():
            src_4pts = utils.to_torch(utils.FULL_CANON4PTS_NP())
            dest_4pts = utils.to_torch(np.array([[0, 0], [0, self.frame_h], [self.frame_w, self.frame_h], [self.frame_w, 0]], dtype=np.float32))
            scaling_transformation = utils.get_perspective_transform(src_4pts[None], dest_4pts[None])
            scaling_transformation = utils.to_numpy(scaling_transformation[0])
            return scaling_transformation

        def get_template_space_scaling_homography():
            src_4pts = utils.to_torch(np.array([[0, 0], [0, self.template_h], [self.template_w, self.template_h], [self.template_w, 0]], dtype=np.float32))
            dest_4pts = utils.to_torch(utils.FULL_CANON4PTS_NP())
            scaling_transformation = utils.get_perspective_transform(src_4pts[None], dest_4pts[None])
            scaling_transformation = utils.to_numpy(scaling_transformation[0])
            return scaling_transformation

        homo_mat = np.loadtxt(homo_path)
        # 1. bring [+-0.5] -> [720, 1280]
        frame_scaling = get_frame_space_scaling_homography()
        homo_mat = np.matmul(homo_mat, frame_scaling)
        # 2. bring [115, 74] -> [+-0.5]
        template_scaling = get_template_space_scaling_homography()
        homo_mat = np.matmul(template_scaling, homo_mat)
        homo_mat = homo_mat / homo_mat[2, 2]
        return homo_mat

    def get_np_seg_by_id(self, seg_id):
        seg_path = self.get_segmentation_path_by_id(seg_id)
        image = self._get_np_seg_by_path(seg_path)
        return image

    def get_np_image_by_id(self, image_id):
        image_path = self.get_image_path_by_id(image_id)
        image = self._get_np_image_by_path(image_path)
        return image

    def get_homography_by_id(self, homo_id):
        homo_path = self.get_homography_path_by_id(homo_id)
        homo_mat = self.get_homography_by_path(homo_path)
        return homo_mat

    def get_warped_tmp_by_id(self, data_id):
        homo_mat = self.get_homography_by_id(data_id)
        warped_tmp = warp.warp_image(self.template_torch, utils.to_torch(homo_mat), out_shape=(self.frame_h, self.frame_w))
        warped_tmp = utils.torch_img_to_np_img(warped_tmp[0])
        return warped_tmp

    def get_paired_data_by_id(self, index):
        frame = self.get_np_image_by_id(index)
        assert frame.shape == (720, 1280, 3)
        frame = np.asarray(Image.fromarray((frame * 255.0).astype(np.uint8)).resize(self.output_image_size, resample=Image.BILINEAR)) / 255.0
        homography = self.get_homography_by_id(index)
        return frame, homography
