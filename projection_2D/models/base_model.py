'''abstract base model
'''

import os
import abc

import torch

from ..utils import utils


class BaseModel(abc.ABC):
    '''base model for resnet-18 based network
    '''

    @abc.abstractmethod
    def check_options(self):
        pass

    def load_pretrained_weights(self):
        '''load pretrained weights
        this function can load weights from another model.
        '''
        # 1. load check point
        checkpoint_path = self._get_checkpoint_path()
        checkpoint = self._load_checkpoint(checkpoint_path)

        # 2. verify check point
        self._verify_checkpoint(checkpoint)

        # 3. try loading weights
        key_name = 'model_state_dict'
        saved_weights = checkpoint[key_name]
        try:
            self.load_state_dict(saved_weights)
        except RuntimeError:
            # handling the DataParallel weights problem
            try:
                weights = saved_weights
                weights = {k.replace('module.', ''): v for k,
                           v in weights.items()}
                self.load_state_dict(weights)
            except RuntimeError:
                try:
                    weights = saved_weights
                    weights = {'module.' + k: v for k, v in weights.items()}
                    self.load_state_dict(weights)
                except RuntimeError:
                    content_list = []
                    content_list += [
                        'Cannot load weights for {0}'.format(self.name)]
                    utils.print_notification(content_list, 'ERROR')
                    exit(1)

        # 4. loaded
        content_list = []
        content_list += ['Weights loaded for {0}'.format(self.name)]
        content_list += ['From: {0}'.format(checkpoint_path)]
        utils.print_notification(content_list)

    @abc.abstractmethod
    def _get_checkpoint_path(self):
        pass

    def _load_checkpoint(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        else:
            print(checkpoint_path)
            raise FileNotFoundError('model check point cannot found: {0}'.format(checkpoint_path))
        return checkpoint

    @abc.abstractmethod
    def _verify_checkpoint(self, checkpoint):
        pass

    def print_resnet_config(self, resnet_config):
        content_list = []
        content_list += ['Resnet backbone config for {0}'.format(self.name)]
        content_list += ['Spectral norm for resnet: {0}'.format(
            resnet_config.need_spectral_norm)]
        if resnet_config.group_norm == 0:
            content_list += ['Using BN for resnet']
        else:
            content_list += ['Using GN for resnet, number of groups: {0}'.format(resnet_config.group_norm)]
        content_list += ['Imagenet pretrain weights for resnet: {0}'.format(
            resnet_config.pretrained)]
        utils.print_notification(content_list)
