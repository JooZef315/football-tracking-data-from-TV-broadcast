import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from .utils import  np_img_to_torch_img
from .image_utils import normalize_single_image, rgb_template_to_coord_conv_template


def initialize_MappingPitch(pitch_path = './projection_2D/data/pitch_template.png', viz = True):
    '''
    this functions is to preprocess normalize the pitch image to use it.

    Parameters
    ----------
    pitch_path : string
        the path of the directory of the pitch image.
    viz : boolen
        whether or not to show the pitch image after normalization.   

    Return
    ----------
    pitch : image
        the pitch image after normalization.                    
    '''
    # read template image
    pitch = imageio.imread(pitch_path, pilmode='RGB')
    pitch = cv2.resize(pitch, (1050,680))
    pitch = pitch / 255.0
    pitch = rgb_template_to_coord_conv_template(pitch)
    if viz:    
        print('size of template: {0}'.format(pitch.shape))
        plt.imshow(pitch)
        plt.show()
    # covert np image to torch image, and do normalization
    pitch = np_img_to_torch_img(pitch)
    pitch = normalize_single_image(pitch)
    if viz: 
        print('mean of template: {0}'.format(pitch.mean()))
        print('std of template: {0}'.format(pitch.std()))

    return pitch 



def initialize_MappingFrame(frame, viz = True):
    '''
    this functions is to preprocess normalize the frame to apply the 2D projection matrix on it.

    Parameters
    ----------
    frame : image
         the frame image.
    viz : boolen
        whether or not to show the frame image after normalization.   

    Return
    ----------
    pitch : frame
        the frame image after normalization.            
    '''
    frame = cv2.resize(frame, (1280,720))
    if viz:   
        plt.imshow(frame)
        plt.show()
    # resize image to square shape, 256 * 256, and squash to [0, 1]
    pil_image = Image.fromarray(np.uint8(frame))
    pil_image = pil_image.resize([256, 256], resample=Image.NEAREST)
    frame = np.array(pil_image)
    if viz:   
        plt.imshow(frame)
        plt.show()
    # covert np image to torch image, and do normalization
    frame = np_img_to_torch_img(frame)
    frame = normalize_single_image(frame)

    return frame   