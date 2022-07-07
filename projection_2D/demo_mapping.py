import imageio
from .utils.pitch import *
from .utils.visualization import *
from .setup import initialize_model

def demo_mapping(img_path = './Data/Capture.JPG'):
    '''
    this functions is to make a test case on a single image.

    Parameters
    ----------
    img_path : string
        the path of the directory of the image to process.
    '''

    pitch_path = './projection_2D/data/pitch_template.png'
    pitch = initialize_MappingPitch(pitch_path, viz = True)
    frame = imageio.imread(img_path, pilmode='RGB')
    norm_frame = initialize_MappingFrame(frame, viz = True)
    e2e = initialize_model()
    _, optim_homography = e2e.optim(norm_frame[None], pitch)
    print(f'the 2D projection matrix is: {optim_homography.cpu()}')
    visualize_overlaying(frame, pitch_path, optim_homography)

