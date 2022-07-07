import cv2
from .yoloV5.load_models import yoloV5l
from .yoloV5.detector import detectXl5

def detect_demo(path   = "./Data/Capture.JPG"):
    '''
    this functions is to apply detection on an image.

    Parameters
    ----------
    path : string
        the path of the directory of the processed image. 
    '''

    image_path   = path
    modeli , ball_model = yoloV5l()
    img, res = detectXl5(modeli, image_path, show=True)
    print(f'bounding boxes: {res}')

    ball_img, ball_res = detectXl5(ball_model, image_path, show=True)
    print(ball_res)

    #show one player
    player = res[5]
    print(f'player with index 5: {player}')
    im = img[player[1] - 30:player[3] + 30, player[0] - 30 :player[2] + 30, ::-1]
    cv2.imshow(im)

    return res

