import torch

def yoloV5l():
    '''
    this functions is to load YoloV5l pytorch models from torch hub

    Return 
    ----------
    modeli : pytorch model. 
        pytorch YoloV5l model.  
    ball_model : pytorch model
         pytorch YoloV5l model to detect the ball specifically.   
    '''

    modeli = torch.hub.load('ultralytics/yolov5', 'yolov5l')  
    modeli.classes = [0,32]

    ball_model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  
    ball_model.classes = [32]
    ball_model.conf = 0.15
    ball_model.max_det  = 1
    print('\n-------------------------------------------------------------------------------------------\n')

    return modeli, ball_model

