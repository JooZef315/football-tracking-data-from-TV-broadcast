import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import cv2
import matplotlib.pyplot as plt
import numpy as np
from .utils import *

YOLO_COCO_CLASSES = "./Detect_and_Track/model_data/coco/coco.names"

def detectXl5(model ,image_path, show=True):
    '''
    this functions is to detect objects in an image using YoloyV5 models:
    
    Parameters
    ----------
    model : pytorch model
        pytorch YoloV5l model.  
    image_path : string
        the path of the directory of the processed image.
    show : boolen
      whether or not to draw the bounding boxes.
     
    Return 
    ----------
    image : image
        image with objects tracked. 
    nbboxes :list
        list of every object tracked in the image.      
         object:[y1, x1, y2, x2, class of the object]
         where y1, x1, y2, x2 are the coordinates of the box around the object
    '''
    #image
    original_image = cv2.imread(image_path)    
    original_image = cv2.resize(original_image, (1280,720))    
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)    

    # Inference
    results = model(original_image)
    pred_bbox = results.xyxy[0].tolist()
    bboxes = [np.array(box) for box in pred_bbox]
    print(f'image shape {original_image.shape[:-1]}')
    print(f'number of objects detected = {len(bboxes)}')

    if show:
      # Show the image
      image = draw_bbox(original_image, bboxes, CLASSES=YOLO_COCO_CLASSES, rectangle_colors=(255,0,0))  
      # cv2.imshow(image[:, :, ::-1])
      plt.figure(figsize=(12, 8))
      plt.imshow(image)
      plt.show()
    else:      
      image =  original_image   

    nbboxes = [[int(round(b)) if b != list(bbox)[4] else round(b, 2) for b in list(bbox)] for bbox in bboxes]    
    return image, nbboxes