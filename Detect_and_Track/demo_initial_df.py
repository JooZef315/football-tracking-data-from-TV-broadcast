import pandas as pd
import matplotlib.pyplot as plt
from .init_dataframe.create_df import creatInitDataFrame
from .yoloV5.load_models import yoloV5l
from .yoloV5.detector import detectXl5

def init_df_demo(path = "./Data/Capture.JPG", colors = []):
  '''
  this functions is to create an initial unmapped dataframe demo on a single frame/ image, 
  and plot the detected objects in the image.

  Parameters
  ----------
  path : string
      the path of the directory of the processed image. 
  '''
  image_path = path
  modelV5l, _ = yoloV5l()
  img ,detections = detectXl5(modelV5l, image_path, False)
  init_df = creatInitDataFrame([detections], [img], frame_colors = colors)
  init_df = init_df.reset_index(drop=True)

  plt.figure(figsize=(12, 8))
  plt.imshow(img)
  for i in range(len(init_df)):
    plt.scatter(init_df.iloc[i][2], init_df.iloc[i][3], s= 20, c = init_df.iloc[i][5])
    plt.annotate(init_df.iloc[i][1], (init_df.iloc[i][2] + 4, init_df.iloc[i][3] + 4), fontsize=15) 
  plt.show() 

  print(init_df)