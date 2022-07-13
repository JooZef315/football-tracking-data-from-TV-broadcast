import cv2
import numpy as np

def classify_teams(player, color): 
  '''
  this functions is to classify players to teams based on thier colors.

  Parameters
  ----------
  player : image of the player
    the dataframe of the objects.
  colors : list of strings         
    list of the colors to choose the player color from.   
    the colors available are : [white, black, red, blue, green, yellow, purple, skyblue] 
  
  Return
  ----------
  player choosen color: string      
  '''
  
  player = cv2.cvtColor(player, cv2.COLOR_RGB2HSV)
  masks = []  
  Hsv_boundaries = [
      ([0,0,185],[180,80,255], 'white'), #white
      ([0,0,0],[180,255,65], 'black'), #black
      ([0,120,20],[15,255,255], 'red'), #red
      ([163,100,20],[180,255,255], 'red'), #dark red/ pink
      ([90,100,20],[130,255,255], 'blue'), #blue
      ([50,100,20],[80,255,255], 'green'), #green
      ([17,150,20],[35,255,255], 'yellow'), #yellow       
      ([131,60,20],[162,255,255], 'purple'), #purple          
      ([80,36,20],[105,255,255], 'skyblue'), #skyblue   
      ]  
  
  filtered_boundaries = list(filter(lambda boundary: boundary[2] in color, Hsv_boundaries))  
  for boundary in filtered_boundaries:
    mask = cv2.inRange(player, np.array(boundary[0]) , np.array(boundary[1]))
    count = np.count_nonzero(cv2.bitwise_and(player, player, mask = mask))
    masks.append(count)  
 
  color_idx = masks.index(max(masks)) 

  return filtered_boundaries[color_idx][2]