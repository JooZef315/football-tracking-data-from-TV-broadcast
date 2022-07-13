import pandas as pd
from .teams_classification import classify_teams
from .players_positions import get_positions

def creatInitDataFrame(tracks, frames, frame_colors = None):  
  '''
  this functions is to create initial dataframe, with players coordinates relative to TV video, 
  and classify them according to teams colors.

  Parameters
  ----------
  tracks : dataframe
    list of every object tracked in every frame.      
    object:[y1, x1, y2, x2, class of the object, id of the object]
    where y1, x1, y2, x2 are the coordinates of the box around the object    
  frames : dataframe
    list of frames of the video with objects tracked.
  frame_colors : list of strings         
    list of the colors to classify the teams in that order:
      [Defense team color, Defense goalkeeper color, Attack team color, Attack goalkeeper color, referee color, ball color]
      or None.
  
  Return
  ----------
  df_points : dataframe
      the dataframe of the the tracked objects.
      consists of 7 columns:
      1 - frame: the index of the frame in the video
      2 - ID: the ID of the tracked object in the frame
      3 - X: the X coordinate of the lower center of the object
      4 - Y: the Y coordinate of the lower center of the object
      5 - Class: the type of the tracked object (ball or player)
      6 - Color: the color of the team of the tracked player
      7 - position: the position of the tracked object (Attack - Defense - ball - Not a player)
  '''

  frame_list = []
  X_list = []
  Y_list = []
  ID = []
  Classes = []
  colors = []    
  for frame_idx, frame_tracks in enumerate(tracks):
    for track in frame_tracks:
      x1 = track[0]
      x2 = track[2]
      y1 = track[1]
      y2 = track[3]

      frame_list.append(frame_idx + 1)
      X = (x1 + x2)/2      
      X_list.append(round(X))
      Y_list.append(y2)     
      ID.append(track[4])
      Classes.append(track[5])

      obj = frames[frame_idx][y1 : y2+1, x1 : x2+1, :] 
      if obj.tolist() == []:
        print(f'frame_idx {frame_idx}')
        print(f'track {track}')
        obj = frames[frame_idx][y2:y1 , x2:x1, :] 

      # frame_colors = [def, def goalkeeper, att, att goalkeeper, ref, ball]
      if track[5] == 32:
        colors.append(frame_colors[-1])
      else:        
        color = classify_teams(obj, frame_colors[:-1])
        colors.append(color)   

  data = {'frame':frame_list,
        'ID':ID,
        'X':X_list,
        'Y':Y_list,
        'Class':Classes,
        'Color':colors}

  df_points = pd.DataFrame(data)  
  df_points['Class'] = df_points['Class'].replace([0],'Player')
  df_points['Class'] = df_points['Class'].replace([32],'Ball')     
  df_points.reset_index(drop=True, inplace=True)

  df_points = get_positions(df_points, frame_colors)

  return df_points