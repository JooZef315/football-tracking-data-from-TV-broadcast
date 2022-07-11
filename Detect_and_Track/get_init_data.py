import pandas as pd
from .init_dataframe.create_df import creatInitDataFrame
from .get_tracks import get_video_tracks

def get_init_data(path, out_name, teams_colors, ball_only = True):
    '''
    this functions is to create initial unmapped dataframe

    Parameters
    ----------
    path : string
        the path of the directory of the processed video.
    out_name : string
        the name to save the video with objects tracked with.   
    teams_colors : list of strings
        list of the colors to classify the teams. 
    ball_only : boolen
        whether or not to save only the frames with the ball detected in them.     

    Return
    ----------
    init_df : pandas dataframe
        the initial unmapped dataframe.  
    '''
        
    ziframes, zitboxes = get_video_tracks(video_path = path , out_name = out_name, ball_only = ball_only)
    init_df = creatInitDataFrame(zitboxes, ziframes, teams_colors)

    #save the initial dataframe
    init_df.to_csv(f'./Out/{out_name}_init_df.csv', index=False)

    return init_df