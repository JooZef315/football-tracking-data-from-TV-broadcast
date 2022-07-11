import time
import cv2
import torch
import pandas as pd
from .utils.pitch import *
from .utils.visualization import *
from .setup import initialize_model

e2e = initialize_model()


def get_frames_from_video(video_path):  
    '''
    this functions is to extract frames out of a video.

    Parameters
    ----------
    video_path : string
        the path of the directory of the video to process.
        
    Return
    ----------
    video_frames : list
        list of frames. 
    '''
    vid = cv2.VideoCapture(video_path) # detect on video
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    print(f'fps = {fps}')
    video_frames = []  
    while True:        
        _, frame = vid.read()
        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.resize(original_frame, (1280,720))            
            video_frames.append(original_frame)            
        except:
            break   

    return video_frames



def apply_projection_matrix(matrix, pointX, pointY):
    '''
    this functions is to apply projection matrix on a single point to map it to the 2D pitch.

    Parameters
    ----------
    matrix : tensor
        the projection matrix.
    pointX: int
        x coordinate of the point.
    pointY: int
        x coordinate of the point.
        
    Return
    ----------
    x_warped: float
        x coordinate of the point after mapping.
    y_warped: float
        x coordinate of the point after mapping. 
    '''
    x = torch.tensor(pointX / 1280 - 0.5).float()
    y = torch.tensor(pointY / 720 - 0.5).float()
    xy = torch.stack([x, y, torch.ones_like(x)])
    xy_warped = torch.matmul(matrix, xy) 
    xy_warped, z_warped = xy_warped.split(2, dim=1)
    # we multiply by 2, since our homographies map to
    # coordinates in the range [-0.5, 0.5] (the ones in our GT datasets)
    xy_warped = 2.0 * xy_warped / (z_warped)
    x_warped, y_warped = torch.unbind(xy_warped, dim=1)
    # [-1, 1] -> [0, 1]
    x_warped = (x_warped.item() * 0.5 + 0.5) * 1050
    y_warped = (y_warped.item() * 0.5 + 0.5) * 680

    return x_warped, y_warped



def create_MappingDataFrame(df, video_path, pitch_path = './projection_2D/data/pitch_template.png', viz = False): 
    '''
    this functions is to create the finial tracking data dataframe

    Parameters
    ----------
    df : dataframe
        the initial dataframe.
    video_path : string
        the path of the directory of the video to process.
    pitch_path : string
        the path of the directory of the pitch image to map into. 
    viz : boolen
        whether or not to show the mapping frame by frame while processing .     

    Return
    ----------
    df_mapped : dataframe
        the final mapped dataframe.  
    ''' 

    start_time =  time.time()
    frames = get_frames_from_video(video_path)
    print(f'length of the dataframe to map: {len(df)}') 
    print(f'length of frames: {len(frames)}') 

    df = df[df['frame'] % 6 == 0]

    frames_idx = df['frame'].unique().tolist()
    print(f'frames to map: {frames_idx}')
    print(f'number of frames to map: {len(frames_idx)}')
    projection_matrices = []

    #get the transformation matrix for each frame
    pitch = initialize_MappingPitch(pitch_path, viz = False)   
    for frame_id in frames_idx:
        frame = frames[frame_id - 1]    
        initFrame = initialize_MappingFrame(frame, False)
        _, mapping_model = e2e.optim(initFrame[None], pitch, refresh = True)
        projection_matrices.append(mapping_model.cpu())
    
        if viz:      
            visualize_overlaying(frame, pitch_path, mapping_model)
            
    print(f'\n ---------- time to calculate transformation matrices : {round((time.time() - start_time) / 60, 2)} min ---------- \n')

    #apply the homography model on x, y
    x_new = []
    y_new = []
    for i in range(len(df)):    
        x = df.iloc[i][2]
        y = df.iloc[i][3]    
        projection_matrix_idx = frames_idx.index(df.iloc[i][0])
        projection_matrix = projection_matrices[projection_matrix_idx]
        x_maapped,y_mapped = apply_projection_matrix(projection_matrix,x,y)    
        x_new.append(round(x_maapped))
        y_new.append(round(y_mapped))   
    
    #create the new DataFrame
    mapped_data = {'frame':df['frame'].to_list(),
            'ID':df['ID'].to_list(),
            'X':x_new,
            'Y':y_new,
            'Class':df['Class'].to_list(),
            'Color': df['Color'].to_list(),
            'Position': df['Position'].to_list()
            }
    df_mapped = pd.DataFrame(mapped_data)

    # to exculde any tracked pbject outside the pitch    
    df_mapped = df_mapped[(df_mapped['X'] >= 0) & (df_mapped['X'] <= 1050)]  
    df_mapped = df_mapped[(df_mapped['Y'] >= 0) & (df_mapped['Y'] <= 680)]  

    df_mapped.reset_index(drop=True, inplace=True)
    print(f'\n length of mapped dataframe: {len(df_mapped)}')
    print(f'---------- total time of 2D pitch projection is : {round((time.time() - start_time) / 60, 2)} mins ---------- \n')

    return df_mapped


