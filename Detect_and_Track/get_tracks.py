from .deep_sort.tracker_implementation import trackingXl5
from .deep_sort.tracks_cleaning import clean_tracks
from .deep_sort.tracking_video import make_tracking_video
from .yoloV5.load_models import yoloV5l

def get_video_tracks(video_path , out_name , ball_only = True):   
    '''
    this functions is to implement tracking players and the ball in the video.

    Parameters
    ----------
    video_path : string
        the path of the directory of the processed video.
    out_name : string
        the name to save the video with objects tracked with.  
    ball_only : boolen
        whether or not to save only the frames with the ball detected in them. 
    
    Return
    ----------
    ziframes : list
        list of frames of the video with objects tracked. 
     zitboxes :list
        list of every object tracked in every frame.      
         object:[y1, x1, y2, x2, class of the object, id of the object]
         where y1, x1, y2, x2 are the coordinates of the box around the object
    '''

    modelv5l, ball_modelv5l = yoloV5l()
    iframes, itboxes, fps = trackingXl5(modelv5l, ball_modelv5l, video_path)
    ziframes, zitboxes = clean_tracks(iframes, itboxes, ball_only)
   
    #make clean video
    make_tracking_video(ziframes, zitboxes, f'./Out/{out_name}_out.mp4', fps, draw = False)

    return ziframes, zitboxes


