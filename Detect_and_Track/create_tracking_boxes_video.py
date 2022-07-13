from .deep_sort.tracker_implementation import trackingXl5
from .deep_sort.tracks_cleaning import clean_tracks
from .deep_sort.tracking_video import make_tracking_video
from .yoloV5.load_models import yoloV5l

def create_tracking_boxes_video(video_path , out_name):
    '''
    this functions is to make video with objects tracked.

    Parameters
    ----------
    video_path : string
        the path of the directory of the processed video.
    out_name : string
        the name to save the video with objects tracked with.  
    '''

    modelv5l, ball_modelv5l = yoloV5l()
    iframes, itboxes, fps = trackingXl5(modelv5l, ball_modelv5l, video_path)
    
    make_tracking_video(iframes, itboxes, f'./Out/{out_name}_out_tracked.mp4', fps, draw = True)
