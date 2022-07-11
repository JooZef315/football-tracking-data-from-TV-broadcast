import pandas as pd
from .mapping import get_frames_from_video, create_MappingDataFrame
from .utils.visualization import visualize_mapping

def get_tracking_data(df_path, tracked_video_path, out_name, pitch_path = './projection_2D/data/pitch_template.png', visualize = True ,test = False):
    '''
    this functions is to create the finial tracking data dataframe

    Parameters
    ----------
    df_path : string
        the path of the directory of the initial dataframe.
    tracked_video_path : string
        the path of the directory of the video to process.
    pitch_path : string
        the path of the directory of the pitch image to map into.
    out_name : string
        the name to save the final dataframe with.   
    visualize : boolen
        whether or not to show the 2D projection on the pitch for each frame.  
    test : boolen
        whether or not to make a test case on a single frame.    

    Return
    ----------
    df_mapped : dataframe
        the final mapped dataframe.  
    '''
    df = pd.read_csv(df_path)    
    mapped_frames = get_frames_from_video(tracked_video_path)    
    df_mapped = create_MappingDataFrame(df, tracked_video_path, pitch_path, viz = visualize)
    
    if test:
        #test on frame 65
        test_frame_idx = 65
        test_frame = mapped_frames[test_frame_idx - 1]
        print(test_frame.shape)

        test_frame_init_df = df[df['frame'] == test_frame_idx]
        test_frame_mapped_df = df_mapped[df_mapped['frame'] == test_frame_idx]
        visualize_mapping(test_frame, pitch_path, test_frame_init_df, test_frame_mapped_df, annotation = True)

    #save the final dataframe
    df_mapped.to_csv(f'./Out/{out_name}_df_mapped.csv', index=False)

    return df_mapped