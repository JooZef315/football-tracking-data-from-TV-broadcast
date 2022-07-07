# import cv2
import matplotlib.pyplot as plt
from .get_tracks import get_video_tracks

def track_demo(video_path = "./Data/benz.mp4" , out_name = 'benz'):
    '''
    this functions is to apply tracking on a video, and track certain player within 10 frames

    Parameters
    ----------
    video_path : string
        the path of the directory of the processed video.
    out_name : string
        the name to save the video with objects tracked with.   . 
    '''

    iframes, itboxes = get_video_tracks(video_path, out_name = out_name)
    plt.figure(figsize=(12, 8))
    frame_num = 30
    print(len(itboxes))
    print(len(iframes))
    print(iframes[frame_num].shape)
    # cv2.imshow(iframes[frame_num][:,:,::-1])
    plt.imshow(iframes[frame_num])
    plt.show()
    
    t10 = itboxes[frame_num]
    x = []
    for t in t10:
        x.append([round(b) for b in t])

    print(x)
    xx = x[2]
    # cv2.imshow(iframes[frame_num][xx[1]:xx[3],xx[0]:xx[2],  ::-1])
    plt.imshow(iframes[frame_num][xx[1]:xx[3],xx[0]:xx[2],  :])
    plt.show()

    for i in range(frame_num, frame_num + 10):
        t10 = itboxes[i]
        x = []
        for t in t10:
            x.append([round(b) for b in t])

        xx = [xx for xx in x if xx[4] == 1][0]
        # cv2.imshow(iframes[i][xx[1]:xx[3],xx[0]:xx[2],  ::-1])
        plt.imshow(iframes[i][xx[1]:xx[3],xx[0]:xx[2],  :])
        plt.show()
    

