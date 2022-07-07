import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
from .utils import torch_img_to_np_img, np_img_to_torch_img
from .warp import warp_image


def visualize_overlaying(frame_draw, pitch_path, model):
    '''
    this functions is to visualize the the 2D projection of the frame from the match on the pitch.

    Parameters
    ----------
    frame_draw : image
         the frame image.
    pitch_path : string
        the path of the directory of the pitch image.
    model : pytorh model
        model to generate 2D projection matrix.  
    '''   
    # reload image and template for visualization
    # overload goal image
    frame_draw = cv2.resize(frame_draw,(1280,720))
    frame_draw = frame_draw / 255.0
    outshape = frame_draw.shape[0:2]
    print(outshape)
    # overload template image
    pitch_draw = imageio.imread(pitch_path, pilmode='RGB')
    pitch_draw = cv2.resize(pitch_draw, (1050,680))
    pitch_draw = pitch_draw / 255.0
    pitch_draw = np_img_to_torch_img(pitch_draw)

    # warp template image with optimized guess
    warped_tmp_optim = warp_image(pitch_draw, model, out_shape=outshape)[0]
    warped_tmp_optim = torch_img_to_np_img(warped_tmp_optim)
    plt.imshow(warped_tmp_optim)
    plt.show()

    # show optimized guess overlay
    show_image = np.copy(frame_draw)
    valid_index = warped_tmp_optim[:, :, 0] > 0.0
    overlay = (frame_draw[valid_index].astype('float32') + warped_tmp_optim[valid_index].astype('float32'))/2
    show_image[valid_index] = overlay
    plt.imshow(show_image)
    plt.show()


def visualize_mapping(img, pitch_path, df, mapped_df, annotation = True):  
    '''
    this functions is to visualize all the players and the ball in the frame before and after the 2D mapping on the pitch.

    Parameters
    ----------
    img : image
         the frame image.
    pitch_path : string
        the path of the directory of the pitch image.
    df : dataframe.
        the unmapped initial dataframe. 
    mapped_df : dataframe.
        the mapped objects dataframe. 
    annotation: boolen
        whether or not to draw the IDs of the players.
    ''' 

    img = cv2.resize(img,(1280,720))
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    for i in range(len(df)):
        if df.iloc[i,4] == 'Ball':
            plt.scatter(df.iloc[i,2], df.iloc[i,3], s= 35, c = df.iloc[i,5])
        else:  
            plt.scatter(df.iloc[i,2], df.iloc[i,3], s= 75, c = df.iloc[i,5])
            if annotation:
                plt.annotate('ID: '+str(df.iloc[i][1]), (df.iloc[i][2] + 8, df.iloc[i][3] + 8), fontsize=14)  
    plt.show() 

    # overload template image
    pitch_draw = imageio.imread(pitch_path, pilmode='RGB')
    pitch_draw = cv2.resize(pitch_draw, (1050,680))
    pitch_draw = pitch_draw / 255.0
    pitch_draw = np_img_to_torch_img(pitch_draw)
    plt.figure(figsize=(12, 8))
    plt.imshow(torch_img_to_np_img(pitch_draw)*0.5)
    for i in range(len(mapped_df)):
        if mapped_df.iloc[i,4] == 'Ball':
            plt.scatter(mapped_df.iloc[i][2], mapped_df.iloc[i][3], s= 35, c=mapped_df.iloc[i][5]) 
            if annotation:
                plt.annotate('ball', (mapped_df.iloc[i][2] + 10, mapped_df.iloc[i][3] + 10), fontsize=14)  
        else:  
            plt.scatter(mapped_df.iloc[i][2], mapped_df.iloc[i][3], s= 75, c=mapped_df.iloc[i][5])
            if annotation:
                plt.annotate('ID: '+str(mapped_df.iloc[i][1]), (mapped_df.iloc[i][2] + 10, mapped_df.iloc[i][3] + 10), fontsize=14)   
    plt.show()