def clean_tracks(tracking_frames, tracking_boxes, ball_only = True):
  """
    function to:
    ----------
    1 - remove the zoomed in frames from the video
    2 - improve ball tracking by using a separate model
    3 - remove any object that is not completely visible in the frame 

    Parameters
    ----------
    tracking_frames : list
        list of frames of the input video with objects tracked. 
    tracking_boxes :list
        list of every object tracked in every input frame.      
         object:[y1, x1, y2, x2, class of the object, id of the object]
         where y1, x1, y2, x2 are the coordinates of the box around the object
    ball_only : boolen
        whether or not to save only the frames with the ball detected in them.
     
    Return 
    ----------
    new_tframes : list
        list of frames of the cleaned video with objects tracked. 
    new_tboxes :list
        list of every object tracked in every cleaned frame.      
         object:[y1, x1, y2, x2, class of the object, id of the object]
         where y1, x1, y2, x2 are the coordinates of the box around the object
  """

  print(f'length of the original frames = {len(tracking_frames)}')
  new_tframes = []
  new_tboxes = []

  out_boxes = 0 
  for boxes, frame in zip(tracking_boxes, tracking_frames):
    zoom = False
    for box in boxes[:]:
      if (box[3] - box[1]) > 160:
        zoom = True
        break    
      if not (0 <= box[0] <= 1280) or not (0 <= box[2] < 1280) or not (0 <= box[1] <= 720)  or not (0 <= box[3] < 720):    
          boxes.remove(box)
          out_boxes += 1
    
    if not zoom:
      new_tframes.append(frame)
      new_tboxes.append(boxes)

  print(f'\n-------------------------------------------------------')
  print(f'number of outliers removed = {out_boxes}')

  print(f'length of zoomed out frames = {len(new_tframes)}')

  #removing the frames that ball is not tracked in
  if ball_only:
    bframes = []
    bboxes = []
    for frame, boxes in zip(new_tframes, new_tboxes): 
      if 32 in [b[-1] for b in boxes]:
        bframes.append(frame)
        bboxes.append(boxes)

    print(f'length of frames only with ball tracked = {len(bframes)}')
    print(f'the final zoomed out cut = {round(len(bframes)/len(tracking_frames), 2)*100}% of the original')
    print(f'-------------------------------------------------------\n')
    return  bframes, bboxes

  else:    
    print(f'length of the final zoomed out frames (with and without the ball) = {len(new_tframes)}')
    print(f'the final zoomed out cut = {round(len(new_tframes)/len(tracking_frames), 2)*100} % of the original')

    return  new_tframes, new_tboxes