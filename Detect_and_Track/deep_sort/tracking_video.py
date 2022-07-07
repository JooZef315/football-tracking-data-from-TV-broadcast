import cv2
import time
from .utils import draw_bbox

YOLO_COCO_CLASSES = "./Detect_and_Track/model_data/coco/coco.names"

def make_tracking_video(frames, tboxes, output_path, fps, draw = True):
  '''
  this functions is to create video with objects tracked, each object has its bounding box. 

  Parameters
  ----------
  frames : list
      list of frames to create the video of. 
  tboxes : list
      list of coordinates of the bounding box of each object.  
  output_path : string
      the path of the directory to save the output video.
  fps: int
      numder of frames per second of the output video.
  draw: boolen
      whether or not to draw the  bounding boxes.
  '''

  codec = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter(output_path, codec, fps, (1280,720)) # output_path must be .mp4
  times= []
  for frame, tracked_bboxes in zip(frames, tboxes):
    if draw: 
      t1 = time.time()    
        # draw detection on frame
      img = draw_bbox(frame, tracked_bboxes, CLASSES=YOLO_COCO_CLASSES, tracking=True)
      t2 = time.time()
      times.append(t2-t1)   
      
      times = times[-20:]   

      ms = sum(times)/len(times)*1000
      fps = 1000 / ms    
      
      img = cv2.putText(img, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

      print("Time: {:.2f}ms,  total FPS: {:.1f}".format(ms, fps))
      out.write(img[:, :, ::-1])

    else:      
      out.write(frame[:, :, ::-1])