# Introduction

This repo for extracting tracking data from broadcast TV feed using AI to analyze football matches in 3 main steps

### 1 - Players / ball **Detection**

we used YOLOv5 PyTorch Hub inference, to load pretrained **YoloV5l** model,
<br > for more visit:
[ultralytics/yolov5](https://github.com/ultralytics/yolov5)
<img src="./readme/det.png" alt="Detection" width="400"/>

### 2 - Players / ball **Tracking**

our tracker was built on top of **Deep SORT** algorithm, <br > which was implemented in [nwojke/deep_sort](https://github.com/nwojke/deep_sort) <br >
<img src="./readme/tr.JPG" alt="Tracking" width="50"/>

### 3 - **Projection** to a 2D pitch

we used resnet model trained on many footages of football pitches from [sportsfield_release](https://github.com/vcg-uvic/sportsfield_release) to generate projection matrix to map the TV video into 2D 1080\*680 football pitch<br >
<img src="./readme/mp1.png" alt="Projection1" width="400"/>
<img src="./readme/mp2.png" alt="Projection2" width="400"/>

# Installation

    #clone the repo
    git clone https://github.com/JooZef315/football-tracking-data-from-TV-broadcast
    #go to the directory
    cd football-tracking-data-from-TV-broadcast
    #install the dependencies
    pip install -r requirements.txt # install

# Usage

    finally after step 3 the tracking data of the TV footage is generated, and could by used to get whatever statistics and insights we want.

<br ><img src="./readme/df.JPG" alt="df" width="400"/>

# License

This project is licensed under the **MIT** - see the LICENSE file for details
