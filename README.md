# 3D-Object-Detection
   This project use an 3D object detection using lidars data from [Waymo Open Dataset](https://waymo.com/open/). The original project files are from the related [Github repository](https://github.com/udacity/nd013-c2-fusion-starter).
   
## 1. Visualize range image channels (ID_S1_EX1)

<p align="center" >
   <img width="100%" src="media/ID_S1_EX1.gif" alt=""/> 
   <br>
   <em>Result of range image</em>
</p>
 
## 2. Visualize lidar point-cloud (ID_S1_EX2)

Write-up file where is the analysis of ID_S1_EX2 is  [HERE](Write-up.pdf). 
<p align="center" >
   <img width="60%" src="media/Frame 1.png" alt=""/>
   <img width="100%" src="media/Frame 1 (2).png" alt=""/>
   <br>
   <em>Result of point cloud visualization and range image </em>
</p>

## 3. Convert sensor coordinates to BEV-map coordinates (ID_S2_EX1)
<p align="center" >
   <img width="60%" src="media/ID_S2_EX1.png" alt=""/> 
   <br>
   <em>BEV-map coordinates Frame 0 from Waymo Dataset</em>
</p>

## 4. Compute intensity layer of the BEV map (ID_S2_EX2) and Compute height layer of the BEV map (ID_S2_EX3)

<p align="center" >
   <img width="50%" src="media/ID_S2_EX1_EX2_EX3.gif" alt=""/> 
   <br>
   <em>Result of range image</em>
</p>

## 5. Add a second model from a GitHub repo (ID_S3_EX1) and Extract 3D bounding boxes from model response (ID_S3_EX2)
Obtain a second model fpn-resnet fron [ Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds](https://github.com/maudzung/SFA3D).
| Frame 50 | Frame 51 |
| ------------- | ------------- | 
| <p align="center" > <img width="90%" src="media/ID_S3_EX2_frame 50.png" alt=""/>  <br> <em>3D bounding boxes added to the images</em> </p>  |  <p align="center" > <img width="90%" src="media/ID_S3_EX2_frame 51.png" alt=""/>  <br> <em>3D bounding boxes added to the images</em> </p> |




## 6. Compute intersection-over-union between labels and detections (ID_S4_EX1)

<p align="center" > <img width="100%" src="media/ID_S4_EX1.png" alt=""/> <br> <em>Results center_devs and IOU values from frame 50</em> </p>

## 7. Compute false-negatives and false-positives (ID_S4_EX2)
<p align="center" > <img width="100%" src="media/ID_S4_EX2.png" alt=""/> <br> <em>Results of det_performance from frame 50 </em> </p>

## 8. Compute precision and recall (ID_S4_EX3)
### With `configs_det.use_labels_as_objects = False`:

<p align="center" > <img width="60%" src="media/ID_S4_EX3_False.png" alt=""/> <br> <em>Results of Presicion and Recall from frames 50 to 150 </em> </p>

The complete results obtained are shown in this [VIDEO 1](https://youtu.be/A9yt2Kcx42I).

### With `configs_det.use_labels_as_objects = True`:

<p align="center" > <img width="60%" src="media/ID_S4_EX3_True.png" alt=""/> <br> <em>Results of Presicion and Recall from frames 50 to 150 </em> </p>

The complete results obtained are shown in this [VIDEO 2](https://youtu.be/7ahBUSe1jLc).



