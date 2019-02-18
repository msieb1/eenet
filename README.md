### README

## TL;DR

The model predicts a probabilistic heatmap of the locations of the left and right finger tip of the end effector of the Baxter robot.
In the images as they are shown during training, the left fingertip prediction is shown as a red dot (ground truth as red triangle),
and the right fingertip detection is shown as a blue dot (ground truth as blue triangle).

### 1. Setup EENet
```
# download venv and data

source venv/bin/activate


# (install missing packages if nec)

```
### 2. Setup Environment


### 3.1 Training of EE Finger Tip Detection
Look at parsed arguments within scripts/train_eenet.py for more details.

```

python3 scripts/train_eenet.py --root_dir data_storage_path/0_view0 -sf

# example: 
python3 scripts/train_eenet.py --root_dir 
/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/imitation_learning/endeffector4/videos/train/0_view0 -sf
```


### 3.2 EE Data Collection

Collects data for EENet training, i.e. images with corresponding left and right finger tip locations in 2D of the left gripper. Each image is stored as an indexed .png file with a corresponding .npy file of the finger 2D positions in the given image (col first like cv2). For example: 00000.png, 00000.npy, 00001.png, 00001.npy etc.

#### 3.2.1 Setup

Setting up the ROS environment
```
1. cd ~/ros_ws && ./baxter.sh

# Now open a bunch of terminal windows, or use terminator - will make your life easier. 
# Ideally, run each of the next commands in a separate window

2. roslaunch realsense2_camera rs_aligned_depth.launch

3. rosrun rqt_reconfigure rqt_reconfigure 
(in GUI, go to the camera menu, turn auto-balance off and turn the value above to the very left, which is 280. These options are at the very bottom of the camera parameters. If you do not adjust these settings, the images will be very reddish.)

4. roslaunch hand_eye_calibration hand_eye_tf_broadcaster.launch

5. rosrun baxter_examples send_urdf_fragment.py -f `rospack find baxter_description`/urdf/left_end_effector_real.urdf.xacro -l left_hand -j left_gripper_base

6. rosrun baxter_examples gripper_cuff_control.py

7. rosrun image_view image_view image:=/camera2/color/image_raw
(visualizing the camera image for debugging)

```

#### 3.2.2 Data Collection Advice

Before proceeding, keep the following things in mind:

1. Use the visualizor from step 7 from section 3.2.1. to make sure you're taking good data.

2. Do NOT move the end effector tips out of the view (and do not move too close to the frame boundaries because it will trigger nasty out of bounds errors during the dataloading)

3. Use the cuff control of the gripper to move the gripper around for more data diversity

4. If you run the script multiple times with the same arguments, it will keep the current data in the folder and will save the new data on top of it.

5. Create two folders/datasets, one for training and one for validation 

6. Change up the environment (essentially other objects seen from the camera view) during data collection so the detector does not overfit on it

7. Make sure your validation consists of environments not trained on

8. You can also block the view of the finger endpoints - they do not have to visible in the image, but they need to be physically within the camera frame (refer to 2.)

If you're done, you can start training!

#### 3.2.3 Function Signature of Data Collection Script

script: collect_ee_data.sh
Arguments: Dataset Name, Sequence Name (file prefix), number of images being taken (keyboard interruption possible)

Note that the script can be interrupted via CTRL + c, so number of pictures taken can be set arbitrarily high and then interrupted.

saves images and npy files into /media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/imitation_learning/{dataset_name}/videos/train/{sequence_name}_view0/


```
. collect_ee_data.sh folder_name sequence_name num_pictures 
# example
. collect_ee_data.sh hand_raw name 20000
```

This will save into the folder /media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/imitation_learning/hand_raw/name_view0/



### 3.3 Evaluate model on dataset
```
python3 scripts/train_eenet.py --root-dir /media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/imitation_learning/endeffector_5/videos/train/0_view0 -sf --model-path /home/zhouxian/git/eenet/trained_models/2019-02-13-09-37-15/eenet-epoch-10.pk --eval
```
