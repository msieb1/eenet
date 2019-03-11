from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import multiprocessing
from multiprocessing import Process
import imageio
import os
import subprocess
import sys
import time
import cv2
import string
import matplotlib
matplotlib.use('TkAgg')

from matplotlib import animation  # pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt
import numpy as np
from six.moves import input
import wave
from scipy import misc
import pyrealsense2 as rs
import io
import rospy
import tf
from geometry_msgs.msg import PointStamped

sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-2]))
from subscribers import img_subscriber, depth_subscriber


# D435 intrinsics matrix
INTRIN = np.array([615.3900146484375, 0.0, 326.35467529296875, 
		0.0, 615.323974609375, 240.33250427246094, 
		0.0, 0.0, 1.0]).reshape(3, 3)

class EEFingerListener(object):
    def __init__(self):
        self.listener = tf.TransformListener()

    def get_3d_pose(self, gripper='l', finger='r', frame=None):
        assert(gripper == 'r' or gripper == 'l')
        assert(finger == 'l' or finger == 'r')
        # camera frame is usually /camera{camID}_color_optical_frame (camID=2 in ourcase)
        self.listener.waitForTransform("/base", "/{}_gripper_{}_finger_tip".format(gripper, finger), rospy.Time(0), rospy.Duration(4.0))
        (trans,rot) = self.listener.lookupTransform("/base", "/{}_gripper_{}_finger_tip".format(gripper, finger), rospy.Time(0))
        p3d=PointStamped()
        p3d.header.frame_id = "base"
        p3d.header.stamp =rospy.Time(0)
        p3d.point.x=trans[0]
        p3d.point.y=trans[1]
        p3d.point.z=trans[2]
        if frame is not None:
            self.listener.waitForTransform("/base", frame, rospy.Time(0),rospy.Duration(4.0))
            p3d_transformed = self.listener.transformPoint(frame, p3d)
        p3d_transformed = np.array([p3d_transformed.point.x, p3d_transformed.point.y, p3d_transformed.point.z])
        return p3d_transformed   


def deproject_pixel_to_point(p2d, depth_z, intrin):
	p3d = np.zeros(3,)
	p3d[0] = (p2d[0] - intrin[0, 2]) / intrin[0, 0]
	p3d[1] = (p2d[1] - intrin[1, 2]) / intrin[1, 1]
	p3d[2] = 1.0
	p3d *= depth_z
	return p3d

finger_listener = EEFingerListener()
   
view_idx = 2
p3d_l = finger_listener.get_3d_pose(gripper='l', finger='l', frame="/camera{}_color_optical_frame".format(device_ids[view_idx]))
p3d_r = finger_listener.get_3d_pose(gripper='l', finger='r', frame="/camera{}_color_optical_frame".format(device_ids[view_idx]))


p3d_deprojected = deproject_pixel_to_point(p2d, depth_z, intrin)

import pdb; pdb.set_trace()