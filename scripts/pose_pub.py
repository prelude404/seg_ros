#!/usr/bin/env python
import rospy
import threading

import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import torch
import numpy as np

import mmcv
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import time

from seg_ros.msg import KeypointsWithScores

pic1_color = Image()

class Pose():
    def __init__(self):
        self.config_file = '/home/joy/mm_ws/src/seg_ros/configs/top_down/hrnet/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
        self.checkpoint_file = '/home/joy/mm_ws/src/seg_ros/checkpoints/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
        self.device = 'cpu'
        self.model = init_model(self.config_file, self.checkpoint_file, device=self.device, cfg_options={'model': {'test_cfg': {'output_heatmaps': False}}})
        self.time = None
        self.img = None
        self.keypoints = None
        self.keypoint_scores = None
        self.message = KeypointsWithScores()
    
    def infer(self,img):
        self.img = img
        result = inference_topdown(self.model, self.img)
        data_samples = merge_data_samples(result)
        self.keypoints = data_samples.pred_instances.keypoints
        self.keypoint_scores = data_samples.pred_instances.keypoint_scores

    def getMessage(self):
        self.message.keypoints = [Point(x=k[0], y=k[1], z=0) for k in self.keypoints[0,:,:]]
        self.message.keypoint_scores = [Float32(data=s) for s in self.keypoint_scores[0,:]]
        self.message.header.stamp = self.time
        # print('keypoints: ',self.message.keypoints)
        # print('keypoint_scores: ',self.message.keypoint_scores)

def spin_job():
    rospy.spin()

def color_cb1(color_msg):
    global pic1_color
    pic1_color = color_msg

if __name__ == '__main__':
    rospy.init_node('pose_pub')
    rate = rospy.Rate(30)
    
    spin_thread = threading.Thread(target = spin_job)
    spin_thread.start()

    rospy.loginfo("Listening to Color Image Messages...")

    # rospy.Subscriber("/cam_1/color/image_raw", Image, color_cb1, queue_size=5)
    rospy.Subscriber("/camera/color/image_raw", Image, color_cb1, queue_size=5)

    pub_pose = rospy.Publisher("cam_1/pose", KeypointsWithScores, queue_size=1)

    pose1 = Pose()

    pic1_color_cur = -1
    
    bridge1 = CvBridge()

    while not rospy.is_shutdown():
        
        if not pic1_color.header.stamp.to_sec() > 0 or pic1_color.header.stamp.to_sec() == pic1_color_cur:
            continue
        
        pic1_color_cur = pic1_color.header.stamp.to_sec()

        pose1.time = pic1_color.header.stamp

        # 保留color_image的时间戳
        img1 = bridge1.imgmsg_to_cv2(pic1_color, pic1_color.encoding)

        if img1 is None:
            continue
        else:
            rospy.loginfo("Received the Cam_1 Color Image.")

        pose1.infer(img1)
        pose1.getMessage()
        pub_pose.publish(pose1.message)

        rate.sleep()
