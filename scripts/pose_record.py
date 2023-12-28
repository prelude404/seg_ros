#!/usr/bin/env python

import rospy
import numpy as np
import threading

from seg_ros.msg import KeypointsWithScores

class pose_reader():
    def __init__(self):
        self.time1 = 0.0
        self.time2 = 0.0

        self.pose01 = np.zeros((1,17))
        self.pose02 = np.zeros((1,17))

        self.num = 17

reader = pose_reader()

def pose_cam1_cb(pose_msg):
    if len(pose_msg.keypoint_scores) > 0:
        reader.time1 = pose_msg.header.stamp.secs + pose_msg.header.stamp.nsecs/1000000000.0
        for i in range(17):
            reader.pose01[0,i] = np.expand_dims(np.array(pose_msg.keypoint_scores[i].data), axis=0)
        # reader.pose01 = np.array(pose_msg.keypoint_scores[0].data)
        # print("Type of keypoint_scores:", type(pose_msg.keypoint_scores))
        # print("Type of keypoint_scores.data:", type(pose_msg.keypoint_scores.data))

def pose_cam2_cb(pose_msg):
    if len(pose_msg.keypoint_scores) > 0:
        reader.time2 = pose_msg.header.stamp.secs + pose_msg.header.stamp.nsecs/1000000000.0
        for i in range(17):
            reader.pose02[0,i] = np.expand_dims(np.array(pose_msg.keypoint_scores[i].data), axis=0)


def spin_job():
    rospy.set_param('~spin_rate', 100)
    rospy.spin()


if __name__=='__main__':
    file_pose = open("/home/joy/mm_ws/src/seg_ros/demo/bag2.txt","r+")
    file_pose.truncate()

    rospy.init_node('pose_record')

    rate = rospy.Rate(100)

    spin_thread = threading.Thread(target = spin_job)
    spin_thread.start()

    rospy.Subscriber("cam_1/pose", KeypointsWithScores, pose_cam1_cb)
    rospy.Subscriber("cam_2/pose", KeypointsWithScores, pose_cam2_cb)

    pose1_cur = -1
    pose2_cur = -1
    
    while not rospy.is_shutdown():
        
        if not reader.time1 > 0.0 or reader.time1 == pose1_cur:
            continue

        if not reader.time2 > 0.0 or reader.time2 == pose2_cur:
            continue

        rospy.loginfo("Received Pose Message.")

        pose1_cur = reader.time1
        pose2_cur = reader.time2

        file_handle = open("/home/joy/mm_ws/src/seg_ros/demo/bag2.txt","a+")

        file_handle.write(str(reader.time1))
        file_handle.write('    ')

        # array_shape = reader.pose01.shape
        # print("Shape of the array:", array_shape)
        # print("Number of rows:", array_shape[0])
        # print("Number of columns:", array_shape[1])

        for i in range(17):
            # print(score)
            file_handle.write(str(reader.pose01[0,i]))
            file_handle.write('    ')

        for i in range(17):
            # print(score)
            file_handle.write(str(reader.pose02[0,i]))
            file_handle.write('    ')

        file_handle.write('\n')
        file_handle.close()

        rate.sleep()