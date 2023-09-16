#!/usr/bin/env python
import rospy
import threading

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import numpy as np

import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmengine.visualization.utils import check_type, tensor2ndarray
from seg_ros.msg import Masks

pic1_color = Image()
pic1_depth = Image()
pic2_color = Image()
pic2_depth = Image()

class Segment():
    def __init__(self):
        self.config_file = '/home/joy/mm_ws/src/seg_ros/configs/scnet/scnet_r50_fpn_1x_coco.py'
        self.checkpoint_file = '/home/joy/mm_ws/src/seg_ros/checkpoints/scnet/scnet_r50_fpn_1x_coco-c3f09857.pth'
        self.result_folder_this_img = '/home/joy/mm_ws/src/seg_ros/demo'
        self.device = 'cpu'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.img = None
        self.result = None
        self.binary_masks = None
        self.visualizer = VISUALIZERS.build(self.model.cfg.visualizer)
        self.visualizer.dataset_meta = self.model.dataset_meta
        self.confidence_threshold = 0.4
        self.target_index = 0

    def infer(self, img):
        self.img = img
        self.result = inference_detector(self.model, self.img)

    def show(self):
        self.visualizer.add_datasample(
            'result',
            self.img,
            data_sample=self.result,
            draw_gt=False,
            wait_time=0,
            # out_file='/home/joy/mm_ws/src/seg_ros/demo/demo_result.jpg'
            # optionally, write to output file
        )
        self.visualizer.show()

    def masks(self):
        instance = self.result.pred_instances
        
        if 'scores' in instance:
            instance = instance[instance.scores > self.confidence_threshold]
        
        if 'labels' in instance:
            instance = instance[instance.labels == self.target_index]
        
        rospy.loginfo("Detected %d humans",len(instance))
        
        if 'masks' in instance:
            masks = instance.masks
            check_type('binary_masks', masks, (np.ndarray, torch.Tensor))
            self.binary_masks = tensor2ndarray(masks)
            assert self.binary_masks.dtype == np.bool_, ('The dtype of binary_masks should be np.bool_, but got {self.binary_masks.dtype}')
            self.binary_masks = self.binary_masks.astype('uint8') * 255
            # print('number of dim:', self.binary_masks.ndim)
            # print('shape:', self.binary_masks.shape)
            # print('size:', self.binary_masks.size)
            # mmcv.imshow(self.binary_masks[0], 'mask0')
            # mmcv.imshow(self.binary_masks[1], 'mask1')

def color_cb1(color_msg):
    global pic1_color
    pic1_color = color_msg

def color_cb2(color_msg):
    global pic2_color
    pic2_color = color_msg

# def depth_cb1(depth_msg):
#     global pic1_depth
#     pic1_depth = depth_msg

# def depth_cb2(depth_msg):
#     global pic2_depth
#     pic2_depth = depth_msg

# def color_cb2(color_msg):
#     global pic2_color, bridge
#     try:
#         pic2_color = bridge.imgmsg_to_cv2(color_msg,color_msg.encoding)
#         # cv2.waitKey(1050)
#     except Exception as e:
#         rospy.logerr("Could not convert color image: %s", str(e))
# 
# def depth_cb2(depth_msg):
#     global pic2_depth, bridge
#     try:
#         pic2_depth = bridge.imgmsg_to_cv2(depth_msg,depth_msg.encoding)
#         # cv2.waitKey(1050)
#     except Exception as e:
#         rospy.logerr("Could not convert depth image: %s", str(e))

def thread_job():
    rospy.spin()


if __name__ == '__main__':
    rospy.init_node('seg_ros')
    rate = rospy.Rate(5)
    add_thread = threading.Thread(target = thread_job)
    add_thread.start()
    
    rospy.loginfo("Listening to Color and Depth Image Messages...")

    rospy.Subscriber("/cam_1/color/image_raw", Image, color_cb1, queue_size=5)
    rospy.Subscriber("/cam_2/color/image_raw", Image, color_cb2, queue_size=5)

    # rospy.Subscriber("/cam_1/aligned_depth_to_color/image_raw", Image, depth_cb1, queue_size=5)
    # rospy.Subscriber("/cam_2/aligned_depth_to_color/image_raw", Image, depth_cb2, queue_size=5)

    # pub_mask1 = rospy.Publisher("/cam_1/human_mask", Image, queue_size=5)
    # pub_mask2 = rospy.Publisher("/cam_2/human_mask", Image, queue_size=5)

    pub_mask1 = rospy.Publisher("/cam_1/human_mask", Masks, queue_size=5)
    pub_mask2 = rospy.Publisher("/cam_2/human_mask", Masks, queue_size=5)

    # pub_mask1 = rospy.Publisher("/cam_1/human_mask", Image, queue_size=5)
    # pub_mask2 = rospy.Publisher("/cam_2/human_mask", Image, queue_size=5)

    seg1 = Segment()
    seg2 = Segment()

    pic1_color_cur = -1
    pic2_color_cur = -1
    pic1_depth_cur = -1
    pic2_depth_cur = -1

    bridge1 = CvBridge()
    bridge2 = CvBridge()

    # mask_msg1 = Image()
    # mask_msg2 = Image()

    while not rospy.is_shutdown():
        if not pic1_color.header.stamp.to_sec() > 0 or not pic2_color.header.stamp.to_sec() > 0:
            # print('Received not valid Image message, continue')
            continue
        # compare the image CompressedImage.seq to avoid process the same image
        if pic1_color.header.stamp.to_sec() == pic1_color_cur or pic2_color.header.stamp.to_sec() == pic2_color_cur:
            # print('Received the same image message, continue')
            continue

        # if not pic1_depth.header.stamp.to_sec() > 0 or not pic2_depth.header.stamp.to_sec() > 0:
        #     # print('Received not valid Image message, continue')
        #     continue
        # # compare the image CompressedImage.seq to avoid process the same image
        # if pic1_depth.header.stamp.to_sec() == pic1_depth_cur or pic2_depth.header.stamp.to_sec() == pic2_depth_cur:
        #     # print('Received the same image message, continue')
        #     continue

        pic1_color_cur = pic1_color.header.stamp.to_sec()
        pic2_color_cur = pic2_color.header.stamp.to_sec()
        # pic1_depth_cur = pic1_depth.header.stamp.to_sec()
        # pic2_depth_cur = pic2_depth.header.stamp.to_sec()
        rospy.loginfo("Received the images.")

        img1 = bridge1.imgmsg_to_cv2(pic1_color, pic1_color.encoding)
        img2 = bridge2.imgmsg_to_cv2(pic2_color, pic2_color.encoding)

        if img1 is None or img2 is None:
            continue
        else:
            print("SUCCESSFULLY SUB THE IMG")

        seg1.infer(img1)
        seg2.infer(img2)

        seg1.masks()
        seg2.masks()

        # msg_masks1 = Image()
        # msg_masks1 = bridge1.cv2_to_imgmsg(seg1.binary_masks[0], "bgr8")

        # msg_masks2 = Image()
        # msg_masks2 = bridge2.cv2_to_imgmsg(seg2.binary_masks[0], "bgr8")

        msg_masks1 = Masks()
        for i in range(len(seg1.binary_masks)):
            mask = seg1.binary_masks[i]
            msg_mask = bridge1.cv2_to_imgmsg(mask, "bgr8")
            msg_masks1.masks.append(msg_mask)
        
        msg_masks2 = Masks()
        for i in range(len(seg2.binary_masks)):
            mask = seg2.binary_masks[i]
            msg_mask = bridge2.cv2_to_imgmsg(mask, "bgr8")
            msg_masks2.masks.append(msg_mask)

        pub_mask1.publish(msg_masks1)
        pub_mask2.publish(msg_masks2)

        # pub_mask1.publish(bridge1.cv2_to_imgmsg(seg1.binary_masks[0],"bgr8"))
        # pub_mask2.publish(bridge2.cv2_to_imgmsg(seg2.binary_masks[0],"bgr8"))

        rate.sleep()