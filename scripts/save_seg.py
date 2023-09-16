#!/usr/bin/env python
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmengine.visualization.utils import check_type, tensor2ndarray
from mmdet.registry import VISUALIZERS
import torch
import numpy as np
import cv2
import time

print('指定模型的配置文件和checkpoint文件路径')
config_file = '/home/joy/mm_ws/src/seg_ros/configs/scnet/scnet_r50_fpn_1x_coco.py'
checkpoint_file = '/home/joy/mm_ws/src/seg_ros/checkpoints/scnet/scnet_r50_fpn_1x_coco-c3f09857.pth'
print('根据配置文件和checkpoint文件构建模型')
device = 'cpu'
# device = 'cuda:0'
model = init_detector(config_file, checkpoint_file, device=device)
print('测试单张图片并展示结果')

time_start = time.time()

for i in range(1, 21):
    img = mmcv.imread('/home/joy/mm_ws/src/seg_ros/images/color/color_'+str(i)+'.png')
    result = inference_detector(model, img)
    instances = result.pred_instances

    print('raw_num: ', len(instances))
    if 'scores' in instances:
        instances = instances[instances.scores > 0.4]

    print('high_confidence_num: ', len(instances))

    if 'labels' in instances:
        instances = instances[instances.labels == 0]

    print('human_num: ', len(instances))

    print(instances)

    if 'masks' in instances:
        masks = instances.masks
        check_type('binary_masks', masks, (np.ndarray, torch.Tensor))
        binary_masks = tensor2ndarray(masks)
        binary_masks = binary_masks.astype('uint8') * 255
        print('number of dim:', binary_masks.ndim)
        print('shape:', binary_masks.shape)
        print('size:', binary_masks.size)

        cv2.imwrite('/home/joy/mm_ws/src/seg_ros/images/seg/seg_'+str(i)+'.png', binary_masks[0])
        # mmcv.imshow(binary_masks[0], 'human_mask')

time_end = time.time()
print('[Segment Time] = {:.4f} , [FPS] = {:.2f}'.format((time_end - time_start), 1.0/(time_end - time_start)))


