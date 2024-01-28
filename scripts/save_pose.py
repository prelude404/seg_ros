#!/usr/bin/env python
import mmcv
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from mmpose.registry import VISUALIZERS
import time
import numpy as np
import json
import pickle

print('指定模型的配置文件和checkpoint文件路径')
config_file = '/home/joy/Documents/configs/top_down/hrnet/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
checkpoint_file = '/home/joy/Documents/checkpoints/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'

print('根据配置文件和checkpoint文件构建模型')
# device = 'cpu'
device = 'cuda:0'
model = init_model(config_file, checkpoint_file, device=device, cfg_options={'model': {'test_cfg': {'output_heatmaps': True}}})

time_start = time.time()

print('测试多张图片并展示结果')

for i in range(1,131):
    img_path = '/home/joy/Documents/24-0122/mask/data/color2/color_'+str(i)+'.png'
    # img_path = '/home/joy/mm_ws/src/seg_ros/demo/yyj1.jpeg'
    img = mmcv.imread(img_path)
    result = inference_topdown(model, img)
    data_samples = merge_data_samples(result)
    
    data = {
        "keypoints": data_samples.pred_instances.keypoints[0,:,:].tolist(),
        "scores": data_samples.pred_instances.keypoint_scores[0,:].tolist()
    }

    # with open('/home/joy/mm_ws/src/seg_ros/images/pose/json/pose_'+str(i)+'.json', 'w') as json_file:
    #     json.dump(data, json_file, indent=2)

    # with open('/home/joy/mm_ws/src/seg_ros/images/pose/keypoints/keypoints_'+str(i)+'.pkl', 'wb') as file1:
    #     pickle.dump(data_samples.pred_instances.keypoints[0,:,:].tolist(), file1)

    # with open('/home/joy/mm_ws/src/seg_ros/images/pose/scores/scores_'+str(i)+'.pkl', 'wb') as file2:
    #     pickle.dump(data_samples.pred_instances.keypoint_scores[0,:].tolist(), file2)

    # 半径
    model.cfg.visualizer.radius = 5
    # 线宽
    model.cfg.visualizer.line_width = 4
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # 元数据
    visualizer.set_dataset_meta(model.dataset_meta)

    output_path = '/home/joy/Documents/24-0122/mask/data/pose2/pose_'+str(i)+'.png'
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    img_output = visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=True,
            draw_bbox=True,
            show_kpt_idx=True,
            show=False,
            wait_time=0,
            out_file=output_path
    )


time_end = time.time()
print('[Pose Estimation Time] = {:.4f} , [FPS] = {:.2f}'.format((time_end - time_start), 1.0/(time_end - time_start)))