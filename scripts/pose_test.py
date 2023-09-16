#!/usr/bin/env python
import mmcv
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from mmpose.registry import VISUALIZERS
import time

print('指定模型的配置文件和checkpoint文件路径')
config_file = '/home/joy/mm_ws/src/seg_ros/configs/top_down/hrnet/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
checkpoint_file = '/home/joy/mm_ws/src/seg_ros/checkpoints/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'

print('根据配置文件和checkpoint文件构建模型')
device = 'cpu'
# device = 'cuda:0'
model = init_model(config_file, checkpoint_file, device=device, cfg_options={'model': {'test_cfg': {'output_heatmaps': True}}})

print('测试单张图片并展示结果')
img_path = '/home/joy/mm_ws/src/seg_ros/demo/yyj1.jpeg'
img = mmcv.imread(img_path)

time_start = time.time()
result = inference_topdown(model, img)
print('显示姿态识别结果')
print(result)
print('结束显示姿态识别结果')
time_end = time.time()

print('[Pose Estimation Time] = {:.4f} , [FPS] = {:.2f}'.format((time_end - time_start), 1.0/(time_end - time_start)))

data_samples = merge_data_samples(result)

print('Shape of Key Points is: ', data_samples.pred_instances.keypoints.shape)
print('Coordinate of Key Points is: ', data_samples.pred_instances.keypoints[0,:,:])

# 半径
model.cfg.visualizer.radius = 10
# 线宽
model.cfg.visualizer.line_width = 8
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# 元数据
visualizer.set_dataset_meta(model.dataset_meta)

output_path = '/home/joy/mm_ws/src/seg_ros/demo/yyj1_pose.jpeg'
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
            out_file='',
            # out_file=output_path
)
visualizer.show()