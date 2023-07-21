import imageio
import numpy as np
import os
import cv2

#使用改脚本你需要更改三个变量
#save_dir 你运行完eval.py后生成的rgb与深度图片存放位置
#frame_num 该位置下rgb图片的个数，即视频帧数
#更改改行代码中对应的分辨率 ：out = cv2.VideoWriter(f'{save_dir}/rgb_video.mp4',fourcc, 10.0, (1600, 800))
#其中，NeRF数据集为(1600, 800)，LLFF为(2016,756)，DTU为(1280, 512)

cap = cv2.VideoCapture(0)
save_dir = 'vanila_nerf_scan114/renderonly_path_049999'
frame_num = 60
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'{save_dir}/rgb_video.mp4',fourcc, 10.0, (1280, 512))

#(1600, 800)
#(2016,756)
#(1280, 512)
rgb_frames = [i for i in range(frame_num)]
dpt_frames = [i for i in range(frame_num)]

for img_path in os.listdir(save_dir):
    if 'color_' in img_path:
        print(img_path)
        img_vis = imageio.imread(os.path.join(save_dir, img_path))
        idx = int(img_path[-7:-4])
        rgb_frames[idx] = img_vis.astype('uint8')
    if 'depth_' in img_path:
        img_vis = imageio.imread(os.path.join(save_dir, img_path))
        idx = int(img_path[-7:-4])
        dpt_frames[idx] = img_vis.astype('uint8')

print(f'deeeeeeeeeeeeeeeeeeebug{len(rgb_frames)}')

for i in range(len(rgb_frames)):
    
    rgb_frames[i] = np.hstack([rgb_frames[i], dpt_frames[i]])
    rgb2bgr = np.array([2,1,0])
    out.write(rgb_frames[i][:,:,rgb2bgr])
#imageio.mimwrite(f'{save_dir}/rgb_video.mov', np.stack(rgb_frames), 'mov', fps=10, quality=10)
