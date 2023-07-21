import os
import numpy as np

#该脚本计算一个数据集所有场景的平均指标
# dataset_dir 为你对数据集的所有场景训练后的所有模型（每个场景对应一个模型）所存储的目录的上级目录
# dataset 为当前计算的数据集和viewsetting 命名为数据集名+view数量，如： 'llff' + '3' = 'llff3'
# 如果是计算mipnerf模型的结果则 记得将 'bl' not in basedir 条件改为 'bl' in basedir

dataset_dir = './ours'
dataset = 'llff3'


metrics = {'psnr':0, 'ssim':0, 'lpips':0}
num = 1e-6

for basedir in os.listdir(dataset_dir):
    if dataset in basedir and 'bl' not in basedir:
        
        metric_path = os.path.join(dataset_dir, basedir, 'test_preds')
        if not os.path.isdir(metric_path):
            continue
        num += 1
        for file in os.listdir(metric_path):
            
            k_to_find = list(metrics.keys())
            for k in k_to_find:
                if k in file and 'mask' not in file:
                        
                    f = open(os.path.join(metric_path, file))
                    line = f.readline()
                    avr = np.array([float(v) for v in line.split()]).mean()
                    
                    if np.isnan(avr):
                        metrics[k] += 0
                    else:
                        metrics[k] += avr
                    k_to_find.remove(k)
                                        

for k in metrics.keys():
    metrics[k] = metrics[k]/ num
    print(f'{k} is {metrics[k]}')

print(f'{int(num)} valid scenes')
