
## Installation

We recommend to use [Anaconda](https://www.anaconda.com/products/individual) to set up the environment: 

```conda env create -f pytorch1.8_skhu.yaml```

Next, activate the environment:

```conda activate pytorch1.8_skhu```

## Data

在当前目录下（readme这一级目录下）建立路径
```mkdir data```
将备份的regnerf中的同名data文件夹复制过来。data文件夹下包含以下数据：

1. DTU,LLFF,NeRF 原本数据集
2. 基于DTU,LLFF,NeRF数据集与pair.npy中定义的前三个train view计算得到的MiDaS深度信息，文件名有midas前缀
3. 使用mvsnerf模型在DTU,LLFF,NeRF数据集上训练后预测的深度信息，文件名有nerf前缀

## Running the code

### Training an new model

1.训练Vanila NeRF:

```CUDA_VISIBLE_DEVICES=0 python run_nerf_view.py --config configs_3view/hotdog.txt --expname a_hotdog3_mask```

2.训练ConsistentNeRF(ours),即加入hardmask与MiDaS,分别代表multi-view consistency 与 single-view consistency:

```CUDA_VISIBLE_DEVICES=4 python run_nerf_view.py --config configs_3view/hotdog.txt --expname a_hotdog3_mask --hardmask --with_depth_loss```

注：当前我们的方法只适用于NeRF Synthetic数据集，在其他数据集上跑会出错。

### Rendering and Evaluating model

在前一步将模型训练好后可以使用以下命令评估,渲染所有test view图像并计算相应metric：

```CUDA_VISIBLE_DEVICES=4 python run_nerf_view.py --config configs_3view/hotdog.txt --expname vanila_nerf_hotdog --i_testset=1```

注：--expname后接训练时存放的模型路径, --expname xxx 对应 ./logs/xxx/

如果需要渲染路径生成演示视频,则使用：

```CUDA_VISIBLE_DEVICES=4 python run_nerf_view.py --config configs_3view/hotdog.txt --expname vanila_nerf_hotdog --render_only```

完成后会得到生成视频所需要的所有图片（你模型路径下的path_renders文件夹），然后请根据video_generation.py中的注释更改该脚本，然后运行该脚本生成视频。

