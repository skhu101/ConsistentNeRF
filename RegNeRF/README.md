
## Installation

We recommend to use [Anaconda](https://www.anaconda.com/products/individual) to set up the environment. First, create a new `regnerf` environment: 

```conda create -n regnerf python=3.7```

Next, activate the environment:

```conda activate regnerf```

You can then install the dependencies:

```pip install -r requirements.txt```

```pip install lpips```

```pip install ipdb```

Finally, install jaxlib with the appropriate CUDA version (tested with jaxlib 0.1.68 and CUDA 11.0):

```pip install --upgrade jaxlib==0.1.68+cuda110 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html```

Note: 
1. If you run into cuda error, try:

```conda install cudatoolkit=11.0```

2. If you meet errors about jax.random, try:

```pip3 install chex==0.1.2```

3. If you have errors about protobuf, try:

```pip install protobuf==3.20.1```

4. 找到备份的regnerf，将configs/pairs.npy拷贝到当前目录相同路径下，或者将mvsnerf中的pairs.th文件存成npy格式，保存到该路径下。

## Data

在当前目录下（readme这一级目录下）建立路径
```mkdir data```
将备份的regnerf中的同名data文件夹复制过来。data文件夹下包含以下数据：

1. DTU,LLFF,NeRF 原本数据集
2. 基于DTU,LLFF,NeRF数据集与pair.npy中定义的前三个train view计算得到的MiDaS深度信息，文件名有midas前缀
3. 使用mvsnerf模型在DTU,LLFF,NeRF数据集上训练后预测的深度信息，文件名有nerf前缀

## Running the code

### Training an new model

1.训练MipNeRF:

```CUDA_VISIBLE_DEVICES=0 python train.py --gin_configs configs/mipnerf3/dtu/scan21_3.gin```

2.训练RegNeRF:

```CUDA_VISIBLE_DEVICES=0 python train.py --gin_configs configs/regnerf3/llff/leaves3.gin```

3.加入hardmask(multiview consistency) 和/或 midas(singleview consistency):

打开internal/config.py文件，其中:

```compute_mono_depth_metrics: bool = True``` 该参数为True则使用MiDaS，无论是训练MipNeRF还是RegNeRF

```use_hardmask: bool = True``` 该参数为True则对于LLFF数据集和NeRF数据集使用mvsnerf的depth计算hardmask，反之则不计算hardmask

```use_nerf_depth: bool = True``` 该参数为True则对于DTU数据集使用mvsnerf的depth计算hardmask，反之则使用gt计算hardmask

```compute_depth_metrics: bool = False```该参数为True则计算深度loss，同时如果通过前两个参数加了hardmask则计算hardmask对应像素的深度loss，如果没有hardmask则计算全部的深度loss。

另外可参考jobs文件夹下 task1.sh - task4.sh

### Rendering and Evaluating model

在前一步将模型训练好后可以使用以下命令评估与渲染：

```CUDA_VISIBLE_DEVICES=0 python eval.py --gin_configs configs/regnerf3/llff/leaves3.gin```

以上命令将自动渲染所有test view，并且计算相应metric. 如需计算某个数据集所有场景的平均metric，则参见calculate_metrics脚本。 

如果需要渲染路径生成演示视频，则需要在internal/config.py文件更改以下参数为True：

```render_path: bool = True```当前因为刚做了生成视频，所以现在是True的状态。

### Generate Videos
在上一步如果你将render_path置为True并运行eval.py，你会得到生成视频所需要的所有图片（你模型路径下的path_renders文件夹），然后请根据video_generation.py中的注释更改该脚本，然后运行该脚本生成视频。
