
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

4. Download backup data at (link required)，extract it and copy 'configs/pairs.npy' to current directory under same path，or save [pairs.th](https://github.com/apchenstu/mvsnerf/tree/main/configs) to .npy format and use it instead.

## Data

Create data path:
```mkdir data```
And then copy the 'data' folder from our backup data here, which contains：

1. original dataset of DTU,LLFF and NeRF 
2. MiDaS Depth of the first three views for each scene in DTU,LLFF and NeRF, as defined in 'pairs.npy', file names begin with 'midas'.
3. The predicted depth of DTU,LLFF and NeRF from a pretrained MVSNeRF model,file names begin with 'nerf'.

## Running the code

### Training an new model

1.Train MipNeRF:

```CUDA_VISIBLE_DEVICES=0 python train.py --gin_configs configs/mipnerf3/dtu/scan21_3.gin```

2.Train RegNeRF:

```CUDA_VISIBLE_DEVICES=0 python train.py --gin_configs configs/regnerf3/llff/leaves3.gin```

3.Add hardmask (for multi-view consistency) and/or MiDaS(for single-view consistency):

Please check the following parameters within internal/config.py:

```compute_mono_depth_metrics: bool = True``` If True, apply single-view consistency loss，for either MipNeRF or RegNeRF

```use_hardmask: bool = True``` If True, use MvSNeRF depth to calculate hardmask on NeRF dataset and LLFF dataset，or else disable hardmask.

```use_nerf_depth: bool = True``` If True, use MvSNeRF depth to calculate hardmask on DTU dataset，or else use Ground Truth to calculate hardmask.

```compute_depth_metrics: bool = False```If True, calculate depth loss on the whole image, and if hardmask is enabled, only calculate depth loss on pixels within the hardmask.
For more reference, please check jobs/task1.sh - task4.sh

### Rendering and Evaluating model

After you have trained the model successfully, use the following commands to evaluate and render test results for each scene:

```CUDA_VISIBLE_DEVICES=0 python eval.py --gin_configs configs/regnerf3/llff/leaves3.gin```

If you want to calculate the metric on average of all scenes，please refer to 'calculate_metrics.py'.

### Generate Videos
If you need to render a video, please change the following parameter within 'internal/config.py'：

```render_path: bool = True```
Set 'render_path' to True and run 'eval.py' again, you will get all the frames under 'path_renders/', then please refer to 'video_generation.py' for final video generation.
