
## Installation

We recommend to use [Anaconda](https://www.anaconda.com/products/individual) to set up the environment: 

```conda env create -f environment.yml```

Next, activate the environment:

```conda activate nerf```

## Data

Create data path:
```mkdir data```
And then copy the 'data' folder from our backup data here(link required), which contains：

1. original dataset of DTU,LLFF and NeRF 
2. MiDaS Depth of the first three views for each scene in DTU,LLFF and NeRF, as defined in 'pairs.npy', file names begin with 'midas'.
3. The predicted depth of DTU,LLFF and NeRF from a pretrained MVSNeRF model,file names begin with 'nerf'.

## Running the code

### Training an new model

1.Train Vanila NeRF:

```CUDA_VISIBLE_DEVICES=0 python run_nerf_view.py --config configs_3view/hotdog.txt --expname a_hotdog3_mask```

2.Train ConsistentNeRF(ours),which adds both hardmask and MiDaS depth for supervision, accounting for multi-view consistency and single-view consistency respectively:

```CUDA_VISIBLE_DEVICES=4 python run_nerf_view.py --config configs_3view/hotdog.txt --expname a_hotdog3_mask --hardmask --with_depth_loss```

Note: Currently our method only runs on NeRF Synthetic dataset, code for LLFF and DTU dataset will be updated soon.

### Rendering and Evaluating model

After you have trained the model successfully, use the following commands to evaluate and render test results for each scene:：

```CUDA_VISIBLE_DEVICES=4 python run_nerf_view.py --config configs_3view/hotdog.txt --expname vanila_nerf_hotdog --i_testset=1```

### Generate Videos
If you need to render a video, run: 

```CUDA_VISIBLE_DEVICES=4 python run_nerf_view.py --config configs_3view/hotdog.txt --expname vanila_nerf_hotdog --render_only```

And you will get all the frames under 'path_renders/', then please refer to 'video_generation.py' for final video generation.

