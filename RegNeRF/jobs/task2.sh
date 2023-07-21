#CUDA_VISIBLE_DEVICES=5 python train.py --gin_configs configs/mipnerf6/dtu/scan8_3.gin
#CUDA_VISIBLE_DEVICES=5 python train.py --gin_configs configs/mipnerf6/dtu/scan8_3.gin#

#CUDA_VISIBLE_DEVICES=5 python train.py --gin_configs configs/mipnerf6/llff/horns3.gin
#CUDA_VISIBLE_DEVICES=5 python train.py --gin_configs configs/mipnerf6/llff/horns3.gin
#CUDA_VISIBLE_DEVICES=5 python train.py --gin_configs configs/mipnerf6/llff/fortress3.gin
#CUDA_VISIBLE_DEVICES=5 python train.py --gin_configs configs/mipnerf6/llff/fortress3.gin#

#CUDA_VISIBLE_DEVICES=5 python train.py --gin_configs configs/mipnerf3/dtu/scan114_3.gin
#CUDA_VISIBLE_DEVICES=5 python train.py --gin_configs configs/mipnerf3/dtu/scan114_3.gin

#CUDA_VISIBLE_DEVICES=5 python train.py --gin_configs configs/regnerf3/llff/fern3.gin
#CUDA_VISIBLE_DEVICES=5 python train.py --gin_configs configs/regnerf3/llff/flower3.gin
#CUDA_VISIBLE_DEVICES=5 python train.py --gin_configs configs/regnerf3/llff/fortress3.gin
#CUDA_VISIBLE_DEVICES=5 python eval.py --gin_configs configs/regnerf3/llff/fern3.gin
#CUDA_VISIBLE_DEVICES=5 python eval.py --gin_configs configs/regnerf3/llff/flower3.gin
#CUDA_VISIBLE_DEVICES=5 python eval.py --gin_configs configs/regnerf3/llff/fortress3.gin
#CUDA_VISIBLE_DEVICES=5 python train.py --gin_configs configs/regnerf3/nerf/hotdog3.gin
#CUDA_VISIBLE_DEVICES=4 python train.py --gin_configs configs/regnerf3/nerf/chair3.gin
#CUDA_VISIBLE_DEVICES=5 python eval.py --gin_configs configs/regnerf3/dtu/scan21_3.gin

#CUDA_VISIBLE_DEVICES=6 python eval.py --gin_configs configs/mipnerf3/llff/room3.gin
CUDA_VISIBLE_DEVICES=5 python eval.py --gin_configs configs/mipnerf3/dtu/scan114_3.gin
CUDA_VISIBLE_DEVICES=5 python eval.py --gin_configs configs/mipnerf3/llff/room3.gin