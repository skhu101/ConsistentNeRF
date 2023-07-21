#CUDA_VISIBLE_DEVICES=6 python train.py --gin_configs configs/mipnerf6/dtu/scan21_3.gin
#CUDA_VISIBLE_DEVICES=6 python train.py --gin_configs configs/mipnerf6/dtu/scan21_3.gin
#
#CUDA_VISIBLE_DEVICES=6 python train.py --gin_configs configs/mipnerf6/llff/leaves3.gin
#CUDA_VISIBLE_DEVICES=6 python train.py --gin_configs configs/mipnerf6/llff/leaves3.gin
#CUDA_VISIBLE_DEVICES=6 python train.py --gin_configs configs/mipnerf6/llff/orchids3.gin
#CUDA_VISIBLE_DEVICES=6 python train.py --gin_configs configs/mipnerf6/llff/orchids3.gin

#CUDA_VISIBLE_DEVICES=6 python train.py --gin_configs configs/regnerf3/llff/leaves3.gin
#CUDA_VISIBLE_DEVICES=6 python train.py --gin_configs configs/regnerf3/llff/orchids3.gin
#CUDA_VISIBLE_DEVICES=6 python train.py --gin_configs configs/regnerf3/llff/room3.gin
#CUDA_VISIBLE_DEVICES=6 python eval.py --gin_configs configs/regnerf3/llff/leaves3.gin
#CUDA_VISIBLE_DEVICES=6 python eval.py --gin_configs configs/regnerf3/llff/orchids3.gin
#CUDA_VISIBLE_DEVICES=6 python eval.py --gin_configs configs/regnerf3/llff/room3.gin

CUDA_VISIBLE_DEVICES=4 python eval.py --gin_configs configs/regnerf3/dtu/scan114_3_temp.gin
CUDA_VISIBLE_DEVICES=4 python eval.py --gin_configs configs/mipnerf3/dtu/scan114_3.gin
CUDA_VISIBLE_DEVICES=4 python eval.py --gin_configs configs/regnerf3/dtu/scan114_3.gin
#CUDA_VISIBLE_DEVICES=4 python eval.py --gin_configs configs/regnerf3/llff/room3_temp.gin