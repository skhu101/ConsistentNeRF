#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/materials.txt --expname a_materials3
#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/mic.txt --expname a_mic3
#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/chair.txt --expname a_chair3
#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/ficus.txt --expname a_ficus3


#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/drums.txt --expname a_vanila_nerf/blender_chair_3view_rgb_update_50000_seed_0 --i_testset=1
#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/drums.txt --expname a_vanila_nerf/blender_drums_3view_rgb_update_50000_seed_0 --i_testset=1
#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/drums.txt --expname a_vanila_nerf/blender_ficus_3view_rgb_update_50000_seed_0 --i_testset=1
#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/drums.txt --expname a_vanila_nerf/blender_hotdog_3view_rgb_update_50000_seed_0 --i_testset=1

CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/chair.txt --expname a_chair3_mask --hardmask --with_depth_loss
CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/ficus.txt --expname a_ficus3_mask --hardmask --with_depth_loss
