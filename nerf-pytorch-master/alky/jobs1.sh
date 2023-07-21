#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/hotdog.txt --expname a_hotdog3
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/ship.txt --expname a_ship3
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/lego.txt --expname a_lego3
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/drums.txt --expname a_drums3

#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/drums.txt --expname a_vanila_nerf/blender_lego_3view_rgb_update_50000_seed_0 --i_testset=1
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/drums.txt --expname a_vanila_nerf/blender_materials_3view_rgb_update_50000_seed_0 --i_testset=1
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/drums.txt --expname a_vanila_nerf/blender_mic_3view_rgb_update_50000_seed_0 --i_testset=1
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/drums.txt --expname a_vanila_nerf/blender_ship_3view_rgb_update_50000_seed_0 --i_testset=1
CUDA_VISIBLE_DEVICES=5 python run_nerf_view.py --config configs_3view/lego.txt --expname a_lego3_mask --hardmask --with_depth_loss
CUDA_VISIBLE_DEVICES=5 python run_nerf_view.py --config configs_3view/drums.txt --expname a_drums3_mask --hardmask --with_depth_loss


