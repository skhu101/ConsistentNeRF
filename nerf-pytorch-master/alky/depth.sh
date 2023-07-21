#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/hotdog.txt --expname a_hotdog3
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/ship.txt --expname a_ship3
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/lego.txt --expname a_lego3
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/drums.txt --expname a_drums3

#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/drums.txt --expname a_vanila_nerf/blender_lego_3view_rgb_update_50000_seed_0 --i_testset=1
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/drums.txt --expname a_vanila_nerf/blender_materials_3view_rgb_update_50000_seed_0 --i_testset=1
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/drums.txt --expname a_vanila_nerf/blender_mic_3view_rgb_update_50000_seed_0 --i_testset=1
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/drums.txt --expname a_vanila_nerf/blender_ship_3view_rgb_update_50000_seed_0 --i_testset=1

CUDA_VISIBLE_DEVICES=4 python run_nerf_view.py --config configs_3view/materials.txt --expname dsnerf_materials --render_only
#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_lky/room.txt --expname dsnerf_room --render_only --no_ndc 
#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/scan114.txt --expname dsnerf_scan114 --render_only

#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/scan114.txt --expname vanila_nerf_scan114 --render_only
#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_lky/room.txt --expname vanila_nerf_room --render_only --no_ndc
CUDA_VISIBLE_DEVICES=4 python run_nerf_view.py --config configs_3view/materials.txt --expname vanila_nerf_materials --render_only
CUDA_VISIBLE_DEVICES=4 python run_nerf_view.py --config configs_3view/materials.txt --expname ours_blender/a_materials3_mask --render_only