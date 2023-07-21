CUDA_VISIBLE_DEVICES=4 python run_nerf_view.py --config configs_3view/hotdog.txt --expname a_hotdog3_mask --hardmask --with_depth_loss
CUDA_VISIBLE_DEVICES=4 python run_nerf_view.py --config configs_3view/ship.txt --expname a_ship3_mask --hardmask --with_depth_loss

#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/hotdog.txt --expname a_hotdog3_mask --hardmask --with_depth_loss --i_testset=1
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/ship.txt --expname a_ship3_mask --hardmask --with_depth_loss --i_testset=1
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/lego.txt --expname a_lego3_mask --hardmask --with_depth_loss --i_testset=1
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/drums.txt --expname a_drums3_mask --hardmask --with_depth_loss --i_testset=1

#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/hotdog.txt --expname a_hotdog3 --i_testset=1
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/ship.txt --expname a_ship3 --i_testset=1
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/lego.txt --expname a_lego3 --i_testset=1
#CUDA_VISIBLE_DEVICES=7 python run_nerf_view.py --config configs_3view/drums.txt --expname a_drums3 --i_testset=1
