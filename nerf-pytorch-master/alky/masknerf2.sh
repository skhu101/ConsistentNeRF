CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/materials.txt --expname a_materials3_mask --hardmask --with_depth_loss
CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/mic.txt --expname a_mic3_mask --hardmask --with_depth_loss

#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/materials.txt --expname a_materials3_mask --hardmask --with_depth_loss --i_testset=1
#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/mic.txt --expname a_mic3_mask --hardmask --with_depth_loss --i_testset=1
#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/chair.txt --expname a_chair3_mask --hardmask --with_depth_loss --i_testset=1
#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/ficus.txt --expname a_ficus3_mask --hardmask --with_depth_loss --i_testset=1

#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/materials.txt --expname a_materials3 --i_testset=1
#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/mic.txt --expname a_mic3 --i_testset=1
#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/chair.txt --expname a_chair3 --i_testset=1
#CUDA_VISIBLE_DEVICES=6 python run_nerf_view.py --config configs_3view/ficus.txt --expname a_ficus3 --i_testset=1