import os, sys, re
import ipdb

import numpy as np
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import cv2

from internal.mask_utils.load_llff import load_llff_data
from internal.mask_utils.load_blender import load_blender_data, load_blender_view_data
from internal.mask_utils.load_dtu import load_dtu_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(0)

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def get_rays_ref(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def get_ref_rays(w2c_ref, c2w_ref, intrinsic_ref, point_samples, img, depths_h=None):
    '''
        point_samples [N_rays N_sample 3]
    '''

    device = img.device

    B, N_rays, N_samples = point_samples.shape[:3] # [B ray_samples N_samples 3]
    point_samples = point_samples.reshape(B, -1, 3)

    N, C, H, W = img.shape
    inv_scale = torch.tensor([W-1, H-1]).to(device)

    # wrap to ref view
    if w2c_ref is not None:
        # R = w2c_ref[:3, :3]  # (3, 3)
        # T = w2c_ref[:3, 3:]  # (3, 1)
        # point_samples = torch.matmul(point_samples, R.t()) + T.reshape(1,3)
        R = w2c_ref[:, :3, :3]  # (B, 3, 3)
        T = w2c_ref[:, :3, 3:]  # (B, 3, 1)
        transform = torch.FloatTensor([[1,0,0],[0,-1,0], [0,0,-1]])[None]
        point_samples = (point_samples @ R.permute(0,2,1) + T.reshape(B, 1, 3)) @ transform # [B, ray_samples*N_samples, 3] [5, 131072, 3]

    if intrinsic_ref is not None:
        # using projection
        # point_samples_pixel = point_samples @ intrinsic_ref.t()
        point_samples_pixel = point_samples @ intrinsic_ref.permute(0,2,1) # [B, ray_samples*N_samples, 3]

        point_samples_pixel_x = (point_samples_pixel[:, :, 0] / point_samples_pixel[:, :, 2] + 0.0).round().detach()  # [B, ray_samples*N_samples]
        point_samples_pixel_y = (point_samples_pixel[:, :, 1] / point_samples_pixel[:, :, 2] + 0.0).round().detach()  # [B, ray_samples*N_samples]

        point_samples_pixel_x_norm = point_samples_pixel_x / inv_scale[0] # [B, ray_samples*N_samples]
        point_samples_pixel_y_norm = point_samples_pixel_y / inv_scale[1] # [B, ray_samples*N_samples]

    # mask
    mask_x = ((point_samples_pixel_x_norm > 0.0) * (point_samples_pixel_x_norm < 1.0)) # [B, N_rays]
    mask_y = ((point_samples_pixel_y_norm > 0.0) * (point_samples_pixel_y_norm < 1.0))  # [B, N_rays]
    mask = mask_x * mask_y

    cent = [intrinsic_ref[0, 0, 2], intrinsic_ref[0, 1, 2]]
    focal = [intrinsic_ref[0, 0, 0], intrinsic_ref[0, 1, 1]]

    directions = torch.stack([(point_samples_pixel_x[mask] - cent[0]) / focal[0], (point_samples_pixel_y[mask] - cent[1]) / focal[1], torch.ones_like(point_samples_pixel_x[mask])], -1)  # (H, W, 3)

    rays_o, rays_d = get_rays_ref(directions, c2w_ref[0])  # both (h*w, 3)

    rgb_ref = img[:, :, point_samples_pixel_y[mask].type(torch.LongTensor), point_samples_pixel_x[mask].type(torch.LongTensor)]
    if depths_h is not None:
        depth_h_ref = depths_h.unsqueeze(1)[:, :, point_samples_pixel_y[mask].type(torch.LongTensor),point_samples_pixel_x[mask].type(torch.LongTensor)]
        return rgb_ref, depth_h_ref, point_samples, rays_o, rays_d, mask
    else:
        return rgb_ref, point_samples, rays_o, rays_d, mask

def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_hard_masks(dataset_name, path, scene, nb_views = 3):
    
    K = None
    
    if dataset_name == 'llff':
        
        llff_path = path
        factor = 4
        spherify = False
        no_ndc = True
        
        datadir = os.path.join(llff_path, scene)
        images, poses, bds, render_poses, i_test = load_llff_data(datadir, factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=spherify)

        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, datadir)

        scene = os.path.basename(datadir)
        depth_files = [os.path.join('./data/nerf_llff_data_depth/', scene, f) for f in sorted(os.listdir(os.path.join('./data/nerf_llff_data_depth/', scene))) \
                if f.endswith('pfm')]

        depths_cas = [cv2.resize(np.array(read_pfm(filename)[0], dtype=np.float32), (1008, 756)) for filename in depth_files]
        depths_cas = np.stack(depths_cas[:images.shape[0]], axis=0)
        
        name = os.path.basename(datadir)
        i_train = torch.load('configs/pairs.th')[f'{name}_train'][:nb_views]
        i_test = torch.load('configs/pairs.th')[f'{name}_val']

        print('DEFINING BOUNDS')
        if no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif dataset_name == 'blender':
        
        nerf_path = path
        white_bkgd = True
        
        datadir = os.path.join(nerf_path, scene)
        images, poses, render_poses, hwf, i_split, depths_cas = load_blender_view_data(datadir, False, 8, train_view_num=nb_views)
        print('Loaded blender', images.shape, depths_cas.shape, render_poses.shape, hwf, datadir)
        name = os.path.basename(datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif dataset_name == 'dtu':
        
        dtu_path = path
        
        datadir = os.path.join(dtu_path, scene)
        images, poses, bds, render_poses, hwf, depths_cas, depths = load_dtu_data(datadir, train_view_num = nb_views)

        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, datadir)

        i_train = torch.load('configs/pairs.th')[f'dtu_train'][:nb_views]
        i_test = torch.load('configs/pairs.th')[f'dtu_val']
  

        print('DEFINING BOUNDS')

        near = np.ndarray.min(bds)
        far = np.ndarray.max(bds)

        print('NEAR FAR', near, far)

    else:
        print('Unknown dataset type', dataset_name, 'exiting')
        return
    
#====================================================================================================================================
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    # Create log dir and copy the config file
    basedir = 'other'
    expname = f'masks/{dataset_name}/{scene}'
    
    # generate mask information
    mask_all = []
    scene = os.path.basename(datadir)
    
    os.makedirs(f'{basedir}/{expname}/', exist_ok=True)
    for tgt_index in range(images.shape[0]):
        if tgt_index in i_train:
            
            #目标视角深度与ray
            pose_tgt = poses[tgt_index, :3, :4]
            rays_o_tgt, rays_d_tgt = get_rays(H, W, K, torch.Tensor(pose_tgt))  # (H, W, 3), (H, W, 3)
            depth_cas_tgt = torch.Tensor(depths_cas[tgt_index])
            rays_o_tgt, rays_d_tgt = rays_o_tgt.reshape(-1, 3), rays_d_tgt.reshape(-1, 3)
            depth_cas_tgt = depth_cas_tgt.reshape(-1)
            mask_tgt = torch.zeros_like(depth_cas_tgt)

            #遍历其他视角作ref
            for ref_index in i_train:
                if ref_index != tgt_index:
                    mask_tgt_mid = torch.zeros_like(depth_cas_tgt)
                    c2w_ref = torch.eye(4)
                    c2w_ref[:3, :4] = torch.Tensor(poses[ref_index, :3, :4])
                    w2c_ref = torch.inverse(c2w_ref)
                    for chunk_idx in range(depth_cas_tgt.shape[0] // 5120 + int(depth_cas_tgt.shape[0] % 5120 > 0)):
                        
                        #从目标视角中取出像素点对应三维点坐标
                        point_samples_w = rays_o_tgt[chunk_idx * 5120:(chunk_idx + 1) * 5120] + \
                                        depth_cas_tgt[chunk_idx * 5120:(chunk_idx + 1) * 5120,None] * rays_d_tgt[chunk_idx * 5120:(chunk_idx + 1) * 5120]  # [N_rays, 3]

                        #计算这些点在ref视角下颜色与深度
                        rgb_target_ref, rays_depth_ref, point_samples_c_ref, rays_o_ref, rays_d_ref, mask_bound = get_ref_rays(
                            w2c_ref.unsqueeze(0), c2w_ref.unsqueeze(0), torch.Tensor(K).unsqueeze(0), point_samples_w[None, :, None, :],
                            torch.Tensor(images[ref_index]).unsqueeze(0).permute(0, 3, 1, 2),
                            torch.Tensor(depths_cas[ref_index]).unsqueeze(0))
                        
                        #计算mask_tgt_mid，意义为tgt视角中与当前ref视角匹配点的像素mask
                        if mask_bound.sum() != 0:
                            mask = torch.ones(rgb_target_ref.shape[2], 1).cuda() < 0
                            occlusion_threshold = 0.1
                            while mask.sum() == 0:
                                depth_diff = point_samples_c_ref[mask_bound][...,-1].unsqueeze(-1)-rays_depth_ref.squeeze(0).squeeze(0)[:, None]
                                mask = (abs(depth_diff)<occlusion_threshold)
                                occlusion_threshold = 2 * occlusion_threshold

                            mask_bound_new = mask_bound.clone()
                            mask_bound_new[mask_bound] = mask.squeeze()
                            mask_tgt_mid[chunk_idx * 5120:(chunk_idx + 1) * 5120] = mask_bound_new.squeeze()
                            del mask_bound_new
                        else:
                            mask_tgt_mid[chunk_idx * 5120:(chunk_idx + 1) * 5120] = False

            
                    #注意true + true = true, false + false = false, true + false = true
                    mask_tgt += mask_tgt_mid
            mask_all.append(np.array((mask_tgt>0).cpu().reshape(H, W)))
            #保存mask
            imageio.imwrite(f'{basedir}/{expname}/{tgt_index}_mask_{nb_views}view.jpg', (mask_all[-1].reshape(H, W) * 255).astype(np.uint8))
        else:
            if dataset_name != 'blender':
                mask_all.append(np.zeros((hwf[0],hwf[1])))
        

            
    masks_cas = np.stack(mask_all, 0)
    
    return masks_cas

