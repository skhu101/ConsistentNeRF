import os, re
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import ipdb


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,60+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split

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

def load_blender_view_data(basedir, half_res=False, testskip=1, train_view_num=3):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    scene = os.path.basename(basedir)
    depth_files = [os.path.join('nerf_synthesic_data_depth/', scene, f) for f in sorted(os.listdir(os.path.join('nerf_synthesic_data_depth/', scene))) \
            if f.endswith('pfm')]

    if half_res:
        depths_cas_lst = [cv2.resize(np.array(read_pfm(filename)[0], dtype=np.float32), (400, 400)) for filename in depth_files]
    else:
        depths_cas_lst = [np.array(read_pfm(filename)[0], dtype=np.float32) for filename in depth_files]
    # depths_cas = np.stack(depths_cas, axis=0)
    # depths = depths_cas

    all_imgs = []
    all_poses = []
    all_depths_cas = []
    mono_dpts = []
    counts = [0]
    for s in splits:
        # meta = metas[s]
        meta = metas['train']
        imgs = []
        poses = []
        depths_cas = []
        
        # if s == 'train' or testskip == 0:
        #     skip = 1
        # else:
        #     skip = testskip

        if s == 'train':
            name = os.path.basename(basedir)
            img_idx = torch.load('configs/pairs.th')[f'{name}_{s}'][:train_view_num]
        else:
            img_idx = torch.load('configs/pairs.th')[f'{name}_val']

        print(s, ': ', img_idx)

        # for frame in meta['frames'][::skip]:
        for idx in img_idx:
            frame = meta['frames'][idx]
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            image = imageio.imread(fname)
            imgs.append(image)
            poses.append(np.array(frame['transform_matrix']))
            depths_cas.append(depths_cas_lst[idx])
            #load midas depth
            depthfile = f'./data/midas_nerf_depth/output_nerf_{scene}/{os.path.basename(fname)[:-4]}-dpt_beit_large_512.pfm'
            if os.path.isfile(depthfile):
                depth = read_pfm(depthfile)[0]
                dpt = np.where(depth < 0, 0, depth)
                
            else:
                dpt = np.zeros(image.shape[:2])
            mono_dpts.append(dpt)
            
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_depths_cas.append(depths_cas)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    depths_cas = np.concatenate(all_depths_cas, 0)
    mono_dpts = np.stack(mono_dpts)
    #ipdb.set_trace()
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-185, -95, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split, depths_cas,mono_dpts

