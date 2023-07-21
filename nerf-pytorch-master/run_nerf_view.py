import os, sys, re
import pdb

import ipdb
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import cv2

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from pytorch_msssim import ssim, ms_ssim
from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data, load_blender_view_data
from load_LINEMOD import load_LINEMOD_data
from load_dtu import load_dtu_data

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

from alky.vis_utils import *
import lpips
lpips_fn = lpips.LPIPS(net="vgg").to(torch.device('cuda', torch.cuda.current_device()))

# img2mse_softmask = lambda x, y, coef : torch.mean((x - y).abs() ** coef)
# img2mse_softmask = lambda x, y: torch.mean(torch.exp((x - y) ** 2) * (x - y).abs() ** 1.8)
# img2mse_softmask = lambda x, y: torch.sum( (torch.exp((x - y) ** 2) - 1) * (x - y) ** 2 ) / torch.sum( (torch.exp((x - y) ** 2) - 1) )
# img2mse_softmask = lambda x, y, temp: torch.sum( (torch.exp((x - y) ** 2 / temp)) * (x - y) ** 2 ) / torch.sum( torch.exp((x - y) ** 2 / temp) ).detach() #+ torch.mean((x - y) ** 2)

img2mse_depth = lambda x, y, depth_scale: torch.mean((x/depth_scale - y/depth_scale) ** 2)


img2mse_softmask = lambda x, y, temp: torch.sum( (torch.exp((x - y) ** 2 / temp)) * (x - y) ** 2 ) / torch.sum( torch.exp( (x - y).detach() ** 2 / temp) ) #+ torch.mean((x - y) ** 2)


# img2mse_softmask = lambda x, y, temp: (1-temp) * torch.sum( (torch.exp((x - y) ** 2)) * (x - y) ** 2 ) / torch.sum( torch.exp((x - y) ** 2 ) ).detach() + temp * torch.mean((x - y) ** 2)

img2mse_depth_softmask = lambda x, y, temp: torch.sum( (torch.exp((x - y) ** 2 / temp)) * (x - y) ** 2 ) / torch.sum( torch.exp( (x - y).detach() ** 2 / temp) )

# img2mse_softLpmask = lambda x, y, coef : (torch.sum((x - y).abs() ** coef)) ** (2/coef) / x.shape[0]
img2mse_softLpmask = lambda x, y, coef : torch.sum( ((x - y).abs() ** coef + 1)  * (x - y) ** 2 )  / torch.sum( (x - y).abs() ** coef + 1 ).detach()


# def img2mse_softmask(pred, target, sigma=2, reduce=True, normalizer=1.0):
#     beta = 0.05 #1. / (sigma ** 2)
#     diff = torch.abs(pred-target)
#     cond = diff < beta
#     # loss = torch.where(cond, 0.5 * diff ** 2/ beta, diff - 0.5 * beta)
#     loss = torch.where(cond, diff ** 2, torch.exp(diff ** 2) * diff ** 1.8)
#     if reduce:
#         return torch.mean(loss) / normalizer
#     return torch.mean(loss, dim = 1) / normalizer

# def softmask_loss(pred, target, sigma=2, reduce=True, normalizer=1.0):
#     beta = 1. / (sigma ** 2)
#     diff = torch.abs(pred-target)
#     cond = diff < beta
#     loss = torch.where(cond, 0.5 * diff ** 2/ beta, diff - 0.5 * beta)
#     if reduce:
#         return torch.mean(loss) / normalizer
#     return torch.mean(loss, dim = 1) / normalizer

class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.33, last_epoch=-1):
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        # self.total_epochs = 150
        self.curr_temp = (1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        if self.curr_temp < self.temp_min:
            self.curr_temp = self.temp_min
        return self.curr_temp


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

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    # inputs: [N_rays, N_samples, 3]; viewdirs: [N_rays, 3]
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) # [N_rays*N_samples, 3]
    embedded = embed_fn(inputs_flat) # [N_rays*N_samples, args.multires*2*3+3]

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape) # [N_rays, N_samples, 3]
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]]) # [N_rays*N_samples, 3]
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1) # [N_rays*N_samples, args.multires_views*2*3+3]

    outputs_flat = batchify(fn, netchunk)(embedded) # [N_rays*N_samples, 4]
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]) # [N_rays, N_samples, 4]
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1) # [N_rand, 11]


    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    accs = []
    
    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        accs.append(acc.cpu().numpy())
        
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, 'color_{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    accs = np.stack(accs, 0)
    
    return rgbs, disps, accs


def  create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, coarse=True, stable_init=args.stable_init).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, stable_init=args.stable_init).to(device)
        grad_vars += list(model_fine.parameters())


    model.load_state_dict(model_fine.state_dict())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']

        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        ckpt['network_fn_state_dict']['temp_rgb'] = torch.Tensor([0.1])
        ckpt['network_fn_state_dict']['temp_depth'] = torch.Tensor([0.1])
        ckpt['network_fn_state_dict']['depth_scale'] = torch.Tensor([0.1])
        ckpt['network_fine_state_dict']['temp_rgb'] = torch.Tensor([0.1])
        ckpt['network_fine_state_dict']['temp_depth'] = torch.Tensor([0.1])
        ckpt['network_fine_state_dict']['depth_scale'] = torch.Tensor([0.1])
        #ipdb.set_trace()
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        
    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # raw2alpha = lambda raw, dists, act_fn=F.sigmoid: 1. - torch.exp(-act_fn(raw) * dists)
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
    # raw2alpha = lambda raw, dists, act_fn=F.softplus: 1. - torch.exp( -act_fn(raw, threshold=0.1) * dists )

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)

    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn) # [N_rays, N_samples, 4]
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, depth_map_0 = rgb_map, disp_map, acc_map, depth_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['depth0'] = depth_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

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
        transform = torch.FloatTensor([[1,0,0],[0,-1,0], [0,0,-1]])[None].cuda()
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

    rgb_ref = img[:, :, point_samples_pixel_y[mask].type(torch.cuda.LongTensor), point_samples_pixel_x[mask].type(torch.cuda.LongTensor)]
    if depths_h is not None:
        depth_h_ref = depths_h.unsqueeze(1)[:, :, point_samples_pixel_y[mask].type(torch.cuda.LongTensor),point_samples_pixel_x[mask].type(torch.cuda.LongTensor)]
        return rgb_ref, depth_h_ref, point_samples, rays_o, rays_d, mask
    else:
        return rgb_ref, point_samples, rays_o, rays_d, mask


def get_test_label(w2c_ref, c2w_ref, intrinsic_ref, point_samples, img):
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
        transform = torch.FloatTensor([[1,0,0],[0,-1,0], [0,0,-1]])[None].cuda()
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

    return point_samples_pixel_y, point_samples_pixel_x, mask, point_samples[:,:,-1]


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')
    parser.add_argument("--seed", type=int, default=0,
                        help='random seed')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*8, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*16, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument('--stable_init', action='store_true')
    parser.add_argument("--train_view_num", type=int, default=3,
                        help='train view number')
    parser.add_argument('--hardmask', action='store_true')
    parser.add_argument("--hardmask_coef", type=float, default=0.2,
                        help='hard mask coef')
    parser.add_argument("--occlusion_threshold", type=float, default=0.1,
                        help='occlusion threshold')
    parser.add_argument('--with_depth_loss', action='store_true')
    parser.add_argument('--with_depth_norm', action='store_true')
    parser.add_argument('--softmask', action='store_true')
    parser.add_argument("--softmask_K",   type=int, default=30,
                        help='top-K in softmask')
    parser.add_argument('--softLpmask', action='store_true')
    parser.add_argument("--Lp_coef", type=float, default=2,
                        help='Lp norm coef')
    parser.add_argument("--total_iters", type=int, default=50001,
                        help='number of iterations to train the model')
    parser.add_argument("--temp_start", type=float, default=1.0, help='starting temp')
    parser.add_argument("--temp_end", type=float, default=1.0, help='ending temp')
    parser.add_argument('--use_test_pseudo_label', action='store_true')
    parser.add_argument('--use_noise', action='store_true')
    parser.add_argument('--use_canny_edge_detection', action='store_true')
    parser.add_argument('--use_sobel_edge_detection', action='store_true')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels / dtu')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test, mono_dpts = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)

        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        scene = os.path.basename(args.datadir)
        depth_files = [os.path.join('nerf_llff_data_depth/', scene, f) for f in sorted(os.listdir(os.path.join('nerf_llff_data_depth/', scene))) \
                if f.endswith('pfm')]

        if args.factor == 8:
            depths_cas = [cv2.resize(np.array(read_pfm(filename)[0], dtype=np.float32), (504, 378)) for filename in depth_files]
        elif args.factor == 4:
            depths_cas = [cv2.resize(np.array(read_pfm(filename)[0], dtype=np.float32), (1008, 756)) for filename in depth_files]

        else:
            raise Exception('Invalid factor!')
        depths_cas = np.stack(depths_cas[:images.shape[0]], axis=0)
        depths = depths_cas
        #ipdb.set_trace()
        mono_dpts = torch.Tensor(mono_dpts)
        #ipdb.set_trace()
        
        # if not isinstance(i_test, list):
        #     i_test = [i_test]

        # if args.llffhold > 0:
        #     print('Auto LLFF holdout,', args.llffhold)
        #     i_test = np.arange(images.shape[0])[::args.llffhold]

        # i_val = i_test
        # i_train = np.array([i for i in np.arange(int(images.shape[0])) if
        #                 (i not in i_test and i not in i_val)])

        name = os.path.basename(args.datadir)
        i_train = torch.load('configs/pairs.th')[f'{name}_train'][:args.train_view_num]
        i_train_aug = torch.load('configs/pairs.th')[f'dtu_train'][args.train_view_num:16]
        i_test = torch.load('configs/pairs.th')[f'{name}_val']
        # i_test_test = torch.load('configs/pairs.th')[f'{name}_val']
        # i_test = [i_train[i] for i in range(i_train.shape[0])]
        # i_test.extend([i_test_test[i] for i in range(i_test_test.shape[0])])


        i_val = i_test

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        # images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        images, poses, render_poses, hwf, i_split, depths_cas, mono_dpts = load_blender_view_data(args.datadir, args.half_res, args.testskip, train_view_num=args.train_view_num)
        print('Loaded blender', images.shape, depths_cas.shape, render_poses.shape, hwf, args.datadir)
        name = os.path.basename(args.datadir)
        #torch.load('configs/pairs.th')[f'{name}_train'][:args.train_view_num]
        i_train, i_val, i_test = i_split
        depths = depths_cas
        mono_dpts = torch.Tensor(mono_dpts)
        #i_train_aug = torch.load('configs/pairs.th')[f'{}_train'][args.train_view_num:16]
        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

        
        # for i in range(images.shape[0]):
        #     imageio.imwrite(f'images_test/{i}.png', to8b(images[i]))


    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    elif args.dataset_type == 'dtu':

        images, poses, bds, render_poses, hwf, depths_cas, depths = load_dtu_data(args.datadir, train_view_num = args.train_view_num)

        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        i_train = torch.load('configs/pairs.th')[f'dtu_train'][:args.train_view_num]
        i_train_aug = torch.load('configs/pairs.th')[f'dtu_train'][args.train_view_num:16]
        # i_test = torch.load('configs/pairs.th')[f'dtu_train'][:args.train_view_num]
        i_test = torch.load('configs/pairs.th')[f'dtu_val']
        # i_train.extend(i_test)
        i_train_aug = i_test
        # i_test = [25, 32]
        # i_train = [25, 2]
        # i_test = [24, 1]
        i_val = i_test

        #midas 还没加上
        mono_dpts = np.zeros(depths.shape)
        mono_dpts = torch.Tensor(mono_dpts)
        
        print('DEFINING BOUNDS')

        near = np.ndarray.min(bds)
        far = np.ndarray.max(bds)

        print('NEAR FAR', near, far)

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

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


    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # tensorboard log file
    writer = SummaryWriter(basedir + '/' + expname + '/runs/')

    # generate mask information
    mask_all = []
    scene = os.path.basename(args.datadir)
    if not args.softmask:
        os.makedirs(f'{basedir}/{expname}/mask/{scene}/{args.train_view_num}view', exist_ok=True)
        for tgt_index in range(images.shape[0]):
            if tgt_index in i_train:
                pose_tgt = poses[tgt_index, :3, :4]
                rays_o_tgt, rays_d_tgt = get_rays(H, W, K, torch.Tensor(pose_tgt))  # (H, W, 3), (H, W, 3)
                depth_cas_tgt = torch.Tensor(depths_cas[tgt_index])
                rays_o_tgt, rays_d_tgt = rays_o_tgt.reshape(-1, 3), rays_d_tgt.reshape(-1, 3)
                depth_cas_tgt = depth_cas_tgt.reshape(-1)
                mask_tgt = torch.zeros_like(depth_cas_tgt)

                for ref_index in i_train:
                    if ref_index != tgt_index:
                        mask_tgt_mid = torch.zeros_like(depth_cas_tgt)
                        c2w_ref = torch.eye(4)
                        c2w_ref[:3, :4] = torch.Tensor(poses[ref_index, :3, :4])
                        w2c_ref = torch.inverse(c2w_ref)
                        for chunk_idx in range(depth_cas_tgt.shape[0] // 5120 + int(depth_cas_tgt.shape[0] % 5120 > 0)):
                            point_samples_w = rays_o_tgt[chunk_idx * 5120:(chunk_idx + 1) * 5120] + \
                                            depth_cas_tgt[chunk_idx * 5120:(chunk_idx + 1) * 5120,None] * rays_d_tgt[chunk_idx * 5120:(chunk_idx + 1) * 5120]  # [N_rays, 3]

                            rgb_target_ref, rays_depth_ref, point_samples_c_ref, rays_o_ref, rays_d_ref, mask_bound = get_ref_rays(
                                w2c_ref.unsqueeze(0), c2w_ref.unsqueeze(0), torch.Tensor(K).unsqueeze(0), point_samples_w[None, :, None, :],
                                torch.Tensor(images[ref_index]).unsqueeze(0).permute(0, 3, 1, 2),
                                torch.Tensor(depths_cas[ref_index]).unsqueeze(0))

                            if mask_bound.sum() != 0:
                                mask = torch.ones(rgb_target_ref.shape[2], 1).cuda() < 0
                                occlusion_threshold = args.occlusion_threshold
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

                        # cv2.imwrite(f'{basedir}/{expname}/mask/{scene}/{args.train_view_num}view/{tgt_index}_ref_{ref_index}_mask_{args.train_view_num}view.jpg',
                        #     (255 - np.array((mask_tgt_mid.reshape(H, W) * 255).cpu())).astype(np.uint8))

                        mask_tgt += mask_tgt_mid
            else:
                mask_tgt = torch.zeros( H * W )

            mask_all.append(np.array((mask_tgt>0).cpu().reshape(H, W)))
            imageio.imwrite(f'{basedir}/{expname}/mask/{scene}/{args.train_view_num}view/{tgt_index}_mask_{args.train_view_num}view.jpg', (mask_all[-1].reshape(H, W) * 255).astype(np.uint8))
    elif args.softmask:
        for tgt_index in range(images.shape[0]):
            if tgt_index in i_train:
                filename = f'Softmask/{args.dataset_type}/{scene}/iter_500/softmask_{tgt_index:04d}_{args.softmask_K}per.png'
                mask_tgt = torch.from_numpy(imageio.imread(filename).astype(np.float32) / 255.).reshape(-1)
            else:
                mask_tgt = torch.zeros( H * W )
            mask_all.append(np.array((mask_tgt>0).cpu().reshape(H, W)))
    masks_cas = np.stack(mask_all, 0)
    images_label = images.copy()

    '''
        images[i_train_aug] = 0
        count_ref = np.array(torch.zeros_like(torch.Tensor(depths_cas)).cpu())
        # generate depth and rgb information for test views
        for tgt_index in i_train:
            pose_tgt = poses[tgt_index, :3, :4]
            rays_o_tgt, rays_d_tgt = get_rays(H, W, K, torch.Tensor(pose_tgt))  # (H, W, 3), (H, W, 3)
            depth_cas_tgt = torch.Tensor(depths_cas[tgt_index])
            rays_o_tgt, rays_d_tgt = rays_o_tgt.reshape(-1, 3), rays_d_tgt.reshape(-1, 3)
            depth_cas_tgt = depth_cas_tgt.reshape(-1)
            # masks_cas[tgt_index] = True
            mask_tgt = torch.LongTensor(masks_cas[tgt_index])

            for ref_index in i_train_aug:
                mask_ref_mid = torch.zeros_like(torch.Tensor(depths_cas[ref_index]))
                depth_cas_ref_mid = torch.zeros_like(torch.Tensor(depths_cas[ref_index]))
                rgb_ref_mid = torch.zeros_like(torch.Tensor(images[ref_index]))
                count_ref_mid = torch.zeros_like(torch.Tensor(depths_cas[ref_index]))
                c2w_ref = torch.eye(4)
                c2w_ref[:3, :4] = torch.Tensor(poses[ref_index, :3, :4])
                w2c_ref = torch.inverse(c2w_ref)
                # mask_ref = torch.zeros_like(torch.Tensor(depths_cas[tgt_index]))

                for chunk_idx in range(depth_cas_tgt[mask_tgt.reshape(-1)>0].shape[0] // 5120 + int(depth_cas_tgt[mask_tgt.reshape(-1)>0].shape[0] % 5120 > 0)):
                    point_samples_w = rays_o_tgt[mask_tgt.reshape(-1)>0][chunk_idx * 5120:(chunk_idx + 1) * 5120] + \
                                    depth_cas_tgt[mask_tgt.reshape(-1)>0][chunk_idx * 5120:(chunk_idx + 1) * 5120, None] * rays_d_tgt[mask_tgt.reshape(-1)>0][chunk_idx * 5120:(chunk_idx + 1) * 5120]  # [N_rays, 3]

                    point_samples_pixel_y, point_samples_pixel_x, mask, point_samples_depth = get_test_label(
                        w2c_ref.unsqueeze(0), c2w_ref.unsqueeze(0), torch.Tensor(K).unsqueeze(0), point_samples_w[None, :, None, :],
                        torch.Tensor(images[ref_index]).unsqueeze(0).permute(0, 3, 1, 2))

                    mask_ref_mid[point_samples_pixel_y[mask].type(torch.cuda.LongTensor), point_samples_pixel_x[mask].type(torch.cuda.LongTensor)] = True
                    depth_cas_ref_mid[point_samples_pixel_y[mask].type(torch.cuda.LongTensor), point_samples_pixel_x[mask].type(torch.cuda.LongTensor)] += point_samples_depth[mask].squeeze()
                    rgb_ref_mid[point_samples_pixel_y[mask].type(torch.cuda.LongTensor), point_samples_pixel_x[mask].type(torch.cuda.LongTensor)] += torch.Tensor(images[tgt_index]).reshape(-1, 3)[mask_tgt.reshape(-1)>0][chunk_idx * 5120:(chunk_idx + 1) * 5120][mask.squeeze()]
                    count_ref_mid[point_samples_pixel_y[mask].type(torch.cuda.LongTensor), point_samples_pixel_x[mask].type(torch.cuda.LongTensor)] += 1

                depths_cas[ref_index] += np.array(depth_cas_ref_mid.cpu())
                masks_cas[ref_index] = np.array(mask_ref_mid.cpu())
                images[ref_index] += np.array(rgb_ref_mid.cpu())
                count_ref[ref_index] += np.array(count_ref_mid.cpu())

        count_ref[count_ref == 0] = 1
        depths_cas = depths_cas / count_ref
        images = images / count_ref[..., np.newaxis]
    '''
    # for ref_index in i_train_aug:

    #     imageio.imwrite(
    #         f'{basedir}/{expname}/mask/{scene}/{args.train_view_num}view/{ref_index}_mask_{args.train_view_num}view.jpg',
    #         (255 - masks_cas[ref_index] * 255).astype(np.uint8))

    #     imageio.imwrite(
    #         f'{basedir}/{expname}/mask/{scene}/{args.train_view_num}view/{ref_index}_rgb_{args.train_view_num}view.jpg',
    #         (images[ref_index] * 255).astype(np.uint8))

    #     imageio.imwrite(
    #         f'{basedir}/{expname}/mask/{scene}/{args.train_view_num}view/{ref_index}_label_{args.train_view_num}view.jpg',
    #         (images_label[ref_index] * 255).astype(np.uint8))

    if args.use_canny_edge_detection:
        # generate mask information
        mask_all = []
        scene = os.path.basename(args.datadir)
        os.makedirs(f'{basedir}/{expname}/canny_mask/{scene}/{args.train_view_num}view', exist_ok=True)
        for tgt_index in range(images.shape[0]):
            if tgt_index in i_train: 
                # Convert to graycsale
                img_gray = cv2.cvtColor(images[tgt_index], cv2.COLOR_BGR2GRAY)
                # Blur the image for better edge detection
                img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
                # Canny Edge Detection
                edges = cv2.Canny(image=(img_blur*255).astype(np.uint8), threshold1=1, threshold2=200) # Canny Edge Detection
                mask_tgt = edges
            else:
                mask_tgt = np.zeros( H * W )

            mask_all.append(np.array((mask_tgt>0).reshape(H, W)))
            imageio.imwrite(f'{basedir}/{expname}/canny_mask/{scene}/{args.train_view_num}view/{tgt_index}_mask_{args.train_view_num}view.jpg', (255-mask_all[-1].reshape(H, W) * 255).astype(np.uint8))

        masks_cas = np.stack(mask_all, 0)

    if args.use_sobel_edge_detection:
        # generate mask information
        mask_all = []
        scene = os.path.basename(args.datadir)
        os.makedirs(f'{basedir}/{expname}/sobel_mask/{scene}/{args.train_view_num}view', exist_ok=True)
        for tgt_index in range(images.shape[0]):
            if tgt_index in i_train: 
                # Convert to graycsale
                img_gray = cv2.cvtColor(images[tgt_index], cv2.COLOR_BGR2GRAY)
                # Blur the image for better edge detection
                img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
                # Sobel Edge Detection
                edges = cv2.Sobel(src=(img_blur*255).astype(np.uint8), ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                mask_tgt = edges
            else:
                mask_tgt = np.zeros( H * W )

            mask_all.append(np.array((mask_tgt!=0).reshape(H, W)))
            imageio.imwrite(f'{basedir}/{expname}/sobel_mask/{scene}/{args.train_view_num}view/{tgt_index}_mask_{args.train_view_num}view.jpg', (255-mask_all[-1].reshape(H, W) * 255).astype(np.uint8))

        masks_cas = np.stack(mask_all, 0)

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start
    # render_kwargs_train['network_fn'].depth_scale.data.fill_(far)
    # render_kwargs_train['network_fine'].depth_scale.data.fill_(far)


    # path = os.path.join(basedir, expname, '{:06d}.tar'.format(0))
    # torch.save({
    #     'global_step': global_step,
    #     'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
    #     'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }, path)
    # print('Saved checkpoints at', path)

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                # images = images[i_test]
                images = images_label[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, disps, accs = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)

            for ind in range(disps.shape[0]):
                aa=lky_visualize_depth(1/disps[ind], accs[ind])
                save_img_u8(aa, os.path.join(testsavedir, f"depth_{ind:03d}.png"))
                
            #test_loss = img2mse(torch.Tensor(rgbs), torch.Tensor(images))
            #test_psnr = mse2psnr(test_loss)
            #print('test psnr: ', test_psnr)

            print('Done rendering', testsavedir)
            #imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None], depths_cas[:,None,:,:,None].repeat(3, axis=-1), masks_cas[:,None,:,:,None].repeat(3, axis=-1)], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]

        if args.use_test_pseudo_label:
            rays_rgb_test = np.concatenate([rays_rgb[i][masks_cas[i]] for i in i_train_aug])

        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        # rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 5, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]

        if args.use_test_pseudo_label:
            rays_rgb = np.concatenate([rays_rgb, rays_rgb_test])

        rays_rgb = rays_rgb.astype(np.float32)

        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
        depths_cas = torch.Tensor(depths_cas).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = args.total_iters + 1 #200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # # automatic select
    # best_psnr_value = 0
    # for hardmask_coef in [0.2, 0.8]:
    #     print('hardmask coef ', hardmask_coef, ' in process')
    #     train_psnr_lst = []
    #     for i in range(0, 250):
    #         # Sample random ray batch
    #         if use_batching:
    #             # Random over all images
    #             batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
    #             batch = torch.transpose(batch, 0, 1)
    #             batch_rays, target_s, depth_cas_s, mask_cas_s = batch[:2], batch[2], batch[3][:,0], batch[4][:,0]
    #
    #             i_batch += N_rand
    #             if i_batch >= rays_rgb.shape[0]:
    #                 print("Shuffle data after an epoch!")
    #                 rand_idx = torch.randperm(rays_rgb.shape[0])
    #                 rays_rgb = rays_rgb[rand_idx]
    #                 i_batch = 0
    #
    #         else:
    #             # Random from one image
    #             img_i = np.random.choice(i_train)
    #
    #             target = images[img_i]
    #             target = torch.Tensor(target).to(device)
    #             pose = poses[img_i, :3,:4]
    #             depth_cas = torch.Tensor(depths_cas[img_i]).to(device)
    #             depth = torch.Tensor(depths[img_i]).to(device)
    #             mask_cas = torch.Tensor(masks_cas[img_i]).to(device)
    #
    #             if N_rand is not None:
    #                 rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
    #
    #                 # if i < args.precrop_iters:
    #                 if i < 250:
    #                     dH = int(H//2 * args.precrop_frac)
    #                     dW = int(W//2 * args.precrop_frac)
    #                     coords = torch.stack(
    #                         torch.meshgrid(
    #                             torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
    #                             torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
    #                         ), -1)
    #                     if i == start:
    #                         print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled")
    #                 else:
    #                     coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
    #
    #                 coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
    #                 select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
    #                 select_coords = coords[select_inds].long()  # (N_rand, 2)
    #                 rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    #                 rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    #                 batch_rays = torch.stack([rays_o, rays_d], 0)
    #                 target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    #                 depth_cas_s = depth_cas[select_coords[:, 0], select_coords[:, 1]]
    #                 depth_s = depth[select_coords[:, 0], select_coords[:, 1]]
    #                 mask_cas_s = mask_cas[select_coords[:, 0], select_coords[:, 1]]
    #
    #
    #         #####  Core optimization loop  #####
    #         rgb, disp, acc, depth_pred, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
    #                                                 verbose=i < 10, retraw=True,
    #                                                 **render_kwargs_train)
    #         loss = 0
    #
    #         optimizer.zero_grad()
    #         temp = 1.0
    #         if args.hardmask:
    #             img_loss = img2mse(rgb[mask_cas_s.squeeze() == 1], target_s[mask_cas_s.squeeze() == 1])
    #             if mask_cas_s.squeeze().sum() != N_rand: img_loss += hardmask_coef * img2mse(rgb[mask_cas_s.squeeze() == 0], target_s[mask_cas_s.squeeze() == 0])
    #         elif args.softmask:
    #             img_loss = img2mse_softmask(rgb, target_s, temp)
    #         elif args.softLpmask:
    #             img_loss = img2mse_softLpmask(rgb, target_s, args.Lp_coef)
    #         else:
    #             img_loss = img2mse(rgb, target_s)
    #
    #         trans = extras['raw'][...,-1]
    #         loss += img_loss
    #         psnr = mse2psnr(img_loss)
    #         if args.with_depth_loss:
    #             if args.hardmask:
    #                 depth_loss = img2mse(depth_pred[mask_cas_s.squeeze() == 1]/far, depth_cas_s[mask_cas_s.squeeze() == 1]/far)
    #                 if mask_cas_s.squeeze().sum() != N_rand: depth_loss += hardmask_coef * img2mse(depth_pred[mask_cas_s.squeeze() == 0]/far, depth_cas_s[mask_cas_s.squeeze() == 0]/far)
    #             elif args.softmask:
    #                 depth_loss = img2mse_depth_softmask(depth_pred/far, depth_cas_s/far, temp)
    #             elif args.softLpmask:
    #                 depth_loss = img2mse_softLpmask(depth_pred/far, depth_cas_s/far, args.Lp_coef)
    #             else:
    #                 depth_loss = img2mse(depth_pred, depth_cas_s)
    #             loss = loss + depth_loss
    #
    #         if 'rgb0' in extras:
    #             if args.hardmask:
    #                 img_loss0 = img2mse(extras['rgb0'][mask_cas_s.squeeze() == 1], target_s[mask_cas_s.squeeze() == 1])
    #                 if mask_cas_s.squeeze().sum() != N_rand: img_loss0 += hardmask_coef * img2mse(extras['rgb0'][mask_cas_s.squeeze() == 0], target_s[mask_cas_s.squeeze() == 0])
    #             elif args.softmask:
    #                 img_loss0 = img2mse_softmask(extras['rgb0'], target_s, temp)
    #             elif args.softLpmask:
    #                 img_loss0 = img2mse_softLpmask(extras['rgb0'], target_s, args.Lp_coef)
    #             else:
    #                 img_loss0 = img2mse(extras['rgb0'], target_s)
    #             loss = loss + img_loss0
    #
    #             psnr0 = mse2psnr(img_loss0)
    #             if args.with_depth_loss:
    #                 if args.hardmask:
    #                     depth_loss0 = img2mse(extras['depth0'][mask_cas_s.squeeze() == 1]/far, depth_cas_s[mask_cas_s.squeeze() == 1]/far)
    #                     if mask_cas_s.squeeze().sum() != N_rand: depth_loss0 += hardmask_coef * img2mse(extras['depth0'][mask_cas_s.squeeze() == 0]/far, depth_cas_s[mask_cas_s.squeeze() == 0]/far)
    #                 elif args.softmask:
    #                     depth_loss0 = img2mse_depth_softmask(extras['depth0']/far, depth_cas_s/far, temp)
    #                 elif args.softLpmask:
    #                     depth_loss0 = img2mse_softLpmask(extras['depth0']/far, depth_cas_s/far, args.Lp_coef)
    #                 else:
    #                     depth_loss0 = img2mse(extras['depth0'], depth_cas_s)
    #                 loss = loss + depth_loss0
    #
    #         loss.backward()
    #         optimizer.step()
    #         # train_psnr_lst.append((psnr+psnr0).detach().item())
    #
    #     with torch.no_grad():
    #         rgbs_train, disps_train = render_path(torch.Tensor(poses[i_train]).to(device), hwf, K, args.chunk, render_kwargs_train, gt_imgs=images[i_train])
    #     print('Saved test set')
    #
    #     train_loss = img2mse(torch.Tensor(rgbs_train), torch.Tensor(images[i_train]))
    #     train_psnr = mse2psnr(train_loss)
    #
    #     if train_psnr > best_psnr_value :
    #         best_hardmask_coef = hardmask_coef
    #         best_psnr_value = train_psnr
    #     print('best: ', best_psnr_value, best_hardmask_coef, ' currrent: ', train_psnr, hardmask_coef)
    #
    #     ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
    #              'tar' in f]
    #     print('Found ckpts', ckpts)
    #     if len(ckpts) > 0 and not args.no_reload:
    #         ckpt_path = ckpts[-1]
    #         print('Reloading from', ckpt_path)
    #         ckpt = torch.load(ckpt_path)
    #
    #         # start = ckpt['global_step']
    #         optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    #
    #         # Load model
    #         render_kwargs_train['network_fn'].load_state_dict(ckpt['network_fn_state_dict'])
    #         if render_kwargs_train['network_fine'] is not None:
    #             render_kwargs_train['network_fine'].load_state_dict(ckpt['network_fine_state_dict'])
    #
    #     # reset hardmask coef
    #     args.hardmask_coef = best_hardmask_coef


    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    temp_scheduler = Temp_Scheduler(args.total_iters, args.temp_start, args.temp_start, temp_min=args.temp_end)
    std_scheduler = Temp_Scheduler(args.total_iters, 0.2, 0.05, temp_min=0.05)


    start = start + 1
    for i in range(start, N_iters):
        time0 = time.time()
        
        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s, depth_cas_s, mask_cas_s = batch[:2], batch[2], batch[3][:,0], batch[4][:,0]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            if not args.use_test_pseudo_label:
                img_i = np.random.choice(i_train)

                target = images[img_i]
                target = torch.Tensor(target).to(device)
                pose = poses[img_i, :3,:4]
                depth_cas = torch.Tensor(depths_cas[img_i]).to(device)
                depth = torch.Tensor(depths[img_i]).to(device)
                mask_cas = torch.Tensor(masks_cas[img_i]).to(device)

                if N_rand is not None:
                    rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                    if i < args.precrop_iters:
                        dH = int(H//2 * args.precrop_frac)
                        dW = int(W//2 * args.precrop_frac)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                            ), -1)
                        if i == start:
                            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                    else:
                        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                    
                    #sample patch
                    patch_size = 16
                    n_patches = 4
                    num = 0
                    patch_idxs = []
                    while num < n_patches:
                    
                        # Sample start locations
                        if i < args.precrop_iters:
                            x0 = np.random.randint(H//2 - dH, H//2 + dH - patch_size, size=(1, 1, 1))
                            y0 = np.random.randint(H//2 - dH, W//2 + dW - patch_size, size=(1, 1, 1))
                        else:
                            x0 = np.random.randint(0, H - patch_size + 1, size=(1, 1, 1))
                            y0 = np.random.randint(0, W - patch_size + 1, size=(1, 1, 1))
                            
                        
                        xy0 = np.concatenate([x0, y0], axis=-1)
                        patch_idx = xy0 + np.stack(
                            np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='xy'),
                            axis=-1).reshape(1, -1, 2)

                        patch_idx = patch_idx.reshape(-1, 2)
                        
                        #ray_indices_patch = patch_idx[:,1] * shape[2] + patch_idx[:,0]

                        rgb_mask = target[patch_idx[:,0], patch_idx[:,1]]
                        rgb_mask = rgb_mask.mean(1)
                        if rgb_mask[rgb_mask == 1].shape[0] < 257:
                            #只要背景白色少于一半的
                            num += 1
                            patch_idxs.append(patch_idx)
                            
                    #ipdb.set_trace()
                    patch_idxs = np.concatenate(patch_idxs, 0)    
                    patch_idxs = torch.Tensor(patch_idxs).long()
                    
                    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()  # (N_rand, 2)
                    select_coords = torch.cat([patch_idxs,select_coords])
                    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o, rays_d], 0)
                    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    depth_cas_s = depth_cas[select_coords[:, 0], select_coords[:, 1]]
                    depth_s = depth[select_coords[:, 0], select_coords[:, 1]]
                    mask_cas_s = mask_cas[select_coords[:, 0], select_coords[:, 1]]
                    mono_dpt_s = mono_dpts[img_i][select_coords[:, 0], select_coords[:, 1]]

                    #ipdb.set_trace()
            else:
                rand_num = random.random()
                if rand_num < 0.9 or i < args.precrop_iters:
                    img_i = np.random.choice(i_train)
                else:
                    img_i = np.random.choice(i_train_aug)

                # if i % 2 == 0:
                #     img_i = np.random.choice(i_train)
                # else:
                #     img_i = np.random.choice(i_test)

                target = images[img_i]
                target = torch.Tensor(target).to(device)
                pose = poses[img_i, :3,:4]
                depth_cas = torch.Tensor(depths_cas[img_i]).to(device)
                depth = torch.Tensor(depths[img_i]).to(device)
                mask_cas = torch.Tensor(masks_cas[img_i]).to(device)

                if N_rand is not None:
                    rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                    if i < args.precrop_iters:
                        dH = int(H//2 * args.precrop_frac)
                        dW = int(W//2 * args.precrop_frac)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                            ), -1)
                        if i == start:
                            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                    else:
                        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                    if img_i in i_train:
                        coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                    else:
                        if i < args.precrop_iters:
                            coords = torch.reshape(coords[mask_cas[H//2 - dH:H//2 + dH, W//2 - dW : W//2 + dW] > 0], [-1, 2])  # (H * W, 2)
                        else:
                            coords = torch.reshape(coords[mask_cas>0], [-1, 2])  # (H * W, 2)

                    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()  # (N_rand, 2)
                    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o, rays_d], 0)
                    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    depth_cas_s = depth_cas[select_coords[:, 0], select_coords[:, 1]]
                    depth_s = depth[select_coords[:, 0], select_coords[:, 1]]
                    mask_cas_s = mask_cas[select_coords[:, 0], select_coords[:, 1]]

        #####  Core optimization loop  #####
        rgb, disp, acc, depth_pred, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        loss = 0

        # if not args.no_ndc:
        #     rays_o, rays_d = batch_rays
        #     t = -(1 + rays_o[..., 2]) / rays_d[..., 2]
        #     rays_o = rays_o + t[..., None] * rays_d
        #     depth_cas_s = 1. - rays_o[...,2] / (rays_o[...,2] + depth_cas_s * rays_d[..., 2])
            # depth_pred = depth_pred * rays_o[...,2] / (rays_d[..., 2] - rays_d[..., 2] * depth_pred)

        # if args.hardmask:
        #     rays_o, rays_d = batch_rays
        #     point_samples_w = rays_o + depth_cas_s[:, None] * rays_d  # [N_rays, 3]
        #
        #     ref_index = np.random.choice(i_test) #random.randint(0, 2)
        #
        #     # while ref_index == img_i:
        #     #     ref_index = np.random.choice(i_train)
        #     c2w_ref = torch.eye(4)
        #     c2w_ref[:3, :4] = poses[ref_index, :3, :4]
        #     w2c_ref = torch.inverse(c2w_ref)
        #     # rgb_target_ref, rays_depth_ref, point_samples_c_ref, rays_o_ref, rays_d_ref, mask_bound = get_ref_rays(w2c_ref.unsqueeze(0), c2w_ref.unsqueeze(0), torch.Tensor(K).unsqueeze(0), point_samples_w[None, :,None, :], torch.Tensor(images[ref_index]).unsqueeze(0).permute(0,3,1,2), torch.Tensor(depths_cas[ref_index]).unsqueeze(0))
        #     _, _, _, rays_o_ref, rays_d_ref, mask_bound = get_ref_rays(w2c_ref.unsqueeze(0), c2w_ref.unsqueeze(0), torch.Tensor(K).unsqueeze(0), point_samples_w[None, :,None, :], torch.Tensor(images[ref_index]).unsqueeze(0).permute(0,3,1,2), torch.Tensor(depths_cas[ref_index]).unsqueeze(0))
        #
        #     mask = torch.ones(rgb_target_ref.shape[2], 1).cuda() > 0
        #     # mask = torch.ones(rgb_target_ref.shape[2], 1).cuda() < 0
        #     # occlusion_threshold = args.occlusion_threshold
        #     # while mask.sum() == 0:
        #     #     depth_diff = point_samples_c_ref[mask_bound][...,-1].unsqueeze(-1)-rays_depth_ref.squeeze()[:, None]
        #     #     mask = (abs(depth_diff)<occlusion_threshold)
        #     #     occlusion_threshold = 2 * occlusion_threshold
        #
        #     point_samples_w = rays_o + depth_pred[:, None] * rays_d  # [N_rays, 3]
        #     _, _, _, point_samples_depth = get_test_label(
        #         w2c_ref.unsqueeze(0), c2w_ref.unsqueeze(0), torch.Tensor(K).unsqueeze(0),
        #         point_samples_w[None, :, None, :],
        #         torch.Tensor(images[ref_index]).unsqueeze(0).permute(0, 3, 1, 2))
        #
        #     batch_rays_ref = torch.stack([rays_o_ref[mask.squeeze()], rays_d_ref[mask.squeeze()]], 0)
        #
        #     rgb_ref, disp_ref, acc_ref, depth_pred_ref, extras_ref = render(H, W, K, chunk=args.chunk, rays=batch_rays_ref,
        #                                             verbose=i < 10, retraw=True, **render_kwargs_train)
        #
        #     loss += 0.1 + img2mse(point_samples_depth[mask_bound].squeeze().detach()/far, depth_pred_ref/far)
        #     loss += 0.1 + img2mse(point_samples_depth[mask_bound].squeeze() / far, depth_pred_ref.detach() / far)

            # img_loss_ref = img2mse(rgb_ref, rgb_target_ref.squeeze().permute(1,0)[mask.squeeze()])
            # loss += img_loss_ref
            #
            # if args.with_depth_loss:
            #     loss += img2mse(depth_pred_ref, rays_depth_ref[mask.squeeze()[None,None]])
            #
            # if 'rgb0' in extras_ref:
            #     loss += img2mse(extras_ref['rgb0'], rgb_target_ref.squeeze().permute(1,0)[mask.squeeze()])
            #     if args.with_depth_loss:
            #         loss += img2mse(extras_ref['depth0'], rays_depth_ref[mask.squeeze()[None,None]])

        if args.use_noise:
            std = std_scheduler.step()
            rgb = rgb + torch.normal(0.0, std, size=rgb.shape).to(rgb.device)
            depth_pred = depth_pred + far * torch.normal(0.0, std, size=depth_pred.shape).to(depth_pred.device)
            extras['rgb0'] = extras['rgb0'] + torch.normal(0.0, std, size=extras['rgb0'].shape).to(extras['rgb0'].device)
            extras['depth0'] = extras['depth0'] + far * torch.normal(0.0, std, size=extras['depth0'].shape).to(extras['depth0'].device)

        optimizer.zero_grad()
        # rand_num = random.randint(0, 1)
        # temp = temp_scheduler.step()
        
        #rgb mse loss fine
        if args.hardmask or args.softmask:
            # hardmask_coef = F.softplus(render_kwargs_train['network_fine'].mask_coef_rgb)
            img_loss = img2mse(rgb[mask_cas_s.squeeze() == 1], target_s[mask_cas_s.squeeze() == 1])
            if mask_cas_s.squeeze().sum() != N_rand: img_loss += args.hardmask_coef * img2mse(rgb[mask_cas_s.squeeze() == 0], target_s[mask_cas_s.squeeze() == 0])
            
            # img_loss = img2mse(rgb[mask_cas_s.squeeze() == 1], target_s[mask_cas_s.squeeze() == 1])
            # if mask_cas_s.squeeze().sum() != N_rand: img_loss += args.hardmask_coef * img2mse(rgb[mask_cas_s.squeeze() == 0], target_s[mask_cas_s.squeeze() == 0])

            # img_loss = img2mse(rgb[mask_bound.squeeze()][mask.squeeze()], target_s[mask_bound.squeeze()][mask.squeeze()])
            #
            # if mask.squeeze().sum() != mask.squeeze().shape[0]: img_loss += 0.2 * img2mse(rgb[mask_bound.squeeze()][~mask.squeeze()], target_s[mask_bound.squeeze()][~mask.squeeze()])
            # if mask_bound.squeeze().sum() != N_rand:
            #     img_loss += 0.2 * img2mse(rgb[~mask_bound.squeeze()], target_s[~mask_bound.squeeze()])
        elif args.softmask:
            temp = F.softplus(render_kwargs_train['network_fine'].temp_rgb)
            img_loss = img2mse_softmask(rgb, target_s, temp)
            # img_loss = softmask_loss(rgb, target_s)
            # img_loss = torch.mean((rgb - target_s) ** 2 / ((rgb-target_s).abs().detach()**1.5))
        elif args.softLpmask:
            img_loss = img2mse_softLpmask(rgb, target_s, args.Lp_coef)
        else:
            img_loss = img2mse(rgb, target_s)

        trans = extras['raw'][...,-1]
        if not use_batching:
            if img_i in i_train:
                loss += img_loss
            else:
                loss = loss + 0.1 * img_loss
        else:
            loss += img_loss
        psnr = mse2psnr(img_loss)
        
        #midas loss fine
        patch_num = 4
        if True:    
            depth_predict_clip = 1 / torch.where(depth_pred <= 0, 0.0001*torch.ones(1), depth_pred)
            # depth_predict = depth_pred[:config.batch_size]
            # depth_gt = batch['dpts'][:config.batch_size]
            depth_mse = 0.0
            ssim_fine = 0.0
            lpips_fine = 0.0
            
            mono_dpts = torch.Tensor(mono_dpts)
            for i_patch in range(patch_num):
                depth_predict = torch.nan_to_num(depth_predict_clip[i_patch * 16*16 : (i_patch+1) * 16*16])
                depth_gt = torch.nan_to_num(mono_dpt_s[i_patch * 16*16 : (i_patch+1) * 16*16])  
                
                mask = torch.where(depth_gt>0, torch.ones(1), torch.zeros(1))
                
                #rgb_mask = target_s[i_patch * 16*16 : (i_patch+1) * 16*16]
                #rgb_mask = rgb_mask.mean(1)
                    
                #ipdb.set_trace()
                img_pred = rgb[i_patch * 16*16 : (i_patch+1) * 16*16].reshape(1, 16, 16, 3)
                img_gt = target_s[i_patch * 16*16 : (i_patch+1) * 16*16].reshape(1, 16, 16, 3)
                ssim_fine += ssim(img_pred, img_gt, data_range = 1, size_average = False)
                
                gt, pred = img_gt, img_pred
                gt, pred = (gt-0.5)*2., (pred-0.5)*2.
                gt, pred = gt.permute([0,3,1,2]),pred.permute([0,3,1,2])
                lpips_fine += lpips_fn(pred, gt).reshape(-1)
                
                
                depth_min = torch.where(depth_gt > 0, depth_gt, torch.ones(1)*10**5).min()
                depth_max = depth_gt.max()
                depth_gt = mask * (depth_gt - depth_min) / (depth_max - depth_min + 0.0001)

                depth_min = torch.where(mask * depth_predict>0, depth_predict, torch.ones(1)*10**5).min()
                depth_max = (mask * depth_predict).max()
                depth_predict = mask * (depth_predict - depth_min) / (depth_max - depth_min + 0.0001)

                alpha = (depth_predict - depth_gt).mean()
                depth_mse += ((depth_gt - depth_predict + alpha)**2).mean() / patch_num / 2
            mono_depth_mses = depth_mse
            ssim_fine = ssim_fine/4
            lpips_fine = lpips_fine/4
        else:
            mono_depth_mses = 0.0
        
        #ipdb.set_trace()
        loss += 0.001 * mono_depth_mses
        loss -= 0.005 * ssim_fine[0]
        loss += 0.005 * lpips_fine[0]
        
        if args.with_depth_loss:
            if args.hardmask or args.softmask:
                # if args.dataset_type == 'blender':
                #     mask = depth_cas_s < near
                #     depth_pred[mask] = 0
                #     depth_cas_s[mask] = 0

                depth_loss = img2mse(depth_pred[mask_cas_s.squeeze() == 1]/far, depth_cas_s[mask_cas_s.squeeze() == 1]/far)
                # if mask_cas_s.squeeze().sum() != N_rand: depth_loss += args.hardmask_coef * img2mse(depth_pred[mask_cas_s.squeeze() == 0]/far, depth_cas_s[mask_cas_s.squeeze() == 0]/far)

                # depth_scale = render_kwargs_train['network_fine'].depth_scale
                # depth_loss = img2mse_depth(depth_pred[mask_cas_s.squeeze() == 1], depth_cas_s[mask_cas_s.squeeze() == 1], depth_scale)
                # if mask_cas_s.squeeze().sum() != N_rand: depth_loss += args.hardmask_coef * img2mse_depth(depth_pred[mask_cas_s.squeeze() == 0], depth_cas_s[mask_cas_s.squeeze() == 0], depth_scale)

                # depth_loss = img2mse(depth_pred[mask_cas_s.squeeze() == 1]/far, depth_cas_s[mask_cas_s.squeeze() == 1]/far)
                # if mask_cas_s.squeeze().sum() != N_rand: depth_loss += args.hardmask_coef * img2mse(depth_pred[mask_cas_s.squeeze() == 0]/far, depth_cas_s[mask_cas_s.squeeze() == 0]/far)

                # loss += img2mse(depth_pred[mask_bound.squeeze()][mask.squeeze()], depth_cas_s[mask_bound.squeeze()][mask.squeeze()]) #if rand_num == 1 else 0
                # if mask.squeeze().sum() != mask.squeeze().shape[0]: loss += 0.2*img2mse(depth_pred[mask_bound.squeeze()][~mask.squeeze()], depth_cas_s[mask_bound.squeeze()][~mask.squeeze()])
                # if mask_bound.squeeze().sum() != N_rand:
                #     loss += 0.2 * img2mse(depth_pred[~mask_bound.squeeze()], depth_cas_s[~mask_bound.squeeze()])
            elif args.softmask:
                # if args.dataset_type == 'blender':
                #     mask = depth_cas_s > near
                #     loss += img2mse_softmask(depth_pred[mask]/far, depth_cas_s[mask]/far, 1.1)
                # else:
                # loss += img2mse_softmask(depth_pred/far, depth_cas_s/far, 1.1)
                temp = F.softplus(render_kwargs_train['network_fine'].temp_depth)
                # depth_scale = F.softplus(render_kwargs_train['network_fine'].depth_scale)
                depth_loss = img2mse_depth_softmask(depth_pred/far, depth_cas_s/far, temp)
            elif args.softLpmask:
                depth_loss = img2mse_softLpmask(depth_pred/far, depth_cas_s/far, args.Lp_coef)
            elif args.with_depth_norm:
                if mask_cas_s.squeeze().sum() != N_rand: depth_cas_s[mask_cas_s.squeeze() == 0] = 0
                depth_loss = img2mse(depth_pred/far, depth_cas_s/far)
            else:
                # if args.dataset_type == 'blender':
                #     mask = depth_cas_s > 0
                #     loss += img2mse(depth_pred[mask], depth_cas_s[mask])
                # else:
                if mask_cas_s.squeeze().sum() != N_rand: depth_cas_s[mask_cas_s.squeeze() == 0] = 0
                depth_loss = img2mse(depth_pred, depth_cas_s)
                # depth_loss = img2mse(depth_pred[mask_cas_s.squeeze() == 1], depth_cas_s[mask_cas_s.squeeze() == 1])
                # if mask_cas_s.squeeze().sum() != N_rand: depth_loss += img2mse(depth_pred[mask_cas_s.squeeze() == 0], torch.zeros_like(depth_cas_s[mask_cas_s.squeeze() == 0]))

            if not use_batching:
                if img_i in i_train:
                    loss = loss + depth_loss
                else:
                    loss = loss + 0.1 * depth_loss
            else:
                loss = loss + depth_loss
        else:
            depth_loss = 0    

        if 'rgb0' in extras:
            if args.hardmask or args.softmask:
                img_loss0 = img2mse(extras['rgb0'][mask_cas_s.squeeze() == 1], target_s[mask_cas_s.squeeze() == 1])
                if mask_cas_s.squeeze().sum() != N_rand: img_loss0 += args.hardmask_coef * img2mse(extras['rgb0'][mask_cas_s.squeeze() == 0], target_s[mask_cas_s.squeeze() == 0])

                # img_loss0 = img2mse(extras['rgb0'][mask_bound.squeeze()][mask.squeeze()], target_s[mask_bound.squeeze()][mask.squeeze()])
                # if mask.squeeze().sum() != mask.squeeze().shape[0]: img_loss0 += 0.2 * img2mse(extras['rgb0'][mask_bound.squeeze()][~mask.squeeze()], target_s[mask_bound.squeeze()][~mask.squeeze()])
                # if mask_bound.squeeze().sum() != N_rand:
                #     img_loss0 += 0.2 * img2mse(extras['rgb0'][~mask_bound.squeeze()], target_s[~mask_bound.squeeze()])
            elif args.softmask:
                temp = F.softplus(render_kwargs_train['network_fn'].temp_rgb)
                img_loss0 = img2mse_softmask(extras['rgb0'], target_s, temp)
            elif args.softLpmask:
                img_loss0 = img2mse_softLpmask(extras['rgb0'], target_s, args.Lp_coef)
            else:
                img_loss0 = img2mse(extras['rgb0'], target_s)

            if not use_batching:
                if img_i in i_train:
                    loss = loss + img_loss0
                else:
                    loss = loss + 0.1 * img_loss0
            else:
                loss = loss + img_loss0

            psnr0 = mse2psnr(img_loss0)
            
            #midas loss coarse
            patch_num = 4
            
            if True:    
                depth_predict_clip = 1 / torch.where(extras['depth0'] <= 0, 0.0001*torch.ones(1), extras['depth0'])
                # depth_predict = depth_pred[:config.batch_size]
                # depth_gt = batch['dpts'][:config.batch_size]
                depth_mse = 0.0
                ssim_coarse = 0.0
                lpips_coarse = 0.0
                
                mono_dpts = torch.Tensor(mono_dpts)
                for i_patch in range(patch_num):
                    depth_predict = torch.nan_to_num(depth_predict_clip[i_patch * 16*16 : (i_patch+1) * 16*16])
                    depth_gt = torch.nan_to_num(mono_dpt_s[i_patch * 16*16 : (i_patch+1) * 16*16])  
                    
                    mask = torch.where(depth_gt>0, torch.ones(1), torch.zeros(1))
                    #if mask[mask == 0].shape[0] > 128:
                        #continue
                    
                    img_pred = extras['rgb0'][i_patch * 16*16 : (i_patch+1) * 16*16].reshape(1, 16, 16, 3)
                    img_gt = target_s[i_patch * 16*16 : (i_patch+1) * 16*16].reshape(1, 16, 16, 3)
                    ssim_coarse += ssim(img_pred, img_gt, data_range = 1, size_average = False)
                    
                    gt, pred = img_gt, img_pred
                    gt, pred = (gt-0.5)*2., (pred-0.5)*2.
                    gt, pred = gt.permute([0,3,1,2]),pred.permute([0,3,1,2])
                    lpips_coarse += lpips_fn(pred, gt).reshape(-1)
                        
                    depth_min = torch.where(depth_gt > 0, depth_gt, torch.ones(1)*10**5).min()
                    depth_max = depth_gt.max()
                    depth_gt = mask * (depth_gt - depth_min) / (depth_max - depth_min + 0.0001)

                    depth_min = torch.where(mask * depth_predict>0, depth_predict, torch.ones(1)*10**5).min()
                    depth_max = (mask * depth_predict).max()
                    depth_predict = mask * (depth_predict - depth_min) / (depth_max - depth_min + 0.0001)

                    alpha = (depth_predict - depth_gt).mean()
                    depth_mse += ((depth_gt - depth_predict + alpha)**2).mean() / patch_num / 2
                mono_depth_mses0 = depth_mse
                ssim_coarse = ssim_coarse/4
                lpips_coarse = lpips_coarse/4
            else:
                mono_depth_mses0 = 0.0
            #mono_depth_mses0 = 0.0
            loss += 0.001 * mono_depth_mses0
            loss -= 0.005 * ssim_coarse[0]
            loss += 0.005 * lpips_coarse[0]
            
            #ipdb.set_trace()
                
            if args.with_depth_loss:
                if args.hardmask or args.softmask:
                    depth_loss0 = img2mse(extras['depth0'][mask_cas_s.squeeze() == 1]/far, depth_cas_s[mask_cas_s.squeeze() == 1]/far)
                    # if mask_cas_s.squeeze().sum() != N_rand: depth_loss0 += args.hardmask_coef * img2mse(extras['depth0'][mask_cas_s.squeeze() == 0]/far, depth_cas_s[mask_cas_s.squeeze() == 0]/far)
                    
                    # depth_scale = render_kwargs_train['network_fn'].depth_scale
                    # depth_loss0 = img2mse_depth(extras['depth0'][mask_cas_s.squeeze() == 1], depth_cas_s[mask_cas_s.squeeze() == 1], depth_scale)
                    # if mask_cas_s.squeeze().sum() != N_rand: depth_loss0 += args.hardmask_coef * img2mse_depth(extras['depth0'][mask_cas_s.squeeze() == 0], depth_cas_s[mask_cas_s.squeeze() == 0], depth_scale)

                    # loss += img2mse(extras['depth0'][mask_bound.squeeze()][mask.squeeze()], depth_cas_s[mask_bound.squeeze()][mask.squeeze()]) #if rand_num == 1 else 0
                    # if mask.squeeze().sum() != mask.squeeze().shape[0]: loss += 0.2 * img2mse(extras['depth0'][mask_bound.squeeze()][~mask.squeeze()], depth_cas_s[mask_bound.squeeze()][~mask.squeeze()])
                    # if mask_bound.squeeze().sum() != N_rand:
                    #     loss += 0.2 * img2mse(extras['depth0'][~mask_bound.squeeze()], depth_cas_s[~mask_bound.squeeze()])
                elif args.softmask:
                    # if args.dataset_type == 'blender':
                    #     mask = depth_cas_s > near
                    #     loss += img2mse_softmask(extras['depth0'][mask], depth_cas_s[mask], 1.1)
                    # else:
                    # loss += img2mse_softmask(extras['depth0']/far, depth_cas_s/far, 1.1)
                    temp = F.softplus(render_kwargs_train['network_fn'].temp_depth)
                    depth_loss0 = img2mse_depth_softmask(extras['depth0']/far, depth_cas_s/far, temp)
                elif args.softLpmask:
                    depth_loss0 = img2mse_softLpmask(extras['depth0']/far, depth_cas_s/far, args.Lp_coef)
                elif args.with_depth_norm:
                    if mask_cas_s.squeeze().sum() != N_rand: depth_cas_s[mask_cas_s.squeeze() == 0] = 0
                    depth_loss0 = img2mse(extras['depth0']/far, depth_cas_s/far)
                else:
                    # if args.dataset_type == 'blender':
                    #     mask = depth_cas_s > 0
                    #     loss += img2mse(extras['depth0'][mask], depth_cas_s[mask])
                    # else:
                    if mask_cas_s.squeeze().sum() != N_rand: depth_cas_s[mask_cas_s.squeeze() == 0] = 0
                    depth_loss0 = img2mse(extras['depth0'], depth_cas_s)


                if not use_batching:
                    if img_i in i_train:
                        loss = loss + depth_loss0
                    else:
                        loss = loss + 0.1 * depth_loss0
                else:
                    loss = loss + depth_loss0
            else:
                depth_loss0 = 0
        # add tensorboard
        writer.add_scalar('train_rgb_mse_loss_fine', img2mse(rgb, target_s), i)
        writer.add_scalar('train_psnr_fine', mse2psnr(img2mse(rgb, target_s)), i)
        if 'rgb0' in extras:
            writer.add_scalar('train_rgb_mse_loss_coarse', img2mse(extras['rgb0'], target_s), i)
            writer.add_scalar('train_psnr_coarse', mse2psnr(img2mse(extras['rgb0'], target_s)), i)
        if args.softmask:
            writer.add_scalar('train_rgb_softmask_loss_fine', img_loss, i)
            writer.add_scalar('train_psnr_softmask_fine', psnr, i)
            if 'rgb0' in extras:
                writer.add_scalar('train_rgb_softmask_loss_coarse', img_loss0, i)
                writer.add_scalar('train_psnr_softmask_coarse', psnr0, i)
        elif args.hardmask:
            writer.add_scalar('train_rgb_hardmask_loss_fine', img_loss, i)
            writer.add_scalar('train_psnr_hardmask_fine', psnr, i)
            if 'rgb0' in extras:
                writer.add_scalar('train_rgb_hardmask_loss_coarse', img_loss0, i)
                writer.add_scalar('train_psnr_hardmask_coarse', psnr0, i)

        if args.with_depth_loss:
            writer.add_scalar('train_depth_mse_loss_fine', img2mse(extras['depth0']/far, extras['depth0']/far), i)
            if 'rgb0' in extras:
                writer.add_scalar('train_depth_mse_loss_coarse', img2mse(extras['depth0']/far, depth_cas_s/far), i)
            if args.softmask:
                writer.add_scalar('train_depth_softmask_loss_fine', depth_loss, i)
                if 'rgb0' in extras:
                    writer.add_scalar('train_depth_softmask_loss_coarse', depth_loss0, i)
            elif args.hardmask:
                writer.add_scalar('train_depth_hardmask_loss_fine', depth_loss, i)
                if 'rgb0' in extras:
                    writer.add_scalar('train_depth_hardmask_loss_coarse', depth_loss0, i)


        # img2mse_tensor = lambda x, y: (x - y) ** 2
        # img2mse_softmask_tensor = lambda x, y, coef: (torch.exp((x - y) ** 2) - 1) * (x - y) ** 2
        #
        # fine_rgb_mse_loss = img2mse_tensor(rgb, target_s).detach().cpu().flatten().tolist()
        # fine_rgb_softmask_loss = img2mse_softmask_tensor(rgb, target_s, 1.1).detach().cpu().flatten().tolist()
        # fine_depth_mse_loss = img2mse_tensor(depth_pred, depth_cas_s).detach().cpu().flatten().tolist()
        # fine_depth_softmask_loss = img2mse_softmask_tensor(depth_pred/far, depth_cas_s/far, 1.1).detach().cpu().flatten().tolist()
        #
        # coarse_rgb_mse_loss = img2mse_tensor(extras['rgb0'], target_s).detach().cpu().flatten().tolist()
        # coarse_rgb_softmask_loss = img2mse_softmask_tensor(extras['rgb0'], target_s, 1.1).detach().cpu().flatten().tolist()
        # coarse_depth_mse_loss = img2mse_tensor(extras['depth0'], depth_cas_s).detach().cpu().flatten().tolist()
        # coarse_depth_softmask_loss = img2mse_softmask_tensor(extras['depth0']/far, depth_cas_s/far, 1.1).detach().cpu().flatten().tolist()

        # dir_name = os.path.join(basedir, expname, f'fig_hist')
        # os.makedirs(dir_name, exist_ok=True)
        # if global_step % 100 == 0:
        #     plt.hist(fine_rgb_mse_loss)
        #     plt.savefig(f"{dir_name}/fine_rgb_mse_loss_{global_step}.png")
        #     plt.close()
        #     plt.hist(fine_rgb_softmask_loss)
        #     plt.savefig(f"{dir_name}/fine_rgb_softmask_loss_{global_step}.png")
        #     plt.close()
            # plt.hist(fine_depth_mse_loss)
            # plt.savefig(f"fig_hist/fine_depth_mse_loss_{global_step}.png")
            # plt.close()
            # plt.hist(fine_depth_softmask_loss)
            # plt.savefig(f"fig_hist/fine_depth_softmask_loss_{global_step}.png")
            # plt.close()

            # plt.hist(coarse_rgb_mse_loss)
            # plt.savefig(f"{dir_name}/coarse_rgb_mse_loss_{global_step}.png")
            # plt.close()
            # plt.hist(coarse_rgb_softmask_loss)
            # plt.savefig(f"{dir_name}/coarse_rgb_softmask_loss_{global_step}.png")
            # plt.close()
            # plt.hist(coarse_depth_mse_loss)
            # plt.savefig(f"fig_hist/coarse_depth_mse_loss_{global_step}.png")
            # plt.close()
            # plt.hist(coarse_depth_softmask_loss)
            # plt.savefig(f"fig_hist/coarse_depth_softmask_loss_{global_step}.png")
            # plt.close()

        loss.backward()
        torch.nn.utils.clip_grad_value_(grad_vars, 0.1)
        
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.N_importance > 0:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        # if i%args.i_video==0 and i > 0:
        #     # Turn on testing mode
        #     with torch.no_grad():
        #         rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
        #     print('Done, saving', rgbs.shape, disps.shape)
        #     moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
        #     imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
        #     imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            #i_test = [i_train[0]]
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                rgbs_test, disps_test, accs = render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images_label[i_test], savedir=testsavedir)
                # rgbs_test, disps_test = render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

            rgbs = rgbs_test
            disps = disps_test
            #ipdb.set_trace()
            test_loss = img2mse(torch.Tensor(rgbs).cpu(), images_label[i_test])
            test_psnr = mse2psnr(test_loss).cpu().numpy()
            test_psnr = float(test_psnr)
            
            
            #test_redefine_psnr = img2psnr_redefine(torch.Tensor(rgbs), images_label[i_test])
            test_ssim, test_msssim = img2ssim(torch.Tensor(rgbs), torch.Tensor(images_label[i_test]))
            
            gt = torch.Tensor(images_label[i_test])
            pred = torch.Tensor(rgbs)
            gt, pred = (gt-0.5)*2., (pred-0.5)*2.
            gt, pred = gt.permute([0,3,1,2]),pred.permute([0,3,1,2])
            test_lpips = lpips_fn(gt, pred).mean()
            
            for ind in range(disps.shape[0]):
                aa=lky_visualize_depth(1/disps[ind], accs[ind])
                save_img_u8(aa, f"{args.basedir}/{args.expname}/depth_{ind:03d}.png")
            
            if args.dataset_type == 'dtu':
                
                mask = depths[i_test] > 0
                test_psnr_mask = img2psnr_mask(torch.Tensor(rgbs), images_label[i_test], torch.Tensor(mask)).cpu().numpy()
                test_psnr_mask = float(test_psnr_mask)
                test_ssim_mask, test_msssim_m = img2ssim(torch.Tensor(rgbs), images_label[i_test], torch.Tensor(mask))
                #ipdb.set_trace()
                
                mask = np.array([mask,mask,mask]).transpose(1,0,2,3)
                gt = gt*mask + (1-mask)
                pred = pred*mask + (1-mask)
                test_lpips_mask = lpips_fn(gt.float(), pred.float()).mean()
                
                with open(f"{args.basedir}/{args.expname}/metrics.txt","w",) as metric_file:
                    metric_file.write(f"PSNR: {test_psnr_mask}\n")
                    metric_file.write(f"SSIM: {test_ssim_mask}\n")
                    metric_file.write(f"LPIPS: {test_lpips_mask}")
                    
            else:
                with open(f"{args.basedir}/{args.expname}/metrics.txt","w",) as metric_file:
                    metric_file.write(f"PSNR: {test_psnr}\n")
                    metric_file.write(f"SSIM: {test_ssim}\n")
                    metric_file.write(f"LPIPS: {test_lpips}")       
                    
            if args.i_testset==1:
                return
            

            
            test_loss = img2mse(torch.Tensor(rgbs_test), torch.Tensor(images_label[i_test]))
            test_psnr = mse2psnr(test_loss)
            print(os.path.basename(args.datadir), ' test psnr: ', test_psnr)

            writer.add_scalar('test_rgb_mse_loss', test_loss, i)
            writer.add_scalar('test_psnr', test_psnr, i)

            dir_name = os.path.join('Softmask', args.dataset_type, os.path.basename(args.datadir), f'iter_{i}')
            os.makedirs(dir_name, exist_ok=True)
            img2mse_tensor = lambda x, y: (x - y) ** 2
            loss_mse = img2mse_tensor(torch.Tensor(rgbs_test), torch.Tensor(images_label[i_test]))
            for index in range(rgbs_test.shape[0]):
                mask = masks_cas[i_test[index]].reshape(-1)
                print('iter ', i, ' ', index, ' mask loss: ', loss_mse[index].view(-1,3)[mask].mean(), ' non mask loss: ', loss_mse[index].view(-1,3)[~mask].mean())

            if args.dataset_type == 'dtu':
                depths_test = torch.Tensor(depths[i_test]).to(device)
                rgbs_test_mask = torch.Tensor(rgbs_test)
                # images_mask = torch.Tensor(images[i_test])
                images_mask = torch.Tensor(images_label[i_test])
                test_mask_loss = []
                test_mask_psnr = []
                for j in range(rgbs_test.shape[0]):
                    mask = depths_test[j]==0
                    test_mask_loss.append(img2mse(rgbs_test_mask[j][~mask], images_mask[j][~mask]))
                    test_mask_psnr.append(mse2psnr(img2mse(rgbs_test_mask[j][~mask], images_mask[j][~mask])))

                # test_loss = img2mse(rgbs_test_mask, images_mask)
                # test_psnr = mse2psnr(test_loss)
                print(os.path.basename(args.datadir), ' test mask psnr: ', sum(test_mask_psnr)/len(test_mask_psnr))

                writer.add_scalar('test_rgb_mask_mse_loss', sum(test_mask_loss)/len(test_mask_loss), i)
                writer.add_scalar('test_mask_psnr', sum(test_mask_psnr)/len(test_mask_psnr), i)
            

            # for index in range(rgbs_test.shape[0]):
            #     ret = torch.topk(loss_mse[index].mean(-1).reshape(-1), k=100000, dim=0)
            #     res = torch.zeros_like(loss_mse[index].mean(-1).reshape(-1))
            #     res.scatter_(0, ret.indices, ret.values)
            #     mask = res != 0
                
            #     test_img = torch.zeros_like(torch.Tensor(images_label)[i_test[index]].reshape(-1,3))
            #     test_img_mask = torch.zeros_like(torch.Tensor(images_label)[i_test[index],:,:,0].reshape(-1))
            #     test_img_hardmask = torch.zeros_like(torch.Tensor(images_label)[i_test[index],:,:,0].reshape(-1))
            #     test_img[mask] = torch.Tensor(images_label)[i_test[index]].reshape(-1,3)[mask]
            #     test_img_mask[mask] = 255
            #     test_img_hardmask[masks_cas[i_test[index]].reshape(-1)] = 255
            #     imageio.imwrite(f'{dir_name}/softmask_{i_test[index]:04d}_30per.png', to8b(test_img_mask.reshape(512, 640).cpu().numpy()))
            #     # imageio.imwrite(f'{dir_name}/{index}_softmask_img.png', to8b(test_img.reshape(*images_label[i_test[index]].shape).cpu().numpy()))
            #     # imageio.imwrite(f'{dir_name}/{index}_softmask_img_gt.png', to8b(images_label[i_test[index]]))
            #     # imageio.imwrite(f'{dir_name}/{index}_hardmask.png', to8b(test_img_hardmask.reshape(512, 640).cpu().numpy()))
            #     mask = masks_cas[i_test[index]].reshape(-1)
            #     print('iter ', i, ' ', index, ' mask loss: ', loss_mse[index].view(-1,3)[mask].mean(), ' non mask loss: ', loss_mse[index].view(-1,3)[~mask].mean())

            # for index in range(rgbs_test.shape[0]):
            #     ret = torch.topk(loss_mse[index].mean(-1).reshape(-1), k=200000, dim=0)
            #     res = torch.zeros_like(loss_mse[index].mean(-1).reshape(-1))
            #     res.scatter_(0, ret.indices, ret.values)
            #     mask = res != 0
                
            #     test_img = torch.zeros_like(torch.Tensor(images_label)[i_test[index]].reshape(-1,3))
            #     test_img_mask = torch.zeros_like(torch.Tensor(images_label)[i_test[index],:,:,0].reshape(-1))
            #     test_img[mask] = torch.Tensor(images_label)[i_test[index]].reshape(-1,3)[mask]
            #     test_img_mask[mask] = 255
            #     imageio.imwrite(f'{dir_name}/softmask_{i_test[index]:04d}_60per.png', to8b(test_img_mask.reshape(512, 640).cpu().numpy()))
            #     # imageio.imwrite(f'{dir_name}/{index}_softmask_img_60.png', to8b(test_img.reshape(*images_label[i_test[index]].shape).cpu().numpy()))

            # for index in range(rgbs_test.shape[0]):
            #     ret = torch.topk(loss_mse[index].mean(-1).reshape(-1), k=30000, dim=0)
            #     res = torch.zeros_like(loss_mse[index].mean(-1).reshape(-1))
            #     res.scatter_(0, ret.indices, ret.values)
            #     mask = res != 0
                
            #     test_img = torch.zeros_like(torch.Tensor(images_label)[i_test[index]].reshape(-1,3))
            #     test_img_mask = torch.zeros_like(torch.Tensor(images_label)[i_test[index],:,:,0].reshape(-1))
            #     test_img[mask] = torch.Tensor(images_label)[i_test[index]].reshape(-1,3)[mask]
            #     test_img_mask[mask] = 255
            #     imageio.imwrite(f'{dir_name}/softmask_{i_test[index]:04d}_10per.png', to8b(test_img_mask.reshape(512, 640).cpu().numpy()))
            #     # imageio.imwrite(f'{dir_name}/{index}_softmask_img_10.png', to8b(test_img.reshape(*images_label[i_test[index]].shape).cpu().numpy()))

            # for index in range(rgbs_test.shape[0]):
            #     ret = torch.topk(loss_mse[index].mean(-1).reshape(-1), k=300000, dim=0)
            #     res = torch.zeros_like(loss_mse[index].mean(-1).reshape(-1))
            #     res.scatter_(0, ret.indices, ret.values)
            #     mask = res != 0
                
            #     test_img = torch.zeros_like(torch.Tensor(images_label)[i_test[index]].reshape(-1,3))
            #     test_img_mask = torch.zeros_like(torch.Tensor(images_label)[i_test[index],:,:,0].reshape(-1))
            #     test_img[mask] = torch.Tensor(images_label)[i_test[index]].reshape(-1,3)[mask]
            #     test_img_mask[mask] = 255
            #     imageio.imwrite(f'{dir_name}/softmask_{i_test[index]:04d}_80per.png', to8b(test_img_mask.reshape(512, 640).cpu().numpy()))
            #     # imageio.imwrite(f'{dir_name}/{index}_softmask_img_10.png', to8b(test_img.reshape(*images_label[i_test[index]].shape).cpu().numpy()))




            # img2mse_tensor = lambda x, y: (x - y) ** 2
            # img2mse_softmask_tensor = lambda x, y: (torch.exp((x - y) ** 2)) * (x - y) ** 2 / torch.sum(torch.exp((x - y) ** 2)) * torch.sum(torch.exp((x - x) ** 2))
            
            # loss_softmask = img2mse_softmask_tensor(torch.Tensor(rgbs_test), torch.Tensor(images[i_test]))
            # loss_mse = img2mse_tensor(torch.Tensor(rgbs_test), torch.Tensor(images[i_test]))
            
            # dir_name = os.path.join(basedir, expname, f'train_curve_iter_{i}')
            # os.makedirs(dir_name, exist_ok=True)
            
            # for test_id in range(len(i_test)):
            #     plt.imshow(np.array(loss_mse[test_id, :, :, 0].cpu()), cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0)
            #     plt.axis('off')
            #     plt.savefig(f'{dir_name}/test_{test_id}_iter_{i}_mse_loss_c_0.jpeg')
            
            #     plt.imshow(np.array(loss_mse[test_id, :, :, 1].cpu()), cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0)
            #     plt.axis('off')
            #     plt.savefig(f'{dir_name}/test_{test_id}_iter_{i}_mse_loss_c_1.jpeg')
            
            #     plt.imshow(np.array(loss_mse[test_id, :, :, 2].cpu()), cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0)
            #     plt.axis('off')
            #     plt.savefig(f'{dir_name}/test_{test_id}_iter_{i}_mse_loss_c_2.jpeg')
            
            #     plt.imshow(np.array(loss_mse[test_id, :, :].cpu()), cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0)
            #     plt.axis('off')
            #     plt.savefig(f'{dir_name}/test_{test_id}_iter_{i}_mse_3loss.jpeg')
            #
            #
            #     plt.imshow(np.array(loss_softmask[test_id, :, :, 0].cpu()), cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0)
            #     plt.axis('off')
            #     plt.savefig(f'{dir_name}/test_{test_id}_iter_{i}_softmask_loss_c_0.jpeg')
            #
            #     plt.imshow(np.array(loss_softmask[test_id, :, :, 1].cpu()), cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0)
            #     plt.axis('off')
            #     plt.savefig(f'{dir_name}/test_{test_id}_iter_{i}_softmask_loss_c_1.jpeg')
            #
            #     plt.imshow(np.array(loss_softmask[test_id, :, :, 2].cpu()), cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0)
            #     plt.axis('off')
            #     plt.savefig(f'{dir_name}/test_{test_id}_iter_{i}_softmask_loss_c_2.jpeg')
            #
            #     plt.imshow(np.array(loss_softmask[test_id, :, :].cpu()), cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0)
            #     plt.axis('off')
            #     plt.savefig(f'{dir_name}/test_{test_id}_iter_{i}_softmask_3loss.jpeg')



            # plt.colorbar()


            # plt.imshow(np.concatenate([np.concatenate([np.array(loss[0, :, :, 0].cpu()), np.array(loss_mse[0, :, :, 0].cpu())], axis=1), np.concatenate([np.array(loss[1, :, :, 0].cpu()), np.array(loss_mse[1, :, :, 0].cpu())], axis=1)]), cmap='hot', interpolation='nearest', vmin = 0.0, vmax = 1.0)
            #
            # dir_name = os.path.join(basedir, expname, 'fig_heatmap')
            # os.makedirs(dir_name, exist_ok=True)
            # plt.savefig(f'{dir_name}/{i}.jpeg')
            # plt.imshow(np.concatenate([np.array(loss[0, :, :, 0].cpu()), np.array(loss_mse[0, :, :, 0].cpu())], axis=1), cmap='hot', interpolation='nearest')
            # plt.show()
            # plt.imshow(np.array(loss_mse[0, :, :, 0].cpu()), cmap='hot', interpolation='nearest')
            # plt.show()


        #ipdb.set_trace()
        if i%args.i_print==0:
            # tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            print(f"[TRAIN] Iter:{i}  Loss:{loss.item():0.4f}  PSNR:{psnr.item():0.4f}  Midas_coarse:{mono_depth_mses0:0.4f}  Midas_fine:{mono_depth_mses:0.4f}")
            print(f'img_mse_fine:{img_loss:0.4f}  img_mse_coarse:{img_loss0:0.4f}')
            print(f'depth_mse_fine:{depth_loss:0.4f}  depth_mse_coarse:{depth_loss0:0.4f}')
            print(f'ssim_fine:{ssim_fine.item():0.4f}  ssim_coarse:{ssim_coarse.item():0.4f}')
            print(f'lpips_fine:{lpips_fine.item():0.4f}  lpips_coarse:{lpips_coarse.item():0.4f}')
            
            #if img_loss0 > 0.15:
                #ipdb.set_trace()        
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
