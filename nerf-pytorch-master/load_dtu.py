import numpy as np
import os, imageio
from scipy.spatial.transform import Rotation as R
import re, cv2

def normalize(x):
  """Normalization helper function."""
  return x / np.linalg.norm(x)

def pad_poses(p):
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[Ellipsis, :1, :4].shape)
  return np.concatenate([p[Ellipsis, :3, :4], bottom], axis=-2)


def unpad_poses(p):
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[Ellipsis, :3, :4]


def recenter_poses(poses):
  """Recenter poses around the origin."""
  cam2world = poses_avg(poses)
  new_poses = np.linalg.inv(pad_poses(cam2world)) @ pad_poses(poses)
  return unpad_poses(new_poses), cam2world

def decenter_poses(poses, center):
  """Recenter poses around the origin."""
  
  new_poses = pad_poses(center) @ pad_poses(poses)
  return unpad_poses(new_poses)

def poses_avg(poses):
  """New pose using average position, z-axis, and up vector of input poses."""
  
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world

def rescale_poses(poses):
    """Rescales camera poses according to maximum x/y/z value."""
    s = np.max(np.abs(poses[:, :3, -1]))
    out = np.copy(poses)
    out[:, :3, -1] /= s
    return out,s
  
def focus_pt_fn(poses):
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt

def viewmatrix(lookdir, up, position, subtract_position=False):
  """Construct lookat view matrix."""
  
  vec2 = normalize((lookdir - position) if subtract_position else lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m

def generate_spiral_path_dtu(poses, n_frames=120, n_rots=2, zrate=.5, perc=60):
  """Calculates a forward facing spiral path for rendering for DTU."""

    
  # Get radii for spiral path using 60th percentile of camera positions.
  positions = poses[:, :3, 3]
  radii = np.percentile(np.abs(positions), perc, 0)
  radii = np.concatenate([radii, [1.]])

  # Generate poses for spiral path.
  render_poses = []
  cam2world = poses_avg(poses)
  up = poses[:, :3, 1].mean(0)
  z_axis = focus_pt_fn(poses)
  for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
    t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
    position = cam2world @ t
    render_poses.append(viewmatrix(z_axis, up, position, True))
  render_poses = np.stack(render_poses, axis=0)
  return render_poses


def gen_render_path(c2ws, N_views=30):
    N = len(c2ws)
    rotvec, positions = [], []
    rotvec_inteplat, positions_inteplat = [], []
    weight = np.linspace(1.0, .0, N_views//N, endpoint=False).reshape(-1, 1)
    for i in range(N):
        r = R.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0])>180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
        positions.append(c2ws[i, :3, 3:].reshape(1, 3))

        if i:
            rotvec_inteplat.append(weight * rotvec[i - 1] + (1.0 - weight) * rotvec[i])
            positions_inteplat.append(weight * positions[i - 1] + (1.0 - weight) * positions[i])

    rotvec_inteplat.append(weight * rotvec[-1] + (1.0 - weight) * rotvec[0])
    positions_inteplat.append(weight * positions[-1] + (1.0 - weight) * positions[0])

    c2ws_render = []
    angles_inteplat, positions_inteplat = np.concatenate(rotvec_inteplat), np.concatenate(positions_inteplat)
    for rotvec, position in zip(angles_inteplat, positions_inteplat):
        c2w = np.eye(4)
        c2w[:3, :3] = R.from_euler('xyz', rotvec, degrees=True).as_matrix()
        c2w[:3, 3:] = position.reshape(3, 1)
        c2ws_render.append(c2w.copy())
    c2ws_render = np.stack(c2ws_render)
    return c2ws_render


def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0]) #* self.scale_factor
    depth_max = depth_min + float(lines[11].split()[1]) * 192 * 1.06 #* self.scale_factor
    return intrinsics, extrinsics, [depth_min, depth_max]

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

def read_depth(filename):
    depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
    depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_NEAREST)  # (600, 800)
    depth_h = depth_h[44:556, 80:720]  # (512, 640)

    return depth_h

def load_dtu_data(basedir, train_view_num = 16):

    root_dir = os.path.dirname(basedir)
    scan = os.path.basename(basedir)
    light_idx = 3

    # opencv2blender = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    imgs = []
    poses = []
    depths_cas = []
    depths = []
    bds = []
    for vid in range(49):
        img_filename = os.path.join(root_dir, f'Rectified/{scan}_train/rect_{vid + 1:03d}_{light_idx}_r5000.png')
        proj_mat_filename = os.path.join(root_dir, f'Depths/Cameras/train/{vid:08d}_cam.txt')
        depth_filename = os.path.join(root_dir, f'Depths/{scan}/depth_map_{vid:04d}.pfm')
        # depth_filename_cas = f'Depths_infer/{scan}/depth_map_{vid:04d}.pfm'
        depth_filename_cas = f'nerf_dtu_data_depth/{scan}/depth_{vid:04d}.pfm'
        intrinsic, w2c, near_far = read_cam_file(proj_mat_filename)
        intrinsic[:2] *= 4
        imgs += [imageio.imread(img_filename).astype(np.float32) / 255.]
        c2w = np.linalg.inv(w2c)
        c2w[:3, 3] *= 1/200
        pose = np.concatenate([c2w[:, :1], -c2w[:, 1:2], -c2w[:, 2:3], c2w[:, 3:4]], axis=-1)
        poses += [pose]
        depths_cas += [np.array(read_pfm(depth_filename_cas)[0], dtype=np.float32)]
        depths += [read_depth(depth_filename)/200]

        bds += [near_far[0]/200, near_far[1]/200]

    imgs = np.stack(imgs, axis=0)
    poses = np.stack(poses, axis=0)
    bds = np.stack(bds, axis=0)
    depths_cas = np.stack(depths_cas, axis=0)
    depths = np.stack(depths, axis=0)

    print('Loaded', basedir, bds.min(), bds.max())

    import ipdb
    #ipdb.set_trace()
    temp_poses = poses.copy()
    temp_poses[:,:,1:3] = -temp_poses[:,:,1:3]
    temp_poses, center = recenter_poses(temp_poses)
    temp_poses, scale = rescale_poses(temp_poses)
    #render_poses = gen_render_path(poses, N_views=120)
    render_poses = generate_spiral_path_dtu(temp_poses[:,:3,:4], 60)
    #render_poses = np.load('poses.npy')
    render_poses[:,:3,-1] = render_poses[:,:3,-1] * scale
    render_poses = decenter_poses(render_poses, center)
    render_poses[:,:,1:3] = -render_poses[:,:,1:3]
    
    render_poses = np.array(render_poses).astype(np.float32)
    print(intrinsic)
    H, W = imgs[0].shape[:2]
    focal = float(intrinsic[0,0])
    hwf = [H, W, focal]

    return imgs, poses, bds, render_poses, hwf, depths_cas, depths


#     poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
#     print('Loaded', basedir, bds.min(), bds.max())
#
#     # Correct rotation matrix ordering and move variable dim to axis 0
#     poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
#     poses = np.moveaxis(poses, -1, 0).astype(np.float32)
#     imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
#     images = imgs
#     bds = np.moveaxis(bds, -1, 0).astype(np.float32)
#
#     # Rescale if bd_factor is provided
#     sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
#     poses[:,:3,3] *= sc
#     bds *= sc
#
#     if recenter:
#         poses = recenter_poses(poses)
#
#     if spherify:
#         poses, render_poses, bds = spherify_poses(poses, bds)
#
#     else:
#
#         c2w = poses_avg(poses)
#         print('recentered', c2w.shape)
#         print(c2w[:3,:4])
#
#         ## Get spiral
#         # Get average pose
#         up = normalize(poses[:, :3, 1].sum(0))
#
#         # Find a reasonable "focus depth" for this dataset
#         close_depth, inf_depth = bds.min()*.9, bds.max()*5.
#         dt = .75
#         mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
#         focal = mean_dz
#
#         # Get radii for spiral path
#         shrink_factor = .8
#         zdelta = close_depth * .2
#         tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
#         rads = np.percentile(np.abs(tt), 90, 0)
#         c2w_path = c2w
#         N_views = 120
#         N_rots = 2
#         if path_zflat:
# #             zloc = np.percentile(tt, 10, 0)[2]
#             zloc = -close_depth * .1
#             c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
#             rads[2] = 0.
#             N_rots = 1
#             N_views/=2
#
#         # Generate poses for spiral path
#         render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
#
#
#     render_poses = np.array(render_poses).astype(np.float32)
#
#     c2w = poses_avg(poses)
#     print('Data:')
#     print(poses.shape, images.shape, bds.shape)
#
#     dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
#     i_test = np.argmin(dists)
#     print('HOLDOUT view is', i_test)
#
#     images = images.astype(np.float32)
#     poses = poses.astype(np.float32)
#
#     return images, poses, bds, render_poses, i_test

