# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for visualizing things."""
from pytorch_msssim import ssim, ms_ssim

import matplotlib.cm as cm
import numpy as np
from PIL import Image
import ipdb

def img2psnr_mask(x,y,mask):
    '''
    we redefine the PSNR function,
    [previous]
    average MSE -> PSNR(average MSE)
    
    [new]
    average PSNR(each image pair)
    '''
    
    image_num = x.size(0)
    mses = ((x-y)**2).mean(-1)
    qw_mses = ((x-y)**2).mean(-1)
    mses_sum = (mses*mask).reshape(image_num, -1).sum(-1)
    
    mses = mses_sum /mask.reshape(image_num, -1).sum(-1)
    psnrs = [mse2psnr(mse) for mse in mses]
    psnr = torch.stack(psnrs).mean()
    return psnr

def img2ssim(x,y, mask=None):
    if mask is not None:
        x = mask.unsqueeze(-1)*x
        y = mask.unsqueeze(-1)*y
     
    x = x.permute(0,3,1,2)
    y = y.permute(0,3,1,2)
    ssim_ = ssim(x,y, data_range=1)
    ms_ssim_ = ms_ssim(x,y, data_range=1)
    return ssim_, ms_ssim_


def open_file(pth, mode='r'):
  return open(pth, mode=mode)

def save_img_u8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  #ipdb.set_trace()
  with open_file(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')


def weighted_percentile(x, w, ps, assume_sorted=False):
  """Compute the weighted percentile(s) of a single vector."""
  x = x.reshape([-1])
  w = w.reshape([-1])
  if not assume_sorted:
    sortidx = np.argsort(x)
    x, w = x[sortidx], w[sortidx]
  acc_w = np.cumsum(w)
  return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)


def matte(vis, acc, dark=0.8, light=1.0, width=8):
  """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
  bg_mask = np.logical_xor(
      (np.arange(acc.shape[0]) % (2 * width) // width)[:, None],
      (np.arange(acc.shape[1]) % (2 * width) // width)[None, :])
  bg = np.where(bg_mask, light, dark)
  bg = np.ones(bg.shape)
  return vis * acc[:, :, None] + (bg * (1 - acc))[:, :, None]

def visualize_cmap(value,
                   weight,
                   colormap,
                   lo=None,
                   hi=None,
                   percentile=99.,
                   curve_fn=lambda x: x,
                   modulus=None,
                   matte_background=True):
  """Visualize a 1D image and a 1D weighting according to some colormap.

  Args:
    value: A 1D image.
    weight: A weight map, in [0, 1].
    colormap: A colormap function.
    lo: The lower bound to use when rendering, if None then use a percentile.
    hi: The upper bound to use when rendering, if None then use a percentile.
    percentile: What percentile of the value map to crop to when automatically
      generating `lo` and `hi`. Depends on `weight` as well as `value'.
    curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
      before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
    modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
      `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
    matte_background: If True, matte the image over a checkerboard.

  Returns:
    A colormap rendering.
  """
  # Identify the values that bound the middle of `value' according to `weight`.
  lo_auto, hi_auto = weighted_percentile(
      value, weight, [50 - percentile / 2, 50 + percentile / 2])

  # If `lo` or `hi` are None, use the automatically-computed bounds above.
  eps = np.finfo(np.float32).eps
  lo = lo or (lo_auto - eps)
  hi = hi or (hi_auto + eps)

  # Curve all values.
  value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

  # Wrap the values around if requested.
  if modulus:
    value = np.mod(value, modulus) / modulus
  else:
    # Otherwise, just scale to [0, 1].
    value = np.nan_to_num(
        np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))

  if colormap:
    colorized = colormap(value)[:, :, :3]
  else:
    assert len(value.shape) == 3 and value.shape[-1] == 3
    colorized = value

  return matte(colorized, weight) if matte_background else colorized


def lky_visualize_depth(x, acc, lo=None, hi=None):
  """Visualizes depth maps."""

  depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
  return visualize_cmap(
      x, acc, cm.get_cmap('turbo'), curve_fn=depth_curve_fn, lo=lo, hi=hi)
