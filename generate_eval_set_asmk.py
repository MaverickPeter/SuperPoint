#!/usr/bin/env python
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2018
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Daniel DeTone (ddetone)
#                       Tomasz Malisiewicz (tmalisiewicz)
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%


import argparse
import glob
import numpy as np
import os
import time
from tqdm import tqdm
import cv2
import torch
import pickle
from datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader, get_pointcloud_with_image_loader
from dsift import DsiftExtractor


# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)
    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return semi, desc


class SuperPointFrontend(object):
  """ Wrapper around pytorch net to help with pre and post image processing. """
  def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
               cuda=False):
    self.name = 'SuperPoint'
    self.cuda = cuda
    self.nms_dist = nms_dist
    self.conf_thresh = conf_thresh
    self.nn_thresh = nn_thresh # L2 descriptor distance for good match.
    self.cell = 8 # Size of each output cell. Keep this fixed.
    self.border_remove = 4 # Remove points this close to the border.

    # Load the network in inference mode.
    self.net = SuperPointNet()
    if cuda:
      # Train on GPU, deploy on GPU.
      self.net.load_state_dict(torch.load(weights_path))
      self.net = self.net.cuda()
    else:
      # Train on GPU, deploy on CPU.
      self.net.load_state_dict(torch.load(weights_path,
                               map_location=lambda storage, loc: storage))
    self.net.eval()

  def nms_fast(self, in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
  
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

  def run(self, img):
    """ Process a numpy image to extract points and descriptors.
    Input
      img - HxW numpy float32 input image in range [0,1].
    Output
      corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      desc - 256xN numpy array of corresponding unit normalized descriptors.
      heatmap - HxW numpy heatmap in range [0,1] of point confidences.
      """
    assert img.ndim == 2, 'Image must be grayscale.'
    assert img.dtype == np.float32, 'Image must be float32.'
    H, W = img.shape[0], img.shape[1]
    inp = img.copy()
    inp = (inp.reshape(1, H, W))
    inp = torch.from_numpy(inp)
    inp = torch.autograd.Variable(inp).view(1, 1, H, W)
    if self.cuda:
      inp = inp.cuda()
    # Forward pass of network.
    outs = self.net.forward(inp)
    semi, coarse_desc = outs[0], outs[1]
    # Convert pytorch -> numpy.
    semi = semi.data.cpu().numpy().squeeze()
    # --- Process points.
    dense = np.exp(semi) # Softmax.
    dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
    # Remove dustbin.
    nodust = dense[:-1, :, :]
    # Reshape to get full resolution heatmap.
    Hc = int(H / self.cell)
    Wc = int(W / self.cell)
    nodust = nodust.transpose(1, 2, 0)
    heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
    # print("heatmap",np.percentile(heatmap,99))
    xs, ys = np.where(heatmap >= self.conf_thresh) # Confidence threshold.
    if len(xs) == 0:
      return np.zeros((3, 0)), None, None
    pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist) # Apply NMS.
    inds = np.argsort(pts[2,:])
    pts = pts[:,inds[::-1]] # Sort by confidence.
    # Remove points along border.
    bord = self.border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    # --- Process descriptor.
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
      desc = np.zeros((D, 0))
    else:
      # Interpolate into descriptor map using 2D point locations.
      samp_pts = torch.from_numpy(pts[:2, :].copy())
      samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
      samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
      samp_pts = samp_pts.transpose(0, 1).contiguous()
      samp_pts = samp_pts.view(1, 1, -1, 2)
      samp_pts = samp_pts.float()
      if self.cuda:
        samp_pts = samp_pts.cuda()
      desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
      desc = desc.data.cpu().numpy().reshape(D, -1)
      desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return pts, desc, heatmap


def load_im_file_for_generate(filename):
    input_image = cv2.imread(filename)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    if input_image.shape[0] != 160:
        output = np.zeros((160,767,3))
        output[:input_image.shape[0],...] = input_image
    output = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    return output


if __name__ == '__main__':

  # Parse command line arguments.
  parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
  parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
      help='Path to pretrained weights file (default: superpoint_v1.pth).')
  parser.add_argument('--dataset_root', type=str, default='/mnt/workspace/workgroup/xxc/datasets/NCLT',
      help='dataset_root is specified (default: /mnt/workspace/workgroup/xxc/datasets/NCLT).')
  parser.add_argument('--dataset_type', type=str, default='nclt',
      help='dataset_type is specified (default: nclt).')
  parser.add_argument('--skip', type=int, default=1,
      help='Images to skip if input is movie or directory (default: 1).')
  parser.add_argument('--show_extra', action='store_true',
      help='Show extra debug outputs (default: False).')
  parser.add_argument('--use_sift', action='store_true',
      help='Use SIFT descriptors (default: False).')
  parser.add_argument('--use_dsift', action='store_true',
      help='Use Dense-SIFT descriptors (default: False).')
  parser.add_argument('--use_orb', action='store_true',
      help='Use ORB descriptors (default: False).')
  parser.add_argument('--display_scale', type=int, default=2,
      help='Factor to scale output visualization (default: 2).')
  parser.add_argument('--nms_dist', type=int, default=4,
      help='Non Maximum Suppression (NMS) distance (default: 4).')
  parser.add_argument('--conf_thresh', type=float, default=0.015,
      help='Detector confidence threshold (default: 0.015).')
  parser.add_argument('--nn_thresh', type=float, default=0.7,
      help='Descriptor matching threshold (default: 0.7).')
  parser.add_argument('--cuda', action='store_true',
      help='Use cuda GPU to speed up network processing speed (default: False)')
  parser.add_argument('--no_display', action='store_true',
      help='Do not display images to screen. Useful if running remotely (default: False).')
  opt = parser.parse_args()
  print(opt)


  if opt.dataset_type == 'nclt':
    pkl_file = "/mnt/workspace/workgroup/xxc/datasets/NCLT/mink_test_2012-02-04_2012-03-17_0.2.pickle"
  elif opt.dataset_type == 'oxford':
    pkl_file = "/mnt/workspace/workgroup/xxc/datasets/Oxford/mink_test_2019-01-11-13-24-51-radar-oxford-10k_2019-01-15-13-06-37-radar-oxford-10k_0.2.pickle"

  eval_set = EvaluationSet()
  eval_set.load(pkl_file)
  print("==> Test data loaded")

  map_set = eval_set.map_set
  query_set = eval_set.query_set
  map_positions = eval_set.get_map_positions()
  query_positions = eval_set.get_query_positions()

  print('==> Loading pre-trained network.')
  # This class runs the SuperPoint network and processes its outputs.
  fe = SuperPointFrontend(weights_path=opt.weights_path,
                          nms_dist=opt.nms_dist,
                          conf_thresh=opt.conf_thresh,
                          nn_thresh=opt.nn_thresh,
                          cuda=opt.cuda)
  print('==> Successfully loaded pre-trained network.')

  vecs = []
  imids = []
  coordx = []
  coordy = []
  strengths = []
  posex = []
  posey = []

  qvecs = []
  qimids = []
  qcoordx = []
  qcoordy = []
  qstrengths = []
  qposex = []
  qposey = []

  sift = cv2.SIFT_create()
  orb = cv2.ORB_create()
  dsift = DsiftExtractor(16,32,1)

  print('==> Computing map_set.')
  for ndx, e in tqdm(enumerate(map_set)):
    start = time.time()

    # Get a new image.
    if opt.dataset_type == 'nclt':
      img_filename = os.path.join(opt.dataset_root, e.rel_scan_filepath)
      img_filename = img_filename.replace('.bin', '.jpg')
      img_filename = img_filename.replace('velodyne_sync', 'sph')
      # img_filename = img_filename.replace('velodyne_sync', 'lb3_u_s_384/Cam1')
      sift_img = cv2.imread(img_filename)
      sift_img = cv2.cvtColor(sift_img, cv2.COLOR_BGR2GRAY)

      grayim = cv2.imread(img_filename, 0)
      interp = cv2.INTER_AREA
      grayim = cv2.resize(grayim, (grayim.shape[1], grayim.shape[0]), interpolation=interp)
      img = (grayim.astype('float32') / 255.)

    elif opt.dataset_type == 'oxford':
      img_filename = os.path.join(opt.dataset_root, e.filepaths[2])
      img_filename = img_filename.replace('mono_left_rect', 'sph')
      sift_img = cv2.imread(img_filename)
      sift_img = cv2.cvtColor(sift_img, cv2.COLOR_BGR2GRAY)

      grayim = cv2.imread(img_filename, 0)
      interp = cv2.INTER_AREA
      grayim = cv2.resize(grayim, (grayim.shape[1], grayim.shape[0]), interpolation=interp)
      img = (grayim.astype('float32') / 255.)

    opt.H = img.shape[0]
    opt.W = img.shape[1]

    if opt.use_sift:
      pts, desc = sift.detectAndCompute(sift_img,None)
      if desc.shape[0] > 1000:
        desc = desc[:1000,...]
      pts = cv2.KeyPoint_convert(pts)
      dim = 128
    elif opt.use_orb:
      pts, desc = orb.detectAndCompute(sift_img,None)
      pts = cv2.KeyPoint_convert(pts)
      dim = 32
    elif opt.use_dsift:
      desc, pts = dsift.process_image(sift_img)
      dim = 128
    else:
      # Get points and descriptors.
      pts, desc, heatmap = fe.run(img)
      pts_ascend = pts[2,:].argsort()[::-1]
      # if pts_ascend.shape[0] > 300:
      #   pts = pts[:,pts_ascend[:300]]
      #   desc = desc[:,pts_ascend[:300]]
      #   desc = desc.transpose()
      # else:
      #   desc = desc.transpose()
      desc = desc.transpose()

    if desc is not None:
      ndxs = np.ones((desc.shape[0]))
      ndxs = ndxs * ndx 
      posex.append(np.ones((1))*e.position[0])
      posey.append(np.ones((1))*e.position[1])
      vecs.append(desc)
      imids.append(ndxs)
      coordx.append(pts[0])
      coordy.append(pts[0])
      strengths.append(np.ones_like(pts[0]))
    else:
      ndxs = np.ones((1))
      ndxs = ndxs * ndx 
      posex.append(np.ones((1))*e.position[0])
      posey.append(np.ones((1))*e.position[1])
      vecs.append(np.ones((1,dim)))
      imids.append(ndxs)
      coordx.append(np.ones((1)))
      coordy.append(np.ones((1)))
      strengths.append(np.ones((1)))

  desc_dict = dict()
  desc_dict['vecs'] = np.vstack(vecs)
  desc_dict['posex'] = np.concatenate(posex)
  desc_dict['posey'] = np.concatenate(posey)
  desc_dict['imids'] = np.concatenate(imids)
  desc_dict['coordx'] = np.concatenate(coordx)
  desc_dict['coordy'] = np.concatenate(coordy)
  desc_dict['strengths'] = np.concatenate(strengths)

  print(desc_dict['vecs'].shape)
  print('==> Computing query_set.')
  for ndx, e in tqdm(enumerate(query_set)):
    start = time.time()

    # Get a new image.
    if opt.dataset_type == 'nclt':
      img_filename = os.path.join(opt.dataset_root, e.rel_scan_filepath)
      img_filename = img_filename.replace('.bin', '.jpg')
      # img_filename = img_filename.replace('velodyne_sync', 'sph')
      img_filename = img_filename.replace('velodyne_sync', 'lb3_u_s_384/Cam1')
      sift_img = cv2.imread(img_filename)
      sift_img = cv2.cvtColor(sift_img, cv2.COLOR_BGR2GRAY)

      grayim = cv2.imread(img_filename, 0)
      interp = cv2.INTER_AREA
      grayim = cv2.resize(grayim, (grayim.shape[1], grayim.shape[0]), interpolation=interp)
      img = (grayim.astype('float32') / 255.)

    elif opt.dataset_type == 'oxford':
      img_filename = os.path.join(opt.dataset_root, e.filepaths[2])
      img_filename = img_filename.replace('mono_left_rect', 'sph')
      sift_img = cv2.imread(img_filename)
      sift_img = cv2.cvtColor(sift_img, cv2.COLOR_BGR2GRAY)

      grayim = cv2.imread(img_filename, 0)
      interp = cv2.INTER_AREA
      grayim = cv2.resize(grayim, (grayim.shape[1], grayim.shape[0]), interpolation=interp)
      img = (grayim.astype('float32') / 255.)

    opt.H = img.shape[0]
    opt.W = img.shape[1]

    if opt.use_sift:
      pts, desc = sift.detectAndCompute(sift_img,None)
      if desc.shape[0] > 1000:
        desc = desc[:1000,...]
      pts = cv2.KeyPoint_convert(pts)
      dim = 128
    elif opt.use_orb:
      pts, desc = orb.detectAndCompute(sift_img,None)
      pts = cv2.KeyPoint_convert(pts)
      dim = 32
    elif opt.use_dsift:
      desc, pts = dsift.process_image(sift_img)
      dim = 128
    else:
      # Get points and descriptors.
      pts, desc, heatmap = fe.run(img)
      pts_ascend = pts[2,:].argsort()[::-1]
      # if pts_ascend.shape[0] > 300:
      #   pts = pts[:,pts_ascend[:300]]
      #   desc = desc[:,pts_ascend[:300]]
      #   desc = desc.transpose()
      # else:
      #   desc = desc.transpose()
      desc = desc.transpose()

    if desc is not None:
      ndxs = np.ones((desc.shape[0]))
      ndxs = ndxs * ndx
      qposex.append(np.ones((1))*e.position[0])
      qposey.append(np.ones((1))*e.position[1])
      qvecs.append(desc)
      qimids.append(ndxs)
      qcoordx.append(pts[0])
      qcoordy.append(pts[0])
      qstrengths.append(np.ones_like(pts[0]))
    else:
      ndxs = np.ones((1))
      ndxs = ndxs * ndx 
      posex.append(np.ones((1))*e.position[0])
      posey.append(np.ones((1))*e.position[1])
      vecs.append(np.ones((1,dim)))
      imids.append(ndxs)
      coordx.append(np.ones((1)))
      coordy.append(np.ones((1)))
      strengths.append(np.ones((1)))

  desc_dict['qvecs'] = np.vstack(qvecs)
  desc_dict['qposex'] = np.concatenate(qposex)
  desc_dict['qposey'] = np.concatenate(qposey)
  desc_dict['qimids'] = np.concatenate(qimids)
  desc_dict['qcoordx'] = np.concatenate(qcoordx)
  desc_dict['qcoordy'] = np.concatenate(qcoordy)
  desc_dict['qstrengths'] = np.concatenate(qstrengths)

  if opt.use_sift:
    with open('./'+opt.dataset_type+'_test_sift.pickle', 'wb') as handle:
      pickle.dump(desc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
  elif opt.use_orb:
    with open('./'+opt.dataset_type+'_test_orb.pickle', 'wb') as handle:
      pickle.dump(desc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
  elif opt.use_dsift:
    with open('./'+opt.dataset_type+'_test_dsift.pickle', 'wb') as handle:
      pickle.dump(desc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
  else:
    with open('./'+opt.dataset_type+'_test.pickle', 'wb') as handle:
      pickle.dump(desc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

  print('==> Finshed Demo.')
