import os
from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch_scatter
import spconv.pytorch as spconv
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.mixssc.utils.ssc_loss import sem_scal_loss, CE_ssc_loss, KL_sep, geo_scal_loss, BCE_ssc_loss
# kitti_360_class_frequencies = np.array(
#     [
#         2264087502,
#         20098728,
#         104972,
#         96297,
#         1149426,
#         4051087,
#         125103,
#         105540713,
#         16292249,
#         45297267,
#         14454132,
#         110397082,
#         6766219,
#         295883213,
#         50037503,
#         1561069,
#         406330,
#         30516166,
#         1950115,
#     ]
# )

class LMSCNet_SS(nn.Module):
    def __init__(self,
                 class_num=None,
                 input_dimensions=None,
                 out_scale=None,
                 gamma=0,
                 alpha=0.5,
                 point_cloud_range=None,
                 ):


        super().__init__()
        self.out_scale=out_scale
        self.nbr_classes = class_num
        self.gamma = gamma
        self.alpha = alpha
        self.input_dimensions = input_dimensions  # Grid dimensions should be (W, H, D).. z or height being axis 1
        f = self.input_dimensions[1]
        spatial_shape=[256,256,32]
        coors_range_xyz = [[point_cloud_range[0], point_cloud_range[3]],
                           [point_cloud_range[1], point_cloud_range[4]],
                           [point_cloud_range[2], point_cloud_range[5]]]
        self.voxelizes = Voxelization(coors_range_xyz, spatial_shape)
        self.pool = nn.MaxPool2d(2)  # [F=2; S=2; P=0; D=1]

        self.pooling = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.class_frequencies_level1 =  np.array([5.41773033e09, 4.03113667e08])
        self.class_weights_level_1 = torch.from_numpy(
            1 / np.log(self.class_frequencies_level1 + 0.001)
        )
        # self.class_frequencies_level1 =  np.array([kitti_360_class_frequencies[0], np.sum(kitti_360_class_frequencies)])
        # print("empty: " + str(self.class_frequencies_level1[0]) + " occupied: " + str(self.class_frequencies_level1[1]))
        # self.class_weights_level_1 = torch.from_numpy(
        #     1 / np.log(self.class_frequencies_level1 + 0.001)
        # )

        self.Encoder_block1 = nn.Sequential(
        nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
        nn.ReLU()
        )

        self.Encoder_block2 = nn.Sequential(
        nn.MaxPool2d(2),
        nn.Conv2d(f, int(f*1.5), kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(int(f*1.5), int(f*1.5), kernel_size=3, padding=1, stride=1),
        nn.ReLU()
        )

        self.Encoder_block3 = nn.Sequential(
        nn.MaxPool2d(2),
        nn.Conv2d(int(f*1.5), int(f*2), kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(int(f*2), int(f*2), kernel_size=3, padding=1, stride=1),
        nn.ReLU()
        )

        self.Encoder_block4 = nn.Sequential(
        nn.MaxPool2d(2),
        nn.Conv2d(int(f*2), int(f*2.5), kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(int(f*2.5), int(f*2.5), kernel_size=3, padding=1, stride=1),
        nn.ReLU()
        )

        # Treatment output 1:8
        self.conv_out_scale_1_8 = nn.Conv2d(int(f*2.5), int(f/8), kernel_size=3, padding=1, stride=1)
        self.deconv_1_8__1_2    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=4, padding=0, stride=4)
        self.deconv_1_8__1_1    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=8, padding=0, stride=8)

        # Treatment output 1:4
        if self.out_scale=="1_4" or self.out_scale=="1_2" or self.out_scale=="1_1":
          self.deconv1_8          = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=6, padding=2, stride=2)
          self.conv1_4            = nn.Conv2d(int(f*2) + int(f/8), int(f*2), kernel_size=3, padding=1, stride=1)
          self.conv_out_scale_1_4 = nn.Conv2d(int(f*2), int(f/4), kernel_size=3, padding=1, stride=1)
          self.deconv_1_4__1_1    = nn.ConvTranspose2d(int(f/4), int(f/4), kernel_size=4, padding=0, stride=4)

        # Treatment output 1:2
        if self.out_scale=="1_2" or self.out_scale=="1_1":
          self.deconv1_4          = nn.ConvTranspose2d(int(f/4), int(f/4), kernel_size=6, padding=2, stride=2)
          self.conv1_2            = nn.Conv2d(int(f*1.5) + int(f/4) + int(f/8), int(f*1.5), kernel_size=3, padding=1, stride=1)
          self.conv_out_scale_1_2 = nn.Conv2d(int(f*1.5), int(f/2), kernel_size=3, padding=1, stride=1)

        # Treatment output 1:1
        if self.out_scale=="1_1":
          self.deconv1_2          = nn.ConvTranspose2d(int(f/2), int(f/2), kernel_size=6, padding=2, stride=2)
          self.conv1_1            = nn.Conv2d(int(f/8) + int(f/4) + int(f/2) + int(f), f, kernel_size=3, padding=1, stride=1)
        
        if self.out_scale=="1_1":
          self.seg_head_1_1 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
        elif self.out_scale=="1_2":
          self.seg_head_1_2 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3],out_feat=128)
        elif self.out_scale=="1_4":
          self.seg_head_1_4 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
        elif self.out_scale=="1_8":
          self.seg_head_1_8 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])

    def forward(self, img_metas, target):

        # input = x['3D_OCCUPANCY']  # Input to LMSCNet model is 3D occupancy big scale (1:1) [bs, 1, W, H, D]
        # input = torch.squeeze(input, dim=1).permute(0, 2, 1, 3)  # Reshaping to the right way for 2D convs [bs, H, W, D]

        # print(input.shape) [4, 32, 256, 256]
        ones = torch.ones_like(target).to(target.device)
        target = torch.where(torch.logical_or(target==255, target==0), target, ones) # [1, 128, 128, 16]
        # target[target==255] = 2
        device = target.device
        len_queue = 1
        # img_metas = [each[len_queue-1] for each in img_metas] # [dict(), dict(), ...] 
        points = []
        batch_idx = []
        tensor = torch.ones((1,), dtype=torch.long).to(device)
        for i, img_meta in enumerate(img_metas):
            pc = torch.from_numpy(img_meta['points']).float().to(device)
            points.append(pc)
            batch_idx.append(tensor.new_full((pc.shape[0],), i))
        points, batch_idx = torch.cat(points), torch.cat(batch_idx)

        # depth =  torch.from_numpy(img_metas[0]["pseudo_pc"]).reshape(256, 256, 32).unsqueeze(0) # [1, 256, 256, 32]

        input = self.voxelizes(points, batch_idx).permute(0, 3, 1, 2)

        # Encoder block
        _skip_1_1 = self.Encoder_block1(input)
        # print('_skip_1_1.shape', _skip_1_1.shape)  # [1, 32, 256, 256]
        _skip_1_2 = self.Encoder_block2(_skip_1_1)
        # print('_skip_1_2.shape', _skip_1_2.shape)  # [1, 48, 128, 128]
        _skip_1_4 = self.Encoder_block3(_skip_1_2) 
        # print('_skip_1_4.shape', _skip_1_4.shape)  # [1, 64, 64, 64]
        _skip_1_8 = self.Encoder_block4(_skip_1_4) 
        # print('_skip_1_8.shape', _skip_1_8.shape)  # [1, 80, 32, 32]

        # Out 1_8
        out_scale_1_8__2D = self.conv_out_scale_1_8(_skip_1_8)

        # print('out_scale_1_8__2D.shape', out_scale_1_8__2D.shape)  # [1, 4, 32, 32]

        if self.out_scale=="1_8":
          out_scale_1_8__3D = self.seg_head_1_8(out_scale_1_8__2D) # [1, 20, 16, 128, 128]
          out_scale_1_8__3D = out_scale_1_8__3D.permute(0, 1, 3, 4, 2) # [1, 20, 128, 128, 16]
          return out_scale_1_8__3D

        elif self.out_scale=="1_4":
          # Out 1_4
          out = self.deconv1_8(out_scale_1_8__2D)
          out = torch.cat((out, _skip_1_4), 1)
          out = F.relu(self.conv1_4(out))
          out_scale_1_4__2D = self.conv_out_scale_1_4(out)

          out_scale_1_4__3D = self.seg_head_1_4(out_scale_1_4__2D) # [1, 20, 16, 128, 128]
          out_scale_1_4__3D = out_scale_1_4__3D.permute(0, 1, 3, 4, 2) # [1, 20, 128, 128, 16]
          return out_scale_1_4__3D

        elif self.out_scale=="1_2":
          # Out 1_4
          out = self.deconv1_8(out_scale_1_8__2D)
          out = torch.cat((out, _skip_1_4), 1)
          out = F.relu(self.conv1_4(out))
          out_scale_1_4__2D = self.conv_out_scale_1_4(out)

          # Out 1_2
          out = self.deconv1_4(out_scale_1_4__2D)
          out = torch.cat((out, _skip_1_2, self.deconv_1_8__1_2(out_scale_1_8__2D)), 1)
          out = F.relu(self.conv1_2(out)) # torch.Size([1, 48, 128, 128])
          out_scale_1_2__2D = self.conv_out_scale_1_2(out) # torch.Size([1, 16, 128, 128])

          out_scale_1_2__3D, pts_feat = self.seg_head_1_2(out_scale_1_2__2D) # [1, 20, 16, 128, 128]
          out_scale_1_2__3D = out_scale_1_2__3D.permute(0, 1, 3, 4, 2) # [1, 20, 128, 128, 16]
          pts_feat = pts_feat.permute(0, 1, 3, 4, 2) # [1, 20, 128, 128, 16]
          return out_scale_1_2__3D, pts_feat

        elif self.out_scale=="1_1":
          # Out 1_4
          out = self.deconv1_8(out_scale_1_8__2D)
        #   print('out.shape', out.shape)  # [1, 4, 64, 64]
          out = torch.cat((out, _skip_1_4), 1)
          out = F.relu(self.conv1_4(out))
          out_scale_1_4__2D = self.conv_out_scale_1_4(out)
          # print('out_scale_1_4__2D.shape', out_scale_1_4__2D.shape)  # [1, 8, 64, 64]

          # Out 1_2
          out = self.deconv1_4(out_scale_1_4__2D)
        #   print('out.shape', out.shape)  # [1, 8, 128, 128]
          out = torch.cat((out, _skip_1_2, self.deconv_1_8__1_2(out_scale_1_8__2D)), 1)
          out = F.relu(self.conv1_2(out)) # torch.Size([1, 48, 128, 128])
          out_scale_1_2__2D = self.conv_out_scale_1_2(out) # torch.Size([1, 16, 128, 128])
          # print('out_scale_1_2__2D.shape', out_scale_1_2__2D.shape)  # [1, 16, 128, 128]

          # Out 1_1
          out = self.deconv1_2(out_scale_1_2__2D)
          out = torch.cat((out, _skip_1_1, self.deconv_1_4__1_1(out_scale_1_4__2D), self.deconv_1_8__1_1(out_scale_1_8__2D)), 1)
          out_scale_1_1__2D = F.relu(self.conv1_1(out)) # [bs, 32, 256, 256]

          out_scale_1_1__3D = self.seg_head_1_1(out_scale_1_1__2D)
          # Take back to [W, H, D] axis order
          out_scale_1_1__3D = out_scale_1_1__3D.permute(0, 1, 3, 4, 2)  # [bs, C, H, W, D] -> [bs, C, W, H, D]
          return out_scale_1_1__3D


    def pack(self, array):
        """ convert a boolean array into a bitwise array. """
        array = array.reshape((-1))

        #compressing bit flags.
        # yapf: disable
        compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
        # yapf: enable

        return np.array(compressed, dtype=np.uint8)

class SegmentationHead(nn.Module):
  '''
  3D Segmentation heads to retrieve semantic segmentation at each scale.
  Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
  '''
  def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list,out_feat = None):
    super().__init__()

    # First convolution
    self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

    # ASPP Block
    self.conv_list = dilations_conv_list
    self.conv1 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.conv2 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.relu = nn.ReLU(inplace=True)

    if out_feat is not None:
       self.out_conv = nn.Sequential(
            nn.Conv3d(planes, out_feat, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True)
        )
    else:
       self.out_conv = None
    # Convolution for output
    self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

  def forward(self, x_in):

    # Dimension exapension
    x_in = x_in[:, None, :, :, :]

    # Convolution to go from inplanes to planes features...
    x_in = self.relu(self.conv0(x_in))

    y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
    for i in range(1, len(self.conv_list)):
      y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
    x_in = self.relu(y + x_in)  # modified
    if self.out_conv is not None:
      x_f = self.out_conv(x_in)
      x_in = self.conv_classes(x_in)
      return x_in,x_f
    else:
      x_in = self.conv_classes(x_in)    
      return x_in

class Voxelization(nn.Module):
    def __init__(self, coors_range_xyz, spatial_shape, out_channels=1):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = coors_range_xyz

        self.PPmodel = nn.Sequential(
            nn.Linear(9, 16),
            nn.LayerNorm(16),
            nn.ReLU(True),
            nn.Linear(16, 16)
        )
        self.soft_model = SoftModel(16, out_channels)

    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        return idx.long()

    def prepare_input(self, point, grid_ind, inv_idx):
        pc_mean = torch_scatter.scatter_mean(point[:, :3], inv_idx, dim=0)[inv_idx]
        nor_pc = point[:, :3] - pc_mean

        coors_range_xyz = torch.Tensor(self.coors_range_xyz)
        cur_grid_size = torch.Tensor(self.spatial_shape)
        crop_range = coors_range_xyz[:, 1] - coors_range_xyz[:, 0]
        intervals = (crop_range / cur_grid_size).to(point.device)
        voxel_centers = grid_ind * intervals + coors_range_xyz[:, 0].to(point.device)
        center_to_point = point[:, :3] - voxel_centers

        pc_feature = torch.cat((point, nor_pc, center_to_point), dim=1)
        return pc_feature

    def filter_pc(self, pc, batch_idx):
        def mask_op(data, x_min, x_max):
            mask = (data > x_min) & (data < x_max)
            return mask
        mask_x = mask_op(pc[:, 0], self.coors_range_xyz[0][0] + 0.0001, self.coors_range_xyz[0][1] - 0.0001)
        mask_y = mask_op(pc[:, 1], self.coors_range_xyz[1][0] + 0.0001, self.coors_range_xyz[1][1] - 0.0001)
        mask_z = mask_op(pc[:, 2], self.coors_range_xyz[2][0] + 0.0001, self.coors_range_xyz[2][1] - 0.0001)
        mask = mask_x & mask_y & mask_z
        filter_pc = pc[mask]
        fiter_batch_idx = batch_idx[mask]
        if filter_pc.shape[0] < 10:
            filter_pc = torch.ones((10, 3), dtype=pc.dtype).to(pc.device)
            filter_pc = filter_pc * torch.rand_like(filter_pc)
            fiter_batch_idx = torch.zeros(10, dtype=torch.long).to(pc.device)
        return filter_pc, fiter_batch_idx

    def forward(self, pc, batch_idx):
        pc, batch_idx = self.filter_pc(pc, batch_idx)
        xidx = self.sparse_quantize(pc[:, 0], self.coors_range_xyz[0], self.spatial_shape[0])
        yidx = self.sparse_quantize(pc[:, 1], self.coors_range_xyz[1], self.spatial_shape[1])
        zidx = self.sparse_quantize(pc[:, 2], self.coors_range_xyz[2], self.spatial_shape[2])

        bxyz_indx = torch.stack([batch_idx, xidx, yidx, zidx], dim=-1).long()
        unq, unq_inv, _ = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)

        pt_fea = self.prepare_input(pc, bxyz_indx[:, 1:], unq_inv)
        pt_fea = self.PPmodel(pt_fea)
        features = torch_scatter.scatter_mean(pt_fea, unq_inv, dim=0)
        input_tensor = spconv.SparseConvTensor(features, unq.int(), spatial_shape=self.spatial_shape, batch_size=batch_idx.max()+1)
        
        return self.soft_model(input_tensor).squeeze(1)
class SoftModel(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv_block = SFE(c_in, c_in, "svpfe")
        self.proj_block = SGFE(input_channels=c_in, output_channels=c_out, reduce_channels=c_in, name="proj")

    def forward(self, input_tensor):
        conv_output = self.conv_block(input_tensor)
        x = self.proj_block(conv_output.features, input_coords=conv_output.indices)
        conv_output = conv_output.replace_feature(x.sigmoid())
        return conv_output.dense()

class BasicBlock(spconv.SparseModule):
    def __init__(self, C_in, C_out, indice_key):
        super(BasicBlock, self).__init__()
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(C_in, C_out, 1, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(C_out),
        )
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(C_in, C_out, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(C_out),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(C_out, C_out, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(C_out)
        )
        self.relu2 = spconv.SparseSequential(
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        identity = self.layers_in(x)
        out = self.layers(x)
        output = spconv.SparseConvTensor(sum([i.features for i in [identity, out]]),
                                         out.indices, out.spatial_shape, out.batch_size)
        output.indice_dict = out.indice_dict
        output.grid = out.grid
        return self.relu2(output)


def make_layers_sp(C_in, C_out, blocks, indice_key):
    layers = []
    layers.append(BasicBlock(C_in, C_out, indice_key))
    for _ in range(1, blocks):
        layers.append(BasicBlock(C_out, C_out, indice_key))
    return spconv.SparseSequential(*layers)


def scatter(x, idx, method, dim=0):
    if method == "max":
        return torch_scatter.scatter_max(x, idx, dim=dim)[0]
    elif method == "mean":
        return torch_scatter.scatter_mean(x, idx, dim=dim)
    elif method == "sum":
        return torch_scatter.scatter_add(x, idx, dim=dim)
    else:
        print("unknown method")
        exit(-1)


class SFE(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, layer_name, layer_num=2):
        super().__init__()
        self.spconv_layers = make_layers_sp(in_channels, out_channels, layer_num, layer_name)

    def forward(self, inputs):
        conv_features = self.spconv_layers(inputs)
        return conv_features


class SGFE(nn.Module):
    def __init__(self, input_channels, output_channels, reduce_channels, name, p_scale=[2, 4, 6, 8]):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.name = name

        self.feature_reduce = nn.Linear(input_channels, reduce_channels)
        self.pooling_scale = p_scale
        self.fc_list = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for _, _ in enumerate(self.pooling_scale):
            self.fc_list.append(nn.Sequential(
            nn.Linear(reduce_channels, reduce_channels//2),
            nn.ReLU(),
            ))
            self.fcs.append(nn.Sequential(nn.Linear(reduce_channels//2, reduce_channels//2)))
        self.scale_selection = nn.Sequential(
            nn.Linear(len(self.pooling_scale) * reduce_channels//2,
                                       reduce_channels),nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(reduce_channels//2, reduce_channels//2, bias=False),
                                nn.ReLU(inplace=False))
        self.out_fc = nn.Linear(reduce_channels//2, reduce_channels, bias=False)
        self.linear_output = nn.Sequential(
            nn.Linear(2 * reduce_channels, reduce_channels, bias=False),
            nn.ReLU(),
            nn.Linear(reduce_channels, output_channels),
        )

    def forward(self, input_data, input_coords):
        reduced_feature = F.relu(self.feature_reduce(input_data))
        output_list = [reduced_feature]
        for j, ps in enumerate(self.pooling_scale):
            index = torch.cat([input_coords[:, 0].unsqueeze(-1),
                              (input_coords[:, 1:] // ps).int()], dim=1)
            _, unq_inv = torch.unique(index, return_inverse=True, dim=0)
            fkm = scatter(reduced_feature, unq_inv, method="mean", dim=0)
            att = self.fc_list[j](fkm)[unq_inv]
            output_list.append(att)
        scale_features = torch.stack(output_list[1:], dim=1)
        feat_S = scale_features.sum(1)
        feat_Z = self.fc(feat_S)
        attention_vectors = [fc(feat_Z) for fc in self.fcs]
        attention_vectors = torch.sigmoid(torch.stack(attention_vectors, dim=1))
        scale_features = self.out_fc(torch.sum(scale_features * attention_vectors, dim=1))

        output_f = torch.cat([reduced_feature, scale_features], dim=1)
        proj = self.linear_output(output_f)
        
        return proj
