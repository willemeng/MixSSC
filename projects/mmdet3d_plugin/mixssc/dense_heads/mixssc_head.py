import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.cnn.bricks.transformer import build_positional_encoding
from projects.mmdet3d_plugin.mixssc.utils.header import Header
from projects.mmdet3d_plugin.mixssc.utils.ssc_loss import sem_scal_loss, KL_sep, geo_scal_loss, CE_ssc_loss,BCE_ssc_loss
from projects.mmdet3d_plugin.mixssc.utils.lovasz_loss import lovasz_softmax
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.mixssc.detectors.lmscnet import LMSCNet_SS
from mmdet3d.models import builder
@HEADS.register_module()
class MixSSCHead(nn.Module):
    def __init__(
        self,
        *args,
        bev_h,
        bev_w,
        bev_z,
        cross_transformer,
        self_transformer,
        positional_encoding,
        embed_dims,
        view_transformer,
        img_bev_encoder_backbone,
        img_bev_encoder_neck,
        CE_ssc_loss=True,
        ups=[2,4,8,16],
        geo_scal_loss=True,
        sem_scal_loss=True,
        save_flag = False,
        dataset = 'semantickitti',
        **kwargs
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w 
        self.bev_z = bev_z
        self.real_w = 51.2
        self.real_h = 51.2
        self.embed_dims = embed_dims
        self.bev_embed = nn.Embedding((self.bev_h) * (self.bev_w) * (self.bev_z), self.embed_dims)
        # self.mask_embed = nn.Embedding(1, self.embed_dims)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        if view_transformer is not None:
            self.view_transformer = builder.build_neck(view_transformer)
        else:
            self.view_transformer = None
        
        if img_bev_encoder_backbone is not None:
            self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        else:
            self.img_bev_encoder_backbone = torch.nn.Identity()
        
        if img_bev_encoder_neck is not None:
            self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

        self.fusion_layer = nn.Sequential(nn.Conv3d(self.embed_dims*len(ups), self.embed_dims, kernel_size=1),
                                          nn.BatchNorm3d(self.embed_dims, momentum=0.1, ),
                                        #   nn.GroupNorm(16,self.embed_dims ),
                                          nn.ReLU(inplace=False),
                                        #   nn.Conv3d(self.embed_dims, self.embed_dims, kernel_size=3,stride=1,padding=1),
                                        #   nn.BatchNorm3d(self.embed_dims, momentum=0.1, ),
                                        #   nn.ReLU(inplace=False),
                                        #   ScConv(self.embed_dims)
                                          # Process(self.embed_dims, nn.BatchNorm3d, 0.1, dilations=[1, 2, 3]),
                                          )

        self.decoder_input_projs = nn.ModuleList()
        self.up_scale = nn.ModuleList()
        self.ups = ups
        for i in ups:
            self.decoder_input_projs.append(
                nn.Conv3d(192, self.embed_dims, kernel_size=1)
                # Process(self.embed_dims, nn.BatchNorm3d, 0.1, dilations=[1, 2, 3]),
                )
            self.up_scale.append(nn.Upsample(scale_factor=i, mode='trilinear', align_corners=True))

        self.cross_transformer = build_transformer(cross_transformer)
        self.self_transformer = build_transformer(self_transformer)
        # self.class_names =  [ "empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
        #                     "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
        # self.class_weights = torch.from_numpy(np.array([0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557, 
        #                                                 0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786]))
        if dataset == 'semantickitti':
            self.class_names =  [ "empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                                "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
            self.class_weights = torch.from_numpy(np.array([0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557, 0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786]))
        elif dataset == 'kitti360':
            self.class_names =  ['empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'road',
         'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'terrain',
         'pole', 'traffic-sign', 'other-structure', 'other-object']
            self.class_weights = torch.from_numpy(np.array([0.464, 0.595, 0.865, 0.871, 0.717, 0.657, 0.852, 0.541, 0.602, 0.567, 0.607, 0.540, 0.636, 0.513, 0.564, 0.701, 0.774, 0.580, 0.690]))
        self.n_classes = len(self.class_names)
        # self.class_frequencies_level1 =  np.array([5.41773033e09, 4.03113667e08])
        # self.class_weights_level_1 = torch.from_numpy(
        #     1 / np.log(self.class_frequencies_level1 + 0.001)
        # )
        self.header = Header(self.n_classes, nn.BatchNorm3d, feature=self.embed_dims)

        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.save_flag = save_flag

    
    def lss(self, mlvl_feats, img_metas):
        rots, trans, intrins, post_rots, post_trans, bda = img_metas[0]['lss_input']['img_inputs'][1:7]
        device = mlvl_feats[0].device
        B,N = mlvl_feats[0].shape[0:2]
        rots = rots.unsqueeze(1).repeat(1, N, 1, 1).to(device)
        trans = trans.unsqueeze(1).repeat(1, N, 1).to(device)
        intrins = intrins.unsqueeze(1).repeat(1, N, 1, 1).to(device)
        post_rots = post_rots.unsqueeze(1).repeat(1, N, 1, 1).to(device)
        post_trans = post_trans.unsqueeze(1).repeat(1, N, 1).to(device)
        bda = bda.repeat(1, N, 1, 1).to(device)
        mlp_input = self.view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
        voxel, depth_prob = self.view_transformer([mlvl_feats[0]] + geo_inputs)
        return voxel, depth_prob
    def forward(self, mlvl_feats, img_metas, target):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
        Returns:
            ssc_logit (Tensor): Outputs from the segmentation head.
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype


        # ===============Forward-Backward Mixture====================

        # Forward projection
        img_voxel, depth_prob = self.lss(mlvl_feats,img_metas)
        proposal = img_metas[0]['proposal']
        proposal = proposal.reshape(self.bev_h, self.bev_w, self.bev_z)
        unmasked_idx = np.asarray(np.where(proposal.reshape(-1) > 0)).astype(np.int32)
        masked_idx = np.asarray(np.where(proposal.reshape(-1) == 0)).astype(np.int32)
        bev_queries = self.bev_embed.weight.to(dtype)  # [128*128*16, dim]
        bev_pos_cross_attn = self.positional_encoding(
            torch.zeros((bs, 512, 512), device=bev_queries.device).to(dtype)).to(dtype)  # [1, dim, 128*4, 128*4]
        bev_pos_self_attn = self.positional_encoding(
            torch.zeros((bs, 512, 512), device=bev_queries.device).to(dtype)).to(dtype)  # [1, dim, 128*4, 128*4]
        vox_coords, ref_3d = self.get_ref_3d()
        # Backward projection
        seed_feats = self.cross_transformer.get_vox_features(
            mlvl_feats[1:],
            bev_queries,
            self.bev_h,
            self.bev_w,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos_cross_attn,
            img_metas=img_metas,
            prev_bev=None,
        )

        # mixture
        vox_feats_flatten = img_voxel.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = seed_feats[0]
        voxel_feat = vox_feats_flatten.reshape(bs, -1, self.bev_h, self.bev_w, self.bev_z)



        # ===============Semantic-Spatial Fusion====================
        
        # ===============Semantic Aggregation Unit====================
        voxel_feats = self.img_bev_encoder_backbone(voxel_feat)
        # ===============Spatial Refining Unit ====================
        # Low-level fusion
        voxel_feats = self.img_bev_encoder_neck(voxel_feats)
        for i in range(len(self.ups)):
            voxel_feats[i] = self.decoder_input_projs[i](voxel_feats[i])
            voxel_feats[i] = self.up_scale[i](voxel_feats[i])
        lss_voxel_feat = self.fusion_layer(torch.cat(voxel_feats, dim=1))
        lss_voxel_feat += voxel_feat
        vox_feats_flatten = lss_voxel_feat.reshape(-1, self.embed_dims)
        # High-level fusion 
        vox_feats_diff = self.self_transformer.diffuse_vox_features(
            mlvl_feats[1:],
            vox_feats_flatten,
            512,
            512,
            unmasked_idx = unmasked_idx,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos_self_attn,
            img_metas=img_metas,
            prev_bev=None,
        )
        vox_feats_diff = vox_feats_diff.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims)
        input_dict = {
            "x3d": vox_feats_diff.permute(3, 0, 1, 2).unsqueeze(0),
        }
        # ===============Header====================
        out = self.header(input_dict)
        out['depth'] = depth_prob
        return out 

    def step(self, out_dict, target, img_metas, step_type):
        """Training/validation function.
        Args:
            out_dict (dict[Tensor]): Segmentation output.
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
            step_type: Train or test.
        Returns:
            loss or predictions
        """

        ssc_pred = out_dict["ssc_logit"]

        if step_type== "train":
            loss_dict = dict()
            if self.view_transformer is not None:
                depth_pred = out_dict["depth"]
                loss_depth = self.view_transformer.get_depth_loss(img_metas[0]['depths'], depth_pred)
                loss_dict['loss_depth'] = loss_depth
            class_weight = self.class_weights.type_as(target)
            if self.CE_ssc_loss:
                loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
                loss_dict['loss_ssc'] = loss_ssc

            if self.sem_scal_loss:
                loss_sem_scal = sem_scal_loss(ssc_pred, target)
                loss_dict['loss_sem_scal'] = loss_sem_scal
            if self.geo_scal_loss:
                loss_geo_scal = geo_scal_loss(ssc_pred, target)
                loss_dict['loss_geo_scal'] = loss_geo_scal
            return loss_dict

        elif step_type== "val" or "test":
            y_true = target.cpu().numpy()
            y_pred = ssc_pred.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)

            result = dict()
            result['y_pred'] = y_pred
            result['y_true'] = y_true

            if self.save_flag:
                self.save_pred(img_metas, y_pred)

            return [result]

    def training_step(self, out_dict, target, img_metas):
        """Training step.
        """
        return self.step(out_dict, target, img_metas, "train")

    def validation_step(self, out_dict, target, img_metas):
        """Validation step.
        """
        return self.step(out_dict, target, img_metas, "val")

    def get_ref_3d(self):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
            ref_3d (Array): 3D reference points
        """
        scene_size = (51.2, 51.2, 6.4)
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = self.real_h / self.bev_h

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1), idx], axis=0).astype(int).T

        # Normalize the voxels centroids in lidar cooridnates
        ref_3d = np.concatenate([(xv.reshape(1,-1)+0.5)/self.bev_h, (yv.reshape(1,-1)+0.5)/self.bev_w, (zv.reshape(1,-1)+0.5)/self.bev_z,], axis=0).astype(np.float64).T 

        return vox_coords, ref_3d

    def save_pred(self, img_metas, y_pred):
        """Save predictions for evaluations and visualizations.

        learning_map_inv: inverse of previous map
        
        0: 0    # "unlabeled/ignored"  # 1: 10   # "car"        # 2: 11   # "bicycle"       # 3: 15   # "motorcycle"     # 4: 18   # "truck" 
        5: 20   # "other-vehicle"      # 6: 30   # "person"     # 7: 31   # "bicyclist"     # 8: 32   # "motorcyclist"   # 9: 40   # "road"   
        10: 44  # "parking"            # 11: 48  # "sidewalk"   # 12: 49  # "other-ground"  # 13: 50  # "building"       # 14: 51  # "fence"          
        15: 70  # "vegetation"         # 16: 71  # "trunk"      # 17: 72  # "terrain"       # 18: 80  # "pole"           # 19: 81  # "traffic-sign"
        """

        y_pred[y_pred==10] = 44
        y_pred[y_pred==11] = 48
        y_pred[y_pred==12] = 49
        y_pred[y_pred==13] = 50
        y_pred[y_pred==14] = 51
        y_pred[y_pred==15] = 70
        y_pred[y_pred==16] = 71
        y_pred[y_pred==17] = 72
        y_pred[y_pred==18] = 80
        y_pred[y_pred==19] = 81
        y_pred[y_pred==1] = 10
        y_pred[y_pred==2] = 11
        y_pred[y_pred==3] = 15
        y_pred[y_pred==4] = 18
        y_pred[y_pred==5] = 20
        y_pred[y_pred==6] = 30
        y_pred[y_pred==7] = 31
        y_pred[y_pred==8] = 32
        y_pred[y_pred==9] = 40

        # save predictions
        pred_folder = os.path.join("./voxformer", "sequences", img_metas[0]['sequence_id'], "predictions") 
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        y_pred_bin = y_pred.astype(np.uint16)
        y_pred_bin.tofile(os.path.join(pred_folder, img_metas[0]['frame_id'] + ".label"))
