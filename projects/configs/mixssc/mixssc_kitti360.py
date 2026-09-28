work_dir = 'result/mixssc-kitti360'
_base_ = [
    '../_base_/default_runtime.py'
]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

_num_layers_cross_ = 3
_num_points_cross_ = 8
_num_layers_self_ = 2
_num_points_self_ = 8
_dim_ = 128
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 5
scale = [8,16,32]
_labels_tag_ = 'labels'
_num_cams_ = 1
_temporal_ = []
point_cloud_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]
voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_size = [voxel_x, voxel_y, voxel_z]
data_config = {
    'input_size': (376, 1408),
    'resize': (0., 0.),
    'rot': (0.0, 0.0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
    }

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * 2],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * 2],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * 2],
    'dbound': [2.0, 58.0, 0.5],
}
voxel_size = [0.2, 0.2, 0.2]
d_bound = [2.0, 58.0, 0.5]
# voxel_channels = [128, 256, 512, 1024]
voxel_channels=[64, 128, 256, 512]
voxel_num_layer = [2, 2, 2, 2]
voxel_strides = [1, 2, 2, 2]
voxel_out_indices = (0, 1, 2, 3)
voxel_out_channels = 192
_sem_scal_loss_ = True
_geo_scal_loss_ = True
_depthmodel_= 'msnet3d'
_nsweep_ = 10
_query_tag_ = 'query_iou5203_pre7712_rec6153'

model = dict(
   type='MixSSC',
   # pretrained=dict(img='/data/B221000559-XYJ/project/VoxFormer-WM/ckpts/resnet50-19c8e357.pth'),
   img_backbone=dict(
        type='CustomEfficientNet',
        arch='b7',
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=False,
        init_cfg=dict(type='Pretrained', prefix='backbone',
                      checkpoint='/data/B221000559-XYJ/project/WM-Project/VoxFormer-lss/ckpts/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth'),
    ),
   img_neck=dict(
        type='SECONDFPN',
        in_channels=[48, 80, 224, 640, 2560],
        upsample_strides=[0.25, 0.5, 1, 2, 2],
        out_channels=[128, 128, 128, 128, 128]),
   pts_bbox_head=dict(
       type='MixSSCHead',
       bev_h=128,
       bev_w=128,
       bev_z=16,
       embed_dims=_dim_,
       CE_ssc_loss=True,
       geo_scal_loss=_geo_scal_loss_,
       sem_scal_loss=_sem_scal_loss_,
       scale = scale,
       dataset = 'kitti360',
    #    save_flag = True,
        view_transformer=dict(
        type='ViewTransformerLiftSplatShootVoxel',
        numC_input=640,
        cam_channels=33,
        loss_depth_weight=1.0,
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=128,
        vp_megvii=False),
       img_bev_encoder_backbone=dict(
           type='GCViT',
           dim=64,
           mlp_ratio=3.0,
           depths=[2, 2, 2, 2],
           num_heads=[2, 4, 8, 16],
           drop_path_rate=0.2,
           out_indices=(0, 1, 2, 3),
           qkv_bias=True,
           qk_scale=None,
           drop_rate=0.,
           attn_drop_rate=0.,
           frozen_stages=-1,
           in_chans=128,
       ),
       img_bev_encoder_neck=dict(
           type='MSDeformAttnPixelDecoder3D',
           strides=[2, 4, 8, 16],
           in_channels=voxel_channels,
           feat_channels=voxel_out_channels,
           out_channels=voxel_out_channels,
        #    norm_cfg=dict(type='BN3d',  momentum=0.1, requires_grad=True),
           norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
           encoder=dict(
               type='DetrTransformerEncoder',
               num_layers=6,
               transformerlayers=dict(
                   type='BaseTransformerLayer',
                   attn_cfgs=dict(
                       type='MultiScaleDeformableAttention3D',
                       embed_dims=voxel_out_channels,
                       num_heads=8,
                       num_levels=3,
                       num_points=4,
                       im2col_step=64,
                       dropout=0.0,
                       batch_first=False,
                       norm_cfg=None,
                       init_cfg=None),
                   ffn_cfgs=dict(
                       embed_dims=voxel_out_channels),
                   feedforward_channels=voxel_out_channels * 4,
                   ffn_dropout=0.0,
                   operation_order=('self_attn', 'norm', 'ffn', 'norm')),
               init_cfg=None),
           positional_encoding=dict(
               type='SinePositionalEncoding3D',
               num_feats=voxel_out_channels // 3,
               normalize=True),
       ),
       cross_transformer=dict(
           type='PerceptionTransformer',
           rotate_prev_bev=True,
           use_shift=True,
           embed_dims=_dim_,
           num_cams = _num_cams_,
           encoder=dict(
               type='PerceptionEncoder',
               num_layers=3,
               pc_range=point_cloud_range,
               num_points_in_pillar=8,
               return_intermediate=False,
               transformerlayers=dict(
                   type='PerceptionLayer',
                   attn_cfgs=[
                       dict(
                           type='DeformCrossAttention',
                           pc_range=point_cloud_range,
                           num_cams=_num_cams_,
                           deformable_attention=dict(
                               type='MSDeformableAttention3D',
                               embed_dims=_dim_,
                               num_points=_num_points_cross_,
                               num_levels=_num_levels_),
                           embed_dims=_dim_,
                       )
                   ],
                   ffn_cfgs=dict(
                       type='FFN',
                       embed_dims=_dim_,
                       feedforward_channels=1024,
                       num_fcs=2,
                       ffn_drop=0.,
                       act_cfg=dict(type='ReLU', inplace=True),
                   ),
                   feedforward_channels=_ffn_dim_,
                   ffn_dropout=0.1,
                   operation_order=('cross_attn', 'norm', 'ffn', 'norm')))),
       self_transformer=dict(
           type='PerceptionTransformer',
           rotate_prev_bev=True,
           use_shift=True,
           embed_dims=_dim_,
           num_cams = _num_cams_,
           encoder=dict(
               type='PerceptionEncoder',
               num_layers=2,
               pc_range=point_cloud_range,
               num_points_in_pillar=8,
               return_intermediate=False,
               transformerlayers=dict(
                   type='PerceptionLayer',
                   attn_cfgs=[
                       dict(
                           type='DeformSelfAttention',
                           embed_dims=_dim_,
                           num_levels=1,
                           num_points=_num_points_self_)
                   ],
                   ffn_cfgs=dict(
                       type='FFN',
                       embed_dims=_dim_,
                       feedforward_channels=1024,
                       num_fcs=2,
                       ffn_drop=0.,
                       act_cfg=dict(type='ReLU', inplace=True),
                   ),
                   feedforward_channels=_ffn_dim_,
                   ffn_dropout=0.1,
                   operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
       positional_encoding=dict(
           type='LearnedPositionalEncoding',
           num_feats=_pos_dim_,
           row_num_embed=512,
           col_num_embed=512,
           )),
   train_cfg=dict(pts=dict(
       grid_size=[512, 512, 1],
       voxel_size=voxel_size,
       point_cloud_range=point_cloud_range,
       out_size_factor=4)))


dataset_type = 'Kitti360DatasetStageLss'
data_root = '/data/B221000559-XYJ/project/WM-Project/kitti360/'
file_client_args = dict(backend='disk')

data = dict(
   samples_per_gpu=1,
   workers_per_gpu=4,
   train=dict(
       type=dataset_type,
       split = "train",
       test_mode=False,
       data_root=data_root,
       preprocess_root=data_root + 'preprocess',
       eval_range = 51.2,
       depthmodel=_depthmodel_,
       nsweep=_nsweep_,
       temporal = _temporal_,
       labels_tag = _labels_tag_,
       query_tag = _query_tag_),
   val=dict(
       type=dataset_type,
       split = "test",
       test_mode=True,
       data_root=data_root,
       preprocess_root=data_root + 'preprocess',
       eval_range = 51.2,
       depthmodel=_depthmodel_,
       nsweep=_nsweep_,
       temporal = _temporal_,
       labels_tag = _labels_tag_,
       query_tag = _query_tag_),
   test=dict(
       type=dataset_type,
       split = "test",
       test_mode=True,
       data_root=data_root,
       preprocess_root=data_root + 'preprocess',
       eval_range = 51.2,
       depthmodel=_depthmodel_,
       nsweep=_nsweep_,
       temporal = _temporal_,
       labels_tag = _labels_tag_,
       query_tag = _query_tag_),
   shuffler_sampler=dict(type='DistributedGroupSampler'),
   nonshuffler_sampler=dict(type='DistributedSampler')
)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999),)

optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    step=[20, 25],
)
total_epochs = 30
# evaluation = dict(interval=1,start=0)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
   interval=50,
   hooks=[
       dict(type='TextLoggerHook'),
       dict(type='TensorboardLoggerHook')
   ])
# fp16 = dict(loss_scale=512.)
# checkpoint_config = None
# load_from = "/data/B221000559-XYJ/project/WM-Project/VoxFormer-lss/result/GC-SSC-seed/best_ssc_SemanticKITTI/mIoU_epoch_3.pth"
checkpoint_config = dict(interval=1)
# ,max_keep_ckpts=3,
