_base_ = '../../_base_/default_runtime.py'

# window_size = [[2, 25], [2, 25], [2, 25], [1, 25]]
# val 83.56@15 | test 83.93

# window_size = [[16, 25], [8, 25], [4, 25], [2, 25]]
# val 83.04@15 | test 83.76

# window_size = [[64, 25], [32, 25], [16, 25], [8, 25]]
# val 81.95@15 | test 82.45

# window_size = [[8, 25], [8, 25], [8, 25], [8, 25]]
# val 82.89@16 | test 83.23

# window_size = [[2, 25], [2, 25], [2, 25], [8, 25]]
# val 83.67@16 | test 84.04
# val 81.36@16 | test 81.74 <- drop_rate 0.5, attn_drop_rate 0.0
# val 83.30@14 | test 83.65 <- drop_rate 0.0, attn_drop_rate 0.5
# val @ | test <- data batchnorm

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='StoneMamba', graph_cfg=dict(layout='nturgb+d', mode='spatial'),
        dim=80, in_dim=32, depths=[1, 3, 8, 4], window_size=[[2, 25], [2, 25], [2, 25], [8, 25]], mlp_ratio=4, num_heads=[2, 4, 8, 16],
        drop_path_rate=0.2, drop_rate=0.0, attn_drop_rate=0.0,),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=640))

dataset_type = 'PoseDataset'
ann_file = 'data/skeleton/ntu60_3d.pkl'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=64),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(
        type='UniformSampleFrames', clip_len=64, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(
        type='UniformSampleFrames', clip_len=64, num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='xsub_train')))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='xsub_val',
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='xsub_val',
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=16, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=16,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True),
    clip_grad=dict(max_norm=40, norm_type=2))
default_hooks = dict(checkpoint=dict(interval=1), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (4 GPUs) x (32 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)
# find_unused_parameters = True
