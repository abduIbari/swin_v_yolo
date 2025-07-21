vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
auto_scale_lr = dict(base_batch_size=16, enable=False)
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet.evaluation.metrics.coco_metric',
    ])
data_root = '/home/abdulbari/uni/seminar/swin_transformer/transformer_dataset/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=5, max_keep_ckpts=3, save_best='auto', type='CheckpointHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
custom_hooks = [
    dict(type='DictLoggerHook')
],
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=(
        'infant seat',
        'child seat',
        'person',
        'everyday object',
    ))
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=96,
        init_cfg=dict(
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=7,
        with_cp=False),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=2.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=4,
        stacked_convs=4,
        type='RetinaHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_input',
        in_channels=[
            192,
            384,
            768,
        ],
        num_outs=5,
        out_channels=256,
        start_level=0,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.5, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.4,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='RetinaNet')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.005, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.1, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=50,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/grayscale_wholeImage'),
        data_root=
        '/home/abdulbari/uni/seminar/swin_transformer/transformer_dataset/',
        metainfo=dict(
            classes=(
                'infant seat',
                'child seat',
                'person',
                'everyday object',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                448,
                448,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(size_divisor=32, type='Pad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/home/abdulbari/uni/seminar/swin_transformer/transformer_dataset/annotations/instances_test.json',
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        448,
        448,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(size_divisor=32, type='Pad'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=50, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=16,
    dataset=dict(
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/grayscale_wholeImage'),
        data_root=
        '/home/abdulbari/uni/seminar/swin_transformer/transformer_dataset/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=(
                'infant seat',
                'child seat',
                'person',
                'everyday object',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                448,
                448,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                border=(
                    -32,
                    -32,
                ),
                max_rotate_degree=5,
                max_shear_degree=0,
                scaling_ratio_range=(
                    0.8,
                    1.2,
                ),
                type='RandomAffine'),
            dict(
                brightness_delta=32,
                contrast_range=(
                    0.8,
                    1.2,
                ),
                hue_delta=10,
                saturation_range=(
                    0.8,
                    1.2,
                ),
                type='PhotoMetricDistortion'),
            dict(size_divisor=32, type='Pad'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        448,
        448,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        border=(
            -32,
            -32,
        ),
        max_rotate_degree=5,
        max_shear_degree=0,
        scaling_ratio_range=(
            0.8,
            1.2,
        ),
        type='RandomAffine'),
    dict(
        brightness_delta=32,
        contrast_range=(
            0.8,
            1.2,
        ),
        hue_delta=10,
        saturation_range=(
            0.8,
            1.2,
        ),
        type='PhotoMetricDistortion'),
    dict(size_divisor=32, type='Pad'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/grayscale_wholeImage'),
        data_root=
        '/home/abdulbari/uni/seminar/swin_transformer/transformer_dataset/',
        metainfo=dict(
            classes=(
                'infant seat',
                'child seat',
                'person',
                'everyday object',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                448,
                448,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(size_divisor=32, type='Pad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/home/abdulbari/uni/seminar/swin_transformer/transformer_dataset/annotations/instances_test.json',
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/final1'
