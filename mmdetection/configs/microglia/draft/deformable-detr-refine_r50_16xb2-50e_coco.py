_base_ = '../deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'

dataset_type = 'CocoDataset'
data_root = '/workspace/dataset/'
classes = ('activated', 'non-activated')
num_classes = len(classes)
interval = 1
max_keep_ckpts = 1
save_best = 'coco/bbox_mAP'

metainfo=dict(classes=classes, palette=[200,20,60])

default_hooks = dict(
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        patience=10,
        min_delta=0.005),
    checkpoint=dict(
        type="CheckpointHook",
        interval=interval,
        save_begin=100,
        max_keep_ckpts=max_keep_ckpts,
        save_best=save_best)
)

model = dict(
    with_box_refine=True,
    bbox_head=dict(
        num_classes=num_classes,
    ),
)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')))

test_evaluator = dict(
        ann_file=data_root + 'test/_annotations.coco.json')
val_evaluator = dict(
        ann_file=data_root + 'valid/_annotations.coco.json')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

# learning policy
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=75, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]

auto_scale_lr = dict(base_batch_size=8)
