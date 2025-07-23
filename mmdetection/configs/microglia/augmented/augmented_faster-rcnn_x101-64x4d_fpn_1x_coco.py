# The new config inherits a base config to highlight the necessary modification
_base_ = '../../faster_rcnn/faster-rcnn_x101-64x4d_fpn_1x_coco.py'

dataset_type = 'CocoDataset'
data_root = '/workspace/dataset-augmented/'
classes = ('activated', 'non-activated')
num_classes = len(classes)
interval = 1
max_keep_ckpts = 1
max_epochs = 200
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
        save_begin=1,
        max_keep_ckpts=max_keep_ckpts,
        save_best=save_best)
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'microglia',
            'group': 'faster-rcnn-augmented'
         })
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')




train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        metainfo=metainfo
    )
)
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        metainfo=metainfo
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        metainfo=metainfo
    )
)

test_evaluator = dict(
        ann_file=data_root + 'test/_annotations.coco.json')
val_evaluator = dict(
        ann_file=data_root + 'valid/_annotations.coco.json')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001))

auto_scale_lr = dict(base_batch_size=8)

# learning policy
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)


# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes)))
