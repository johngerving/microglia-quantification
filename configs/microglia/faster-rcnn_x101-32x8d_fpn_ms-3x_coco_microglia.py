# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster-rcnn_x101-32x8d_fpn_ms-3x_coco.py'

# custom_imports = dict(imports=['mmdet.engine.hooks.find_iou'], allow_failed_imports=False)
# custom_hooks = [
#     dict(type='FindIoU', name='find_iou')
# ]

# Modify dataset related settings
custom_imports = dict(imports=['mmdet.engine.runner.custom_runner', 'mmdet.engine.hooks.custom_logger_hook'], allow_failed_imports=False)
runner_type = 'CustomRunner'
custom_hooks = [
    dict(type='CustomLoggerHook')        
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        max_keep_ckpts=5
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'microglia',
            'group': 'test'
         })
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

dataset_type = 'CocoDataset'
data_root = '/workspace/dataset/'
classes = ('microglia', )
backend_args = None

metainfo=dict(classes=classes, palette=[200,20,60])

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            data_root=data_root,
            metainfo=metainfo,
            ann_file='train/_annotations.coco.json',
            data_prefix=dict(img='train/'))))
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


# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=15, val_interval=1)
