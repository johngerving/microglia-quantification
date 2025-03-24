# The new config inherits a base config to highlight the necessary modification
_base_ = '../detr/detr_r50_8xb2-150e_coco.py'

# custom_imports = dict(imports=['mmdet.engine.hooks.find_iou'], allow_failed_imports=False)
# custom_hooks = [
#     dict(type='FindIoU', name='find_iou')
# ]

# Modify dataset related settings
# custom_imports = dict(imports=['mmdet.engine.runner.custom_runner', 'mmdet.engine.hooks.custom_logger_hook'], allow_failed_imports=False)
# runner_type = 'CustomRunner'
# custom_hooks = [
#     dict(type='CustomLoggerHook')        
# ]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        max_keep_ckpts=5
    )
)

dataset_type = 'CocoDataset'
data_root = '/workspace/copy-dataset/'
classes = ('microglia', )
backend_args = None

metainfo=dict(classes=classes, palette=[200,20,60])

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
        type='CocoMetric',
        ann_file=data_root + 'test/_annotations.coco.json',
        metric='bbox',
        backend_args=backend_args)

val_evaluator = dict(
        type='CocoMetric',
        ann_file=data_root + 'valid/_annotations.coco.json',
        metric='bbox',
        backend_args=backend_args)

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(num_classes=1))


# 15 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=150, val_interval=1)
